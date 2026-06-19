from typing import Optional

import lightning
import torch
from omegaconf import DictConfig

from model.gaussian_vqvae import GaussianVQVAE
from utils.gaussian_vqvae_utils import int_cube_root
from utils.optim import (
    get_constant_lr_scheduler,
    get_cosine_annealing_with_warmup_scheduler,
    get_linear_warmup_warmdown_scheduler,
)
from utils.pos_tokens import (
    dense_chunk_inverse_order_indices,
    pos_tokens_to_centered_coords,
)
from utils.render import GaussianScene

from .gpt import NanoGPTGaussianModel
from .gpt.rope_config import resolve_rope_layout

# disable unused arguments and arguments differ for entire script due to lightning hooks
# pylint: disable=unused-argument, W0221


class GaussianGPT(lightning.LightningModule):
    def __init__(
        self,
        model_config: DictConfig,
        training_config: DictConfig,
        vqvae: GaussianVQVAE = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["vqvae"])

        self.model_config = model_config
        self.training_config = training_config

        # if we get a vqvae use it, else we expect it to be in the config position
        if vqvae is not None:
            self.vqvae = vqvae
        else:
            # pylint: disable-next=no-value-for-parameter
            self.vqvae = GaussianVQVAE.load_from_checkpoint(
                self.model_config.vqvae.checkpoint_path
            )
        self.vqvae.eval()
        self.vqvae.requires_grad_(False)

        self.num_position_tokens = getattr(self.model_config, "num_position_tokens", 1)
        self.dense_chunks = getattr(self.model_config, "dense_chunks", False)
        chunk_shape = getattr(self.model_config, "chunk_shape", None)
        if isinstance(chunk_shape, int):
            chunk_shape = [chunk_shape] * 3
        self.chunk_shape = chunk_shape
        self.chunk_order = getattr(self.model_config, "chunk_order", "xyz")
        self.rope_basis = getattr(self.model_config, "rope_basis")
        self.rope_layout = resolve_rope_layout(
            rope_basis=self.rope_basis,
            dense_chunks=self.dense_chunks,
        )
        self.num_feature_tokens = self.vqvae.autoencoder.vq.num_tokens
        self.feature_vocab_size = int(self.vqvae.autoencoder.vq.codebook_size)
        self.position_vocab_size = int(
            getattr(self.model_config, "position_vocab_size", self.feature_vocab_size)
        )
        self.position_side_length = int_cube_root(self.position_vocab_size)
        if self.position_side_length**3 != self.position_vocab_size:
            raise ValueError("position_vocab_size must be a perfect cube (k^3).")
        self.shared_vocab = bool(getattr(self.model_config, "shared_vocab", False))
        self.feature_token_offset = 0 if self.shared_vocab else self.position_vocab_size
        self.tokens_per_latent = self.num_feature_tokens + (
            0 if self.dense_chunks else self.num_position_tokens
        )
        if self.rope_layout.is_dense and self.rope_layout.is_position:
            if self.num_feature_tokens != 1:
                raise ValueError(
                    "Dense position-based RoPE currently requires a single feature token."
                )
        if self.rope_layout.is_sparse and self.rope_layout.is_position:
            if self.num_position_tokens != 1:
                raise ValueError(
                    "Sparse position-based RoPE currently requires num_position_tokens=1."
                )
        if self.dense_chunks:
            if not self.chunk_shape or len(self.chunk_shape) != 3:
                raise ValueError("chunk_shape must be set for dense chunks.")
            dense_len = (
                int(self.chunk_shape[0] * self.chunk_shape[1] * self.chunk_shape[2])
                * self.num_feature_tokens
            )
            required_ctx = dense_len  # BOS + dense_len tokens without final feed-back
            if self.model_config.n_ctx < required_ctx:
                raise ValueError(
                    f"n_ctx={self.model_config.n_ctx} is too small for dense sequences "
                    f"(requires >= {required_ctx})."
                )
        self._dense_order_inv_cache: Optional[torch.Tensor] = None

        model = NanoGPTGaussianModel(
            self.model_config,
            self.feature_vocab_size,
            tokens_per_latent=self.tokens_per_latent,
            num_feature_tokens=self.num_feature_tokens,
        )
        self.gpt = torch.compile(model)

    def general_step(self, tokens, lengths=None, **kwargs):
        # tokens: (B, T) padded token indices without BOS/EOS, lengths: (B,) valid lengths
        if lengths is None:
            lengths = torch.full(
                (tokens.shape[0],),
                tokens.shape[1],
                device=tokens.device,
                dtype=torch.long,
            )
        lengths = lengths.to(device=tokens.device)

        eos_token_id = self.gpt.eos_token_id
        pad_token_id = self.gpt.pad_token_id
        max_len = int(lengths.max().item())
        bos_token = torch.full(
            (tokens.shape[0], 1),
            eos_token_id,
            device=tokens.device,
            dtype=torch.long,
        )
        if self.dense_chunks:
            tokens = torch.cat((bos_token, tokens[:, :max_len]), dim=1)
            lengths = lengths + 1
        else:
            tokens_eos = torch.full(
                (tokens.shape[0], max_len + 1),
                pad_token_id,
                device=tokens.device,
                dtype=torch.long,
            )
            for i, length in enumerate(lengths.tolist()):
                if length:
                    tokens_eos[i, :length] = tokens[i, :length]
                tokens_eos[i, length] = eos_token_id

            tokens = torch.cat((bos_token, tokens_eos), dim=1)
            lengths = lengths + 2
        attention_mask = (
            torch.arange(tokens.shape[1], device=tokens.device)[None, :]
            < lengths[:, None]
        )

        return self.gpt(
            input_ids=tokens,
            attention_mask=attention_mask,
            labels=tokens,
            **kwargs,
        )

    def training_step(self, batch, batch_idx):
        # batch: tokens (B, T) or (tokens, lengths) with lengths (B,)
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            tokens, lengths = batch
        else:
            tokens, lengths = batch, None
        loss = self.general_step(tokens, lengths=lengths)

        self.log(
            "loss/train",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        # batch: tokens (B, T) or (tokens, lengths) with lengths (B,)
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            tokens, lengths = batch
        else:
            tokens, lengths = batch, None
        loss = self.general_step(tokens, lengths=lengths)

        self.log("loss/val", loss, on_epoch=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        depth = self.model_config.gpt_size.n_layer
        weight_decay_scaled = self.training_config.weight_decay * (12 / depth) ** 2
        if self.trainer.is_global_zero:
            print(f"Scaled weight decay: {weight_decay_scaled}")
        self._muon_weight_decay_scaled = weight_decay_scaled
        self._muon_weight_decay_schedule = getattr(
            self.training_config, "muon_weight_decay_schedule", False
        )

        total_epochs = self.training_config.lr_schedule.total_epochs
        if total_epochs is None:
            total_steps = self.trainer.estimated_stepping_batches
        else:
            steps_per_epoch = self.trainer.estimated_stepping_batches // max(
                self.trainer.max_epochs, 1
            )
            total_steps = int(steps_per_epoch * total_epochs)
        self._muon_weight_decay_total_steps = max(int(total_steps), 1)
        optimizer = self.gpt.transformer.setup_optimizer(
            weight_decay=weight_decay_scaled
        )
        schedule_kind = getattr(self.training_config.lr_schedule, "kind", "constant")
        if schedule_kind == "constant":
            sched = get_constant_lr_scheduler(optimizer)
        elif schedule_kind == "cosine":
            warmup_ratio = getattr(
                self.training_config.lr_schedule, "warmup_ratio", 0.0
            )
            final_lr_frac = getattr(
                self.training_config.lr_schedule, "final_lr_frac", 0.1
            )
            sched = get_cosine_annealing_with_warmup_scheduler(
                optimizer,
                total_steps=total_steps,
                warmup_ratio=warmup_ratio,
                final_lr_frac=final_lr_frac,
            )
        elif schedule_kind == "linear_warmup_warmdown":
            warmup_ratio = getattr(
                self.training_config.lr_schedule, "warmup_ratio", 0.0
            )
            warmdown_ratio = getattr(
                self.training_config.lr_schedule, "warmdown_ratio", 0.5
            )
            final_lr_frac = getattr(
                self.training_config.lr_schedule, "final_lr_frac", 0.1
            )
            sched = get_linear_warmup_warmdown_scheduler(
                optimizer,
                total_steps=total_steps,
                warmup_ratio=warmup_ratio,
                warmdown_ratio=warmdown_ratio,
                final_lr_frac=final_lr_frac,
            )
        else:
            raise ValueError(f"Unknown lr_schedule.kind: {schedule_kind}")

        return [optimizer], [sched]

    def on_before_optimizer_step(self, optimizer):
        # Linear ramp from 0.85 -> 0.95 over the first 300 steps.
        it = self.global_step
        frac = min(it / 300, 1)
        momentum = (1 - frac) * 0.85 + frac * 0.95
        weight_decay = self._muon_weight_decay_scaled
        if self._muon_weight_decay_schedule:
            weight_decay = weight_decay * (1 - it / self._muon_weight_decay_total_steps)
            if weight_decay < 0:
                weight_decay = 0.0
        self.log(
            "optim/muon_weight_decay",
            float(weight_decay),
            rank_zero_only=True,
            on_step=True,
            on_epoch=False,
        )
        for group in optimizer.param_groups:
            if group.get("kind") == "muon":
                group["momentum"] = momentum
                if self._muon_weight_decay_schedule:
                    group["weight_decay"] = weight_decay

    @torch.no_grad()
    def sample(
        self,
        max_length=None,
        num_samples=1,
        temperature=1.0,
        top_k=None,
        top_p=None,
        condition: list[GaussianScene] | GaussianScene = None,
        return_lengths: bool = False,
        seed: Optional[int] = None,
    ) -> list[GaussianScene] | tuple[list[GaussianScene], torch.Tensor | None]:
        # sample a sequence of gaussian splats
        self.gpt.eval()

        if self.dense_chunks:
            if not self.chunk_shape or len(self.chunk_shape) != 3:
                raise ValueError("chunk_shape must be set for dense sampling.")
            num_features = self.vqvae.autoencoder.vq.num_tokens
            expected_len = (
                int(self.chunk_shape[0] * self.chunk_shape[1] * self.chunk_shape[2])
                * num_features
            )
            if max_length is None:
                max_length = expected_len
            elif max_length != expected_len:
                raise ValueError("max_length must match dense chunk sequence length.")
        elif max_length is None:
            max_length = self.model_config.n_ctx

        if condition is not None:
            raise NotImplementedError(
                "Conditioning is not supported after switching vqvae.tokenize to return coords/feature_ids."
            )

        tokens = self.gpt.sample_sequence(
            max_length,
            num_samples,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            partials=condition,
            stop_on_eos=not self.dense_chunks,
            seed=seed,
        )
        full_len = tokens.shape[1]
        tokens_per_latent = self.tokens_per_latent
        if self.dense_chunks:
            lengths = torch.full(
                (tokens.shape[0],),
                full_len,
                device=tokens.device,
                dtype=torch.long,
            )
        else:
            lengths = torch.full(
                (tokens.shape[0],),
                full_len,
                device=tokens.device,
                dtype=torch.long,
            )
            if self.gpt.eos_token_id is not None:
                eos_mask = tokens == self.gpt.eos_token_id
                has_eos = eos_mask.any(dim=1)
                first_eos = eos_mask.float().argmax(dim=1)
                lengths = torch.where(has_eos, first_eos, lengths)
            if self.gpt.pad_token_id is not None:
                pad_mask = tokens == self.gpt.pad_token_id
                has_pad = pad_mask.any(dim=1)
                first_pad = pad_mask.float().argmax(dim=1)
                lengths = torch.minimum(
                    lengths, torch.where(has_pad, first_pad, lengths)
                )

            lengths = (lengths // tokens_per_latent) * tokens_per_latent

        # tokens to gaussian splats
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)

        num_pos_tokens = self.num_position_tokens
        feature_vocab_size = self.feature_vocab_size
        position_vocab_size = self.position_vocab_size
        feature_offset = self.feature_token_offset
        base_side_length = self.position_side_length
        pad_id = self.gpt.pad_token_id
        scenes = []
        for idx in range(tokens.shape[0]):
            sample_len = int(lengths[idx].item())
            if (not self.dense_chunks) and sample_len % tokens_per_latent != 0:
                raise ValueError(
                    f"Token length {sample_len} is not a multiple of tokens_per_latent={tokens_per_latent}."
                )
            if self.dense_chunks:
                decoded = self._decode_dense_tokens(tokens[idx, :sample_len])
            else:
                sample_tokens = tokens[idx, :sample_len]
                if sample_tokens.numel() == 0:
                    decoded = self._empty_scene_dict(tokens.device)
                else:
                    token_rows = sample_tokens.view(-1, tokens_per_latent)
                    pos_tokens = token_rows[:, :num_pos_tokens]
                    feature_ids = token_rows[:, num_pos_tokens:]
                    all_pad = (feature_ids == pad_id).all(dim=1)
                    pos_invalid = (pos_tokens < 0) | (pos_tokens >= position_vocab_size)
                    feature_out_of_range = (feature_ids < feature_offset) | (
                        feature_ids >= feature_offset + feature_vocab_size
                    )
                    feature_invalid = (~all_pad).unsqueeze(1) & feature_out_of_range
                    invalid = (~all_pad) & (
                        pos_invalid.any(dim=1) | feature_invalid.any(dim=1)
                    )
                    if invalid.any():
                        print(
                            "WARNING: Sparse tokens contain invalid ids; replacing with safe defaults.",
                            flush=True,
                        )
                        token_rows = token_rows.clone()
                        pos_tokens = token_rows[:, :num_pos_tokens]
                        feature_ids = token_rows[:, num_pos_tokens:]
                        pos_tokens = pos_tokens.masked_fill(pos_invalid, 0)
                        feature_ids = feature_ids.masked_fill(
                            feature_invalid, feature_offset
                        )
                        token_rows[:, :num_pos_tokens] = pos_tokens
                        token_rows[:, num_pos_tokens:] = feature_ids
                        pos_tokens = token_rows[:, :num_pos_tokens]
                        feature_ids = token_rows[:, num_pos_tokens:]

                    valid_mask = ~all_pad
                    if not valid_mask.any():
                        decoded = self._empty_scene_dict(tokens.device)
                    else:
                        coords = pos_tokens_to_centered_coords(
                            pos_tokens[valid_mask],
                            num_pos_tokens,
                            base_side_length,
                        ).to(dtype=torch.long)
                        raw_feature_ids = (feature_ids[valid_mask] - feature_offset).to(
                            dtype=torch.long
                        )
                        decoded = self.vqvae.decode(coords, raw_feature_ids)
            scenes.append(GaussianScene.from_dict(decoded))

        if return_lengths:
            return scenes, lengths.detach().cpu()
        return scenes

    def _decode_dense_tokens(self, tokens: torch.Tensor) -> dict:
        num_features = self.vqvae.autoencoder.vq.num_tokens
        if tokens.numel() % num_features != 0:
            raise ValueError("Dense token length is not divisible by num_features.")
        if not self.chunk_shape or len(self.chunk_shape) != 3:
            raise ValueError("chunk_shape must be set for dense decoding.")

        chunk = torch.tensor(self.chunk_shape, device=tokens.device)
        num_voxels = int(chunk[0] * chunk[1] * chunk[2])
        if tokens.numel() != num_voxels * num_features:
            raise ValueError("Dense token length does not match chunk_shape.")

        token_rows = tokens.view(num_voxels, num_features)
        if self.chunk_order != "xyz":
            inv = self._dense_order_inv(tokens.device)
            token_rows = token_rows[inv]
        pad_id = self.gpt.pad_token_id
        all_pad = (token_rows == pad_id).all(dim=1)
        feature_out_of_range = (token_rows < self.feature_token_offset) | (
            token_rows >= self.feature_token_offset + self.feature_vocab_size
        )
        feature_invalid = (~all_pad).unsqueeze(1) & feature_out_of_range
        invalid = (~all_pad) & feature_invalid.any(dim=1)
        if invalid.any():
            print(
                "WARNING: Dense tokens contain invalid ids; replacing with safe defaults.",
                flush=True,
            )
            token_rows = token_rows.clone()
            token_rows = token_rows.masked_fill(
                feature_invalid, self.feature_token_offset
            )

        valid_mask = ~all_pad
        if not valid_mask.any():
            return self._empty_scene_dict(tokens.device)

        xs = torch.arange(chunk[0], device=tokens.device)
        ys = torch.arange(chunk[1], device=tokens.device)
        zs = torch.arange(chunk[2], device=tokens.device)
        grid_x, grid_y, grid_z = torch.meshgrid(xs, ys, zs, indexing="ij")
        coords = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)
        coords = coords[valid_mask]
        feature_ids = (token_rows[valid_mask] - self.feature_token_offset).to(
            dtype=torch.long
        )
        return self.vqvae.decode(coords, feature_ids)

    def _dense_order_inv(self, device: torch.device) -> torch.Tensor:
        if (
            self._dense_order_inv_cache is None
            or self._dense_order_inv_cache.device != device
        ):
            self._dense_order_inv_cache = dense_chunk_inverse_order_indices(
                self.chunk_shape, self.chunk_order, device=device
            )
        return self._dense_order_inv_cache

    def _empty_scene_dict(self, device: torch.device) -> dict:
        default_dims = {
            "coords": 3,
            "opacities": 1,
            "scales": 3,
            "quats": 4,
            "sh0": 3,
        }
        features = getattr(self.vqvae.autoencoder, "features", {})
        for key, params in features.items():
            if key in default_dims or key == "coord_offset":
                continue
            if params.get("in_output", False):
                dim = int(params.get("dimension", 0))
                if dim > 0:
                    default_dims[key] = dim
        empty = {}
        for key, default_dim in default_dims.items():
            dim = features.get(key, {}).get("dimension", default_dim)
            empty[key] = torch.empty((0, dim), device=device)
        return empty
