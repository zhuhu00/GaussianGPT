import os

import lightning
import MinkowskiEngine as ME
import torch

from conf.dataclasses import ImageKeys, PCKeys
from serialization import encode
from utils.config import instantiate_feature_config
from utils.gaussian_vqvae_utils import compute_loss_coords, split_batch_dict
from utils.inference import inference_call
from utils.optim import (
    get_constant_lr_scheduler,
    get_cosine_annealing_with_warmup_scheduler,
    get_decay_groups,
    get_linear_warmup_warmdown_scheduler,
)
from utils.rerendering_loss import RerenderingLoss

from .vqvae.cnn.configurable_vqvae import ConfigurableVQVAE

# disable unused arguments and arguments differ for entire script due to lightning hooks
# pylint: disable=unused-argument, W0221


class GaussianVQVAE(lightning.LightningModule):
    def __init__(
        self,
        model_config,
        training_config=None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model_config = model_config
        self.training_config = training_config

        # self.autoencoder = get_autoencoder_from_config(model_config, training_config)
        self.autoencoder = ConfigurableVQVAE(
            features=instantiate_feature_config(model_config.feature_configs),
            default_grid_size=model_config.grid_size,
            encoder_config=model_config.encoder,
            decoder_config=model_config.decoder,
            vq_config=model_config.vq,
            make_unique=model_config.get("make_unique", False),
            gaussians_per_voxel=model_config.get("gaussians_per_voxel", 1),
        )

        # rerendering loss - only load when training
        rerendering_loss = (
            RerenderingLoss(
                self.device,
                background_color="white",
            )
            if training_config is not None
            else None
        )
        # hacky way to bypass registering this as a submodule of self
        object.__setattr__(self, "rerendering_loss", rerendering_loss)
        self.background_color = "white"

        self._codebook_usage_counts = (
            None  # per-code usage histogram, accumulated over a val epoch
        )

        self.train_loss_str = "loss/train"
        self.val_loss_str = "loss/val"

    # overwrite ".to" so that the rerendering loss is also moved to the correct device
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        if self.rerendering_loss is not None:
            self.rerendering_loss.to(*args, **kwargs)
            self.rerendering_loss.device = self.device
        return self

    def set_background_color(self, background_color: str) -> None:
        if background_color not in ("white", "black"):
            raise ValueError(
                f"Unsupported background color '{background_color}'. Expected 'white' or 'black'."
            )
        self.background_color = background_color
        if self.rerendering_loss is not None:
            self.rerendering_loss.set_background_color(background_color)

    def _adjust_decoder_prune_thresholds(self):
        """
        Due to memory reasons, we start with a high threshold and linearly interpolate to the final one.
        This schedule can be disabled via model.decoder.prune_thresh_schedule=False.
        """
        schedule_enabled = (
            self.model_config.decoder.get("prune_thresh_schedule", True)
            if hasattr(self.model_config.decoder, "get")
            else getattr(self.model_config.decoder, "prune_thresh_schedule", True)
        )
        if not schedule_enabled:
            for i in range(len(self.autoencoder.decoder.prune_thresh)):
                self.autoencoder.decoder.prune_thresh[i] = (
                    self.model_config.decoder.prune_thresh[i]
                )
            return

        start_thresh = 0.1
        total_steps = 10_000

        progress = min((self.global_step / total_steps) ** 2, 1.0)
        for i in range(len(self.autoencoder.decoder.prune_thresh)):
            final_thresh = self.model_config.decoder.prune_thresh[i]
            self.autoencoder.decoder.prune_thresh[i] = start_thresh + progress * (
                final_thresh - start_thresh
            )

    def general_step(self, batch, batch_idx, mode="unspecified"):
        self._adjust_decoder_prune_thresholds()
        # create subdict of batch that contains point cloud
        point_dict, image_dict = split_batch_dict(batch, device=self.device)

        input_points = dict(point_dict)

        if ImageKeys.IMAGES not in image_dict:
            if self.rerendering_loss is None:
                raise RuntimeError(
                    "Rerendering loss is required when supervision images are missing."
                )
            with torch.no_grad():
                image_dict = dict(image_dict)
                image_dict[ImageKeys.IMAGES] = [
                    image.detach()
                    for image in self.rerendering_loss.render_images(
                        input_points, image_dict
                    )
                ]

        assert (
            self.training_config is not None
        ), "Training config must be provided for training steps."

        # log the number of input points
        num_scenes = input_points["batch"].unique().numel()
        num_input_points = input_points[PCKeys.COORDS].shape[0]
        logging_params: dict = {
            "sync_dist": mode == "val",
            "on_epoch": mode == "val",
            "batch_size": num_scenes,
        }
        self.log(
            f"num_gaussians/input_{mode}",
            num_input_points / num_scenes,
            **logging_params,
        )

        rerender_cfg = self.training_config.losses.rerendering
        loss_weights = {
            "l1_loss": rerender_cfg.recon_weight,
            "vgg_loss": rerender_cfg.vgg_weight,
            "lpips_loss": rerender_cfg.lpips_weight,
            "ssim_loss": rerender_cfg.ssim_weight,
            "depth_loss": float(getattr(rerender_cfg, "depth_weight", 0.0)),
        }

        # add target key if training or sanity checking
        add_target_key = self.training
        points, out_dict = self.autoencoder.autoencode(
            input_points, add_target_key=add_target_key
        )
        aux_loss = out_dict["loss_commit"]

        if mode == "val" and "idxs" in out_dict:
            self._accumulate_codebook_usage(out_dict["idxs"])

        # log the auxiliary loss
        total_loss = aux_loss * self.training_config.losses.commit_weight
        if mode != "val":
            # there is no aux loss during validation
            self.log(f"loss_aux/{mode}", aux_loss, **logging_params)

        # log number of latents
        num_latents = out_dict.get("num_latents", 0)
        self.log(f"num_latents/{mode}", num_latents, **logging_params)

        # log prune thresholds and update afterwards
        if mode == "train":
            for i, thresh in enumerate(self.autoencoder.decoder.prune_thresh):
                self.log(
                    f"prune_thresholds/{mode}_prune{i+1}",
                    thresh,
                    **logging_params,
                )

        # we add the images from the sanity check for further sanity checks
        if self.trainer.sanity_checking and batch_idx == 0:
            sanity_image_dir = (
                os.path.join(self.trainer.log_dir, "sanity_images")
                if self.trainer.sanity_checking
                else None
            )
        else:
            sanity_image_dir = None

        # log mean # gaussians, ie. all points / #batches
        num_gaussians = points[PCKeys.COORDS].shape[0] / num_scenes
        self.log(f"num_gaussians/{mode}", num_gaussians, **logging_params)

        # rerendering loss, perceptual terms enabled only if weights > 0
        rerendering_output = self.rerendering_loss(
            dict(points),
            image_dict,
            calc_metrics=(mode == "val"),
            use_lpips_loss=loss_weights["lpips_loss"] > 0.0,
            use_ssim_loss=loss_weights["ssim_loss"] > 0.0,
            use_vgg_loss=loss_weights["vgg_loss"] > 0.0,
            output_dir=sanity_image_dir,
            depth_weight=loss_weights["depth_loss"],
        )  # type: ignore

        for loss_name, weight in loss_weights.items():
            if weight > 0.0:
                total_loss += rerendering_output[loss_name] * weight
                self.log(
                    f"loss_recon/{mode}_{loss_name}",
                    rerendering_output[loss_name],
                    **logging_params,
                )
        if "mask_coverage" in rerendering_output:
            self.log(
                f"loss_recon/{mode}_mask_coverage",
                rerendering_output["mask_coverage"],
                **logging_params,
            )

        if self.training:
            loss_coords, accuracy, coord_metrics = compute_loss_coords(
                out_dict["out_cls"],
                out_dict["targets"],
                points[PCKeys.COORDS].device,
            )
            total_loss += loss_coords * 1.0  # weight for coord loss is just 1.0

            # log coord loss and accuracy
            self.log(f"coords/{mode}_loss", loss_coords, **logging_params)
            self.log(f"coords/{mode}_accuracy", accuracy, **logging_params)
            self.log(
                f"coords/{mode}_false_negative_rate",
                coord_metrics["false_negative_rate_total"],
                **logging_params,
            )
            self.log(
                f"coords/{mode}_false_positive_rate",
                coord_metrics["false_positive_rate_total"],
                **logging_params,
            )
            for idx, (acc, fnr, fpr) in enumerate(
                zip(
                    coord_metrics["accuracy"],
                    coord_metrics["false_negative_rate"],
                    coord_metrics["false_positive_rate"],
                )
            ):
                self.log(f"coords/{mode}_accuracy_layer_{idx}", acc, **logging_params)
                self.log(
                    f"coords/{mode}_false_negative_rate_layer_{idx}",
                    fnr,
                    **logging_params,
                )
                self.log(
                    f"coords/{mode}_false_positive_rate_layer_{idx}",
                    fpr,
                    **logging_params,
                )

        # metrics, last output only
        if mode == "val":
            # Log the metrics only during validation
            self.log("metrics/psnr", rerendering_output["psnr"], **logging_params)
            self.log("metrics/ssim", rerendering_output["ssim"], **logging_params)
            self.log("metrics/lpips", rerendering_output["lpips"], **logging_params)

        # total loss is important for callback so we use a variable to access
        loss_str = self.val_loss_str if mode == "val" else self.train_loss_str
        self.log(loss_str, total_loss, **logging_params)
        if mode == "train":
            self.log(
                "loss/train_epoch",
                total_loss,
                sync_dist=True,
                on_step=False,
                on_epoch=True,
                batch_size=num_scenes,
            )

        return total_loss

    def training_step(self, batch, batch_idx):
        # Run the general step
        total_loss = self.general_step(batch, batch_idx, "train")

        return total_loss

    def validation_step(self, batch, batch_idx):
        # Run the general step
        total_loss = self.general_step(batch, batch_idx, "val")

        return total_loss

    def on_validation_epoch_start(self) -> None:
        self._codebook_usage_counts = None

    def _accumulate_codebook_usage(self, idxs: torch.Tensor) -> None:
        # defensive: never let usage logging break a training step
        try:
            codebook_size = int(getattr(self.autoencoder.vq, "codebook_size", 0))
            if codebook_size <= 0:
                return
            flat = idxs.detach().reshape(-1).long()
            if flat.numel() == 0:
                return
            in_range = (flat >= 0) & (flat < codebook_size)
            if not bool(in_range.all()):
                flat = flat[in_range]
            if flat.numel() == 0:
                return
            counts = torch.bincount(flat, minlength=codebook_size)
            if self._codebook_usage_counts is None:
                self._codebook_usage_counts = counts
            else:
                self._codebook_usage_counts = self._codebook_usage_counts + counts
        except Exception:  # noqa: BLE001
            self._codebook_usage_counts = None

    def on_validation_epoch_end(self) -> None:
        counts = self._codebook_usage_counts
        if counts is None:
            return
        try:
            if (
                torch.distributed.is_available()
                and torch.distributed.is_initialized()
                and self.trainer.world_size > 1
            ):
                counts = counts.clone()
                torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)
            counts_f = counts.float()
            used_pct = (counts > 0).float().mean().item() * 100.0
            median_count = counts_f.median().item()
            self.log(
                "vq/usage_pct",
                used_pct,
                rank_zero_only=True,
                on_epoch=True,
                on_step=False,
            )
            self.log(
                "vq/usage_median",
                median_count,
                rank_zero_only=True,
                on_epoch=True,
                on_step=False,
            )
        except Exception:  # noqa: BLE001
            pass
        finally:
            self._codebook_usage_counts = None

    def configure_optimizers(self):
        optim_groups = get_decay_groups(
            self.autoencoder.named_parameters(),
            self.training_config.weight_decay,
            verbose=self.trainer.is_global_zero,
        )

        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.training_config.lr, betas=(0.9, 0.95)
        )

        total_epochs = self.training_config.lr_schedule.total_epochs
        if total_epochs is None:
            total_steps = self.trainer.estimated_stepping_batches
        else:
            steps_per_epoch = self.trainer.estimated_stepping_batches // max(
                self.trainer.max_epochs, 1
            )
            total_steps = int(steps_per_epoch * total_epochs)
        schedule_kind = getattr(self.training_config.lr_schedule, "kind", "constant")
        if schedule_kind == "constant":
            scheduler = get_constant_lr_scheduler(optimizer)
        elif schedule_kind == "cosine":
            warmup_ratio = getattr(
                self.training_config.lr_schedule, "warmup_ratio", 0.0
            )
            final_lr_frac = getattr(
                self.training_config.lr_schedule, "final_lr_frac", 0.1
            )
            scheduler = get_cosine_annealing_with_warmup_scheduler(
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
            scheduler = get_linear_warmup_warmdown_scheduler(
                optimizer,
                total_steps=total_steps,
                warmup_ratio=warmup_ratio,
                warmdown_ratio=warmdown_ratio,
                final_lr_frac=final_lr_frac,
            )
        else:
            raise ValueError(f"Unknown lr_schedule.kind: {schedule_kind}")

        # return optimizer, scheduler
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def tokenize(
        self,
        points: dict,
        sort_latents=None,
    ) -> torch.Tensor:
        # assume coords are centered around 0
        ae = self.autoencoder
        coords, idxs = ae.get_idxs(points)  # (N, 4), (num_tokens, N)
        if idxs.dim() == 1:
            idxs = idxs.unsqueeze(0)

        assert torch.all(coords[:, 0] == 0), "Batching not supported in tokenize."
        coords = coords[:, 1:]  # (N, 3)

        if sort_latents is not None and coords.numel() > 0:
            coord_min = coords.min(dim=0).values
            grid_coord = (coords - coord_min).long()
            depth = int(grid_coord.max().item() + 1).bit_length()
            codes = encode(grid_coord, depth=depth, order=sort_latents)
            order = torch.argsort(codes.reshape(-1))

            coords = coords.index_select(0, order)
            idxs = idxs.index_select(1, order)

        return {"coords": coords, "feature_ids": idxs.T}

    @inference_call
    def decode(self, coords: torch.Tensor, feature_ids: torch.Tensor) -> dict:
        return self.autoencoder.decode_idxs(coords, feature_ids.T)

    @inference_call
    def decode_embeddings(self, coords: torch.Tensor, embeddings: torch.Tensor) -> dict:
        """
        Decode from continuous latent embeddings at given latent grid coords.
        """
        device = embeddings.device
        coords = coords.to(device)
        embeddings = embeddings.to(device)

        coords = coords * self.autoencoder.stride
        if coords.shape[-1] == 3:
            batch_dim = torch.zeros(
                coords.shape[0], 1, device=device, dtype=torch.int32
            )
            coords = torch.cat([batch_dim, coords.to(torch.int32)], dim=-1)
        else:
            coords = coords.to(torch.int32)

        latents = ME.SparseTensor(
            features=embeddings,
            coordinates=coords,
            device=device,
            tensor_stride=torch.tensor([self.autoencoder.stride] * 3, device=device),
        )

        points, _ = self.autoencoder.decode(latents)
        pred_grid_coords = points[PCKeys.COORDS] + points[PCKeys.COORD_OFFSET]
        points[PCKeys.COORDS] = pred_grid_coords * self.autoencoder.default_grid_size
        return points
