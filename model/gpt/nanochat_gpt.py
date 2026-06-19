"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration
"""

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from utils.gaussian_vqvae_utils import int_cube_root
from utils.pos_tokens import dense_chunk_token_positions, oned_to_threed_indices

# Our custom Flash Attention module that automatically uses FA3 on Hopper+ and SDPA fallback elsewhere
from .flash_attention import flash_attn
from .rope import RotaryPositionEmbedder
from .rope_config import resolve_rope_layout


def is_ddp_requested() -> bool:
    """
    True if launched by torchrun (env present), even before init.
    Used to decide whether we *should* initialize a PG.
    """
    return all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))


def is_ddp_initialized() -> bool:
    """
    True if torch.distributed is available and the process group is initialized.
    Used at cleanup to avoid destroying a non-existent PG.
    """
    return dist.is_available() and dist.is_initialized()


def get_dist_info():
    if is_ddp_requested():
        # We rely on torchrun's env to decide if we SHOULD init.
        # (Initialization itself happens in compute init.)
        assert all(var in os.environ for var in ["RANK", "LOCAL_RANK", "WORLD_SIZE"])
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        return True, ddp_rank, ddp_local_rank, ddp_world_size
    else:
        return False, 0, 0, 1


def print0(*args, **kwargs):
    ddp, rank, _, _ = get_dist_info()
    if not ddp or rank == 0:
        print(*args, **kwargs)


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6  # number of query heads
    n_kv_head: int = 6  # number of key/value heads (GQA)
    n_embd: int = 768
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (half context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    # Recommended to only use "L" for now since compatibility with 3D RoPE has not been verified otherwise.
    window_pattern: str = "L"
    rope_basis: str = "sequence"
    rope_bos_coord: Tuple[int, int, int] = (-1, -1, -1)
    learned_pos_embed: bool = False
    dense_chunks: bool = False
    dense_chunk_shape: Optional[Tuple[int, int, int]] = None
    dense_chunk_order: str = "xyz"
    dense_num_features: int = 1
    sparse_tokens_per_latent: int = 1
    sparse_num_position_tokens: int = 1
    sparse_position_hard_constraints: bool = False
    value_embed_every_n_layers: Optional[int] = 2
    sparse_position_vocab_size: int = 32768
    sparse_feature_vocab_size: int = 32768
    sparse_feature_token_offset: int = 32768
    sparse_shared_vocab: bool = False


def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer, every_n_layers: Optional[int]):
    """Returns True if GPT layer should have Value Embedding."""
    if every_n_layers is None:
        return False
    every_n_layers = int(every_n_layers)
    if every_n_layers < 1:
        raise ValueError("value_embed_every_n_layers must be >= 1 or None.")
    return layer_idx % every_n_layers == (n_layer - 1) % every_n_layers


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]  # split up last dim into two halves
    y1 = x1 * cos + x2 * sin  # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = (
            nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
            if has_ve(layer_idx, config.n_layer, config.value_embed_every_n_layers)
            else None
        )

    def forward(self, x, ve, rope, window_size, kv_cache):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(
                self.ve_gate(x[..., : self.ve_gate_channels])
            )  # (B, T, n_kv_head), range (0, 2)
            v = v + gate.unsqueeze(-1) * ve

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        if rope is not None:
            if isinstance(rope, tuple):
                cos, sin = rope
                q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
            else:
                q = RotaryPositionEmbedder.apply_rotary_embedding(q, rope)
                k = RotaryPositionEmbedder.apply_rotary_embedding(k, rope)
        q, k = norm(q), norm(k)  # QK norm

        # Flash Attention (FA3 on Hopper+, PyTorch SDPA fallback elsewhere)
        # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
        if kv_cache is None:
            # Training: causal attention with optional sliding window
            y = flash_attn.flash_attn_func(
                q, k, v, causal=True, window_size=window_size
            )
        else:
            # Inference: use flash_attn_with_kvcache which handles cache management
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q,
                k_cache,
                v_cache,
                k=k,
                v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            # Advance position after last layer processes
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        # Re-assemble the heads and project back to residual stream
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, rope, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, rope, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        self.learned_pos_embed = bool(getattr(config, "learned_pos_embed", False))
        self.use_rope = not self.learned_pos_embed
        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        self.window_sizes = self._compute_window_sizes(config)
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = (
            (config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to
        ) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(
                f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency"
            )
        transformer_modules = {
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList(
                [Block(config, layer_idx) for layer_idx in range(config.n_layer)]
            ),
        }
        if self.learned_pos_embed:
            transformer_modules["wpe"] = nn.Embedding(
                config.sequence_len, config.n_embd
            )
        self.transformer = nn.ModuleDict(transformer_modules)
        self.lm_head = nn.Linear(config.n_embd, padded_vocab_size, bias=False)
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        # Separate parameters so they can have different optimizer treatment
        self.resid_lambdas = nn.Parameter(
            torch.ones(config.n_layer)
        )  # fake init, real init in init_weights()
        self.x0_lambdas = nn.Parameter(
            torch.zeros(config.n_layer)
        )  # fake init, real init in init_weights()
        # Value embeddings (ResFormer-style): configurable cadence, always aligned to include last layer.
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict(
            {
                str(i): nn.Embedding(padded_vocab_size, kv_dim)
                for i in range(config.n_layer)
                if has_ve(i, config.n_layer, config.value_embed_every_n_layers)
            }
        )
        self.rope_layout = resolve_rope_layout(
            rope_basis=config.rope_basis,
            dense_chunks=config.dense_chunks,
        )
        self.rope_dim = self.rope_layout.dim
        self.sparse_position_hard_constraints = bool(
            config.sparse_position_hard_constraints
        )
        if self.use_rope and self.rope_dim > 1 and head_dim % (2 * self.rope_dim) != 0:
            raise ValueError(
                f"head_dim must be divisible by {2 * self.rope_dim} for "
                f"{self.rope_dim}D RoPE, got {head_dim}."
            )
        self.sparse_tokens_per_latent = int(config.sparse_tokens_per_latent)
        self.sparse_num_position_tokens = int(config.sparse_num_position_tokens)
        self.sparse_position_vocab_size = int(config.sparse_position_vocab_size)
        self.sparse_feature_vocab_size = int(config.sparse_feature_vocab_size)
        self.sparse_feature_token_offset = int(config.sparse_feature_token_offset)
        self.sparse_shared_vocab = bool(getattr(config, "sparse_shared_vocab", False))
        if self.sparse_position_vocab_size <= 0:
            raise ValueError("sparse_position_vocab_size must be > 0.")
        if self.sparse_feature_vocab_size <= 0:
            raise ValueError("sparse_feature_vocab_size must be > 0.")
        if self.sparse_shared_vocab:
            if self.sparse_feature_token_offset != 0:
                raise ValueError(
                    "sparse_feature_token_offset must be 0 with sparse_shared_vocab=True."
                )
            base = max(self.sparse_position_vocab_size, self.sparse_feature_vocab_size)
            self.sparse_eos_token_id = base
            self.sparse_pad_token_id = base + 1
            expected_vocab_size = base + 2
        else:
            if self.sparse_feature_token_offset != self.sparse_position_vocab_size:
                raise ValueError(
                    "sparse_feature_token_offset must equal sparse_position_vocab_size."
                )
            self.sparse_eos_token_id = (
                self.sparse_feature_token_offset + self.sparse_feature_vocab_size
            )
            self.sparse_pad_token_id = self.sparse_eos_token_id + 1
            expected_vocab_size = self.sparse_pad_token_id + 1
        if int(config.vocab_size) != expected_vocab_size:
            raise ValueError(
                "vocab_size must match sparse vocabulary layout "
                f"(expected {expected_vocab_size}, got {int(config.vocab_size)})."
            )
        if self.rope_layout.is_sparse and self.sparse_tokens_per_latent < 1:
            raise ValueError("sparse_tokens_per_latent must be >= 1.")
        if self.rope_layout.is_sparse and self.rope_layout.is_position:
            if self.sparse_num_position_tokens != 1:
                raise ValueError(
                    "Sparse position-based RoPE currently requires num_position_tokens=1."
                )
        self.dense_chunk_shape = None
        self.dense_chunk_order = config.dense_chunk_order
        self.dense_num_features = int(config.dense_num_features)
        if self.sparse_position_hard_constraints and self.rope_layout.is_sparse:
            if self.sparse_num_position_tokens != 1:
                raise ValueError(
                    "Monotonic sparse position sampling currently requires sparse_num_position_tokens=1."
                )
            if self.dense_chunk_order not in {"xyz", "xzy"}:
                raise ValueError(
                    "Monotonic sparse position sampling is only implemented for "
                    "chunk_order in {'xyz', 'xzy'}."
                )
        self.rope_bos_coord = tuple(int(x) for x in config.rope_bos_coord)
        if len(self.rope_bos_coord) != 3:
            raise ValueError("rope_bos_coord must have 3 elements.")
        if self.rope_layout.is_dense and self.rope_layout.is_position:
            if config.dense_chunk_shape is None or len(config.dense_chunk_shape) != 3:
                raise ValueError(
                    "dense_chunk_shape must be set for dense position-based RoPE."
                )
            if self.dense_num_features != 1:
                raise ValueError(
                    "Dense position-based RoPE currently requires dense_num_features=1."
                )
            self.dense_chunk_shape = tuple(int(x) for x in config.dense_chunk_shape)
        if self.rope_layout.is_sparse and self.rope_layout.is_position:
            self.sparse_rope_side_length = int_cube_root(
                self.sparse_position_vocab_size
            )
            if self.sparse_rope_side_length**3 != self.sparse_position_vocab_size:
                raise ValueError(
                    "sparse_position_vocab_size must be a perfect cube for position-based RoPE."
                )
        else:
            self.sparse_rope_side_length = 0
        if self.use_rope and self.rope_dim > 1:
            self.rope_embedder = RotaryPositionEmbedder(head_dim, dim=self.rope_dim)
        self.register_buffer(
            "sparse_position_rank_map",
            torch.empty(0, dtype=torch.long),
            persistent=False,
        )
        if self.sparse_position_hard_constraints and self.rope_layout.is_sparse:
            self.sparse_position_rank_map = self._build_sparse_position_rank_map()
        self.register_buffer(
            "dense_rope_phases",
            torch.empty(1, 0, head_dim // 2, dtype=torch.complex64),
            persistent=False,
        )
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = (
            config.sequence_len * 10
        )  # 10X over-compute should be enough, TODO make nicer?
        if self.use_rope and self.rope_dim == 1:
            cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
            self.register_buffer(
                "cos", cos, persistent=False
            )  # persistent=False means it's not saved to the checkpoint
            self.register_buffer("sin", sin, persistent=False)
        else:
            self.register_buffer("cos", torch.empty(0), persistent=False)
            self.register_buffer("sin", torch.empty(0), persistent=False)

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """

        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        if self.learned_pos_embed:
            torch.nn.init.normal_(self.transformer.wpe.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = (
            3**0.5 * n_embd**-0.5
        )  # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        for block in self.transformer.h:
            torch.nn.init.uniform_(
                block.attn.c_q.weight, -s, s
            )  # weights use Uniform to avoid outliers
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)  # projections are zero
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)  # 1.0 => typical residual connections at init
        self.x0_lambdas.fill_(
            0.1
        )  # 0.1 => small initial weight for skip connection to input embedding

        # Value embeddings (init like c_v: uniform with same std)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # Gate weights init to zero so gates start at sigmoid(0) = 0.5, scaled by 2 -> 1.0 (neutral)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)

        # Rotary embeddings / phases
        head_dim = self.config.n_embd // self.config.n_head
        if self.use_rope:
            if self.rope_dim == 1:
                cos, sin = self._precompute_rotary_embeddings(
                    self.rotary_seq_len, head_dim
                )
                self.cos, self.sin = cos, sin
            elif self.rope_layout.is_dense and self.rope_layout.is_position:
                self.dense_rope_phases = self._build_dense_rope_phases(
                    self.transformer.wte.weight.device
                )

        # Cast embeddings to bf16: optimizer can tolerate it and it saves memory
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)
            if self.learned_pos_embed:
                self.transformer.wpe.to(dtype=torch.bfloat16)
            for ve in self.value_embeds.values():
                ve.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()  # keep them in bfloat16
        cos, sin = (
            cos[None, :, None, :],
            sin[None, :, None, :],
        )  # add batch and head dims for later broadcasting
        return cos, sin

    def _build_dense_rope_phases(self, device):
        positions = dense_chunk_token_positions(
            self.dense_chunk_shape,
            self.dense_num_features,
            chunk_order=self.dense_chunk_order,
            device=device,
            dtype=torch.long,
        )
        bos = torch.tensor(self.rope_bos_coord, device=device, dtype=torch.long)
        positions = torch.cat([bos.view(1, 3), positions], dim=0)
        return self.rope_embedder(positions.unsqueeze(0))

    def _get_dense_rope_phases(self, device):
        if (
            self.dense_rope_phases.numel() == 0
            or self.dense_rope_phases.device != device
        ):
            self.dense_rope_phases = self._build_dense_rope_phases(device)
        return self.dense_rope_phases

    def _build_sparse_sequence_indices(
        self,
        *,
        batch_size: int,
        start_pos: int,
        length: int,
        device: torch.device,
    ) -> torch.Tensor:
        positions = torch.arange(
            start_pos, start_pos + length, device=device, dtype=torch.long
        )
        token_positions = torch.clamp(positions - 1, min=0)
        latent_idx = torch.div(
            token_positions,
            self.sparse_tokens_per_latent,
            rounding_mode="floor",
        )
        slot_idx = torch.remainder(token_positions, self.sparse_tokens_per_latent)
        bos_mask = positions == 0
        latent_idx = torch.where(
            bos_mask,
            torch.full_like(latent_idx, -1),
            latent_idx,
        )
        slot_idx = torch.where(bos_mask, torch.zeros_like(slot_idx), slot_idx)
        indices = torch.stack([latent_idx, slot_idx], dim=-1)
        return indices.unsqueeze(0).expand(batch_size, -1, -1)

    def _build_sparse_position_indices(self, full_tokens: torch.Tensor) -> torch.Tensor:
        batch_size, total_len = full_tokens.shape
        device = full_tokens.device
        positions = torch.arange(total_len, device=device, dtype=torch.long)
        token_positions = torch.clamp(positions - 1, min=0)
        latent_idx = torch.div(
            token_positions,
            self.sparse_tokens_per_latent,
            rounding_mode="floor",
        )
        slot_idx = torch.remainder(token_positions, self.sparse_tokens_per_latent)
        slot0_positions = 1 + latent_idx * self.sparse_tokens_per_latent
        slot0_positions = torch.where(
            positions == 0,
            torch.zeros_like(slot0_positions),
            slot0_positions,
        )
        slot0_positions = slot0_positions.unsqueeze(0).expand(batch_size, -1)
        position_token_ids = full_tokens.gather(1, slot0_positions)
        valid_position_token = (position_token_ids >= 0) & (
            position_token_ids < self.sparse_position_vocab_size
        )
        safe_position_token_ids = position_token_ids.masked_fill(
            ~valid_position_token, 0
        )
        coords = oned_to_threed_indices(
            safe_position_token_ids,
            self.sparse_rope_side_length,
        ).to(dtype=torch.long)
        coords = coords.masked_fill(~valid_position_token.unsqueeze(-1), 0)
        bos_mask = (positions == 0).unsqueeze(0).unsqueeze(-1)
        bos_coords = torch.tensor(self.rope_bos_coord, device=device, dtype=torch.long)
        coords = torch.where(
            bos_mask,
            bos_coords.view(1, 1, 3),
            coords,
        )
        slot_idx = slot_idx.unsqueeze(0).expand(batch_size, -1)
        slot_idx = torch.where(
            (positions == 0).unsqueeze(0),
            torch.zeros_like(slot_idx),
            slot_idx,
        )
        return torch.cat([coords, slot_idx.unsqueeze(-1)], dim=-1)

    def _build_sparse_rope_indices(
        self,
        idx: torch.Tensor,
        *,
        start_pos: int,
        kv_cache=None,
    ) -> torch.Tensor:
        if self.rope_layout.is_sequence:
            return self._build_sparse_sequence_indices(
                batch_size=idx.size(0),
                start_pos=start_pos,
                length=idx.size(1),
                device=idx.device,
            )

        if kv_cache is None or start_pos == 0:
            full_tokens = idx
        else:
            prefix = kv_cache.get_tokens(start_pos)
            full_tokens = torch.cat((prefix, idx), dim=1)

        full_indices = self._build_sparse_position_indices(full_tokens)
        return full_indices[:, start_pos : start_pos + idx.size(1)]

    def _get_rope(self, idx: torch.Tensor, kv_cache=None):
        if not self.use_rope:
            return None
        token_count = idx.size(1)
        start_pos = 0 if kv_cache is None else kv_cache.get_pos()
        if self.rope_dim == 1:
            if start_pos + token_count > self.cos.size(1):
                raise ValueError(
                    "Sequence length grew beyond the rotary embeddings cache: "
                    f"{start_pos + token_count} > {self.cos.size(1)}"
                )
            if idx.device != self.cos.device:
                raise ValueError(
                    "Rotary embeddings and idx are on different devices: "
                    f"{idx.device} != {self.cos.device}"
                )
            if self.cos.dtype != torch.bfloat16:
                raise ValueError("Rotary embeddings must be in bfloat16.")
            return (
                self.cos[:, start_pos : start_pos + token_count],
                self.sin[:, start_pos : start_pos + token_count],
            )

        if self.rope_layout.is_dense and self.rope_layout.is_position:
            phases = self._get_dense_rope_phases(idx.device)
            if start_pos + token_count > phases.size(1):
                raise ValueError(
                    "Sequence length grew beyond dense RoPE cache: "
                    f"{start_pos + token_count} > {phases.size(1)}"
                )
            return phases[:, start_pos : start_pos + token_count]

        indices = self._build_sparse_rope_indices(
            idx,
            start_pos=start_pos,
            kv_cache=kv_cache,
        )
        return self.rope_embedder(indices)

    def _prediction_slot_indices(
        self,
        *,
        start_pos: int,
        length: int,
        device: torch.device,
    ) -> torch.Tensor:
        positions = torch.arange(
            start_pos + 1, start_pos + length + 1, device=device, dtype=torch.long
        )
        token_positions = torch.clamp(positions - 1, min=0)
        return torch.remainder(token_positions, self.sparse_tokens_per_latent)

    def _apply_slot_logits_mask(self, logits: torch.Tensor, *, start_pos: int) -> None:
        if logits.ndim != 3:
            raise ValueError(
                f"logits must have shape (B, T, V), got {tuple(logits.shape)}"
            )
        _, token_count, _ = logits.shape
        if token_count == 0:
            return

        slot_idx = self._prediction_slot_indices(
            start_pos=start_pos, length=token_count, device=logits.device
        )
        is_position_slot = slot_idx < self.sparse_num_position_tokens
        vocab_size = logits.size(-1)
        position_vocab = self.sparse_position_vocab_size
        feature_vocab = self.sparse_feature_vocab_size
        eos_id = self.sparse_eos_token_id
        pad_id = self.sparse_pad_token_id

        logits[..., pad_id] = -torch.inf
        if self.sparse_shared_vocab:
            if is_position_slot.any():
                # Position slot: valid ids are [0, position_vocab) ∪ {eos_id}.
                if position_vocab < eos_id:
                    logits[:, is_position_slot, position_vocab:eos_id] = -torch.inf
                if eos_id + 1 < vocab_size:
                    logits[:, is_position_slot, eos_id + 1 :] = -torch.inf
            if (~is_position_slot).any():
                # Feature slot: valid ids are [0, feature_vocab); eos and pad sit in [feature_vocab, vocab).
                logits[:, ~is_position_slot, feature_vocab:vocab_size] = -torch.inf
        else:
            feature_start = self.sparse_feature_token_offset
            feature_end = feature_start + feature_vocab
            if is_position_slot.any():
                logits[:, is_position_slot, feature_start:feature_end] = -torch.inf
            if (~is_position_slot).any():
                logits[:, ~is_position_slot, :position_vocab] = -torch.inf
                logits[:, ~is_position_slot, eos_id] = -torch.inf

    def _build_sparse_position_rank_map(self) -> torch.Tensor:
        ids = torch.arange(self.sparse_position_vocab_size, dtype=torch.long)
        if self.dense_chunk_order == "xyz":
            return ids
        if self.dense_chunk_order == "xzy":
            coords = oned_to_threed_indices(ids, self.sparse_rope_side_length)
            side = self.sparse_rope_side_length
            return coords[:, 0] * (side * side) + coords[:, 2] * side + coords[:, 1]
        raise ValueError(
            "Monotonic sparse position sampling is only implemented for "
            "chunk_order in {'xyz', 'xzy'}."
        )

    def _apply_monotonic_position_mask(
        self,
        logits: torch.Tensor,
        *,
        idx: torch.Tensor,
        start_pos: int,
        kv_cache=None,
    ) -> None:
        if not self.sparse_position_hard_constraints or kv_cache is None:
            return
        if not self.rope_layout.is_sparse or self.sparse_num_position_tokens != 1:
            return
        if logits.ndim != 3:
            raise ValueError(
                f"logits must have shape (B, T, V), got {tuple(logits.shape)}"
            )
        batch_size, token_count, _ = logits.shape
        if token_count == 0:
            return

        full_len = start_pos + idx.size(1)
        full_tokens = kv_cache.get_tokens(full_len)
        if full_tokens.shape[0] != batch_size:
            raise ValueError(
                "Batch size mismatch between logits and kv token cache "
                f"({batch_size} != {full_tokens.shape[0]})."
            )

        positions = torch.arange(
            start_pos + 1,
            start_pos + token_count + 1,
            device=logits.device,
            dtype=torch.long,
        )
        token_positions = torch.clamp(positions - 1, min=0)
        latent_idx = torch.div(
            token_positions, self.sparse_tokens_per_latent, rounding_mode="floor"
        )
        slot_idx = torch.remainder(token_positions, self.sparse_tokens_per_latent)
        is_position_slot = slot_idx < self.sparse_num_position_tokens
        if not is_position_slot.any():
            return

        prev_latent_idx = latent_idx - 1
        valid_prev_latent = prev_latent_idx >= 0
        if not (valid_prev_latent & is_position_slot).any():
            return

        prev_slot0_pos = 1 + prev_latent_idx * self.sparse_tokens_per_latent
        prev_slot0_pos = prev_slot0_pos.clamp(min=0, max=max(full_len - 1, 0))
        gather_idx = prev_slot0_pos.unsqueeze(0).expand(batch_size, -1)
        prev_pos_ids = full_tokens.gather(1, gather_idx)

        valid_prev = (
            valid_prev_latent.unsqueeze(0)
            & is_position_slot.unsqueeze(0)
            & (prev_pos_ids >= 0)
            & (prev_pos_ids < self.sparse_position_vocab_size)
        )
        if not valid_prev.any():
            return

        rank_map = self.sparse_position_rank_map.to(logits.device)
        safe_prev_ids = prev_pos_ids.masked_fill(~valid_prev, 0)
        prev_ranks = rank_map[safe_prev_ids]

        position_end = self.sparse_position_vocab_size
        for t in range(token_count):
            valid_rows = valid_prev[:, t]
            if not valid_rows.any():
                continue
            thresholds = prev_ranks[valid_rows, t]
            sub = logits[valid_rows, t, :position_end]
            invalid = rank_map.unsqueeze(0) <= thresholds.unsqueeze(1)
            sub[invalid] = -torch.inf
            logits[valid_rows, t, :position_end] = sub

    def _compute_window_sizes(self, config):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (half context)
        """
        pattern = config.window_pattern.upper()
        assert all(
            c in "SL" for c in pattern
        ), f"Invalid window_pattern: {pattern}. Use only S and L."
        # Map characters to window sizes
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
        With sliding windows, effective_seq_len varies per layer (capped by window size).
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
        """
        nparams = sum(p.numel() for p in self.parameters())
        # Exclude non-matmul params: embeddings and per-layer scalars
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        wpe_numel = self.transformer.wpe.weight.numel() if self.learned_pos_embed else 0
        nparams_exclude = (
            self.transformer.wte.weight.numel()
            + wpe_numel
            + value_embeds_numel
            + self.resid_lambdas.numel()
            + self.x0_lambdas.numel()
        )
        h, q, t = (
            self.config.n_head,
            self.config.n_embd // self.config.n_head,
            self.config.sequence_len,
        )
        # Sum attention FLOPs per layer, accounting for sliding window
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]  # (left, right) tuple, we use left
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops
        return num_flops_per_token

    def num_scaling_params(self):
        """
        Return detailed parameter counts for scaling law analysis.
        Different papers use different conventions:
        - Kaplan et al. excluded embedding parameters
        - Chinchilla included all parameters
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper)

        Returns a dict with counts for each parameter group, so downstream analysis
        can experiment with which combination gives the cleanest scaling laws.
        """
        # Count each group separately (mirrors the grouping in setup_optimizers)
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        wpe = (
            sum(p.numel() for p in self.transformer.wpe.parameters())
            if self.learned_pos_embed
            else 0
        )
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + wpe + value_embeds + lm_head + transformer_matrices + scalars
        assert total == sum(
            p.numel() for p in self.parameters()
        ), "Parameter count mismatch"
        return {
            "wte": wte,
            "wpe": wpe,
            "value_embeds": value_embeds,
            "lm_head": lm_head,
            "transformer_matrices": transformer_matrices,
            "scalars": scalars,
            "total": total,
        }

    def setup_optimizer(
        self,
        unembedding_lr=0.004,
        embedding_lr=0.2,
        matrix_lr=0.02,
        weight_decay=0.0,
        adam_betas=(0.8, 0.95),
        scalar_lr=0.5,
    ):
        try:
            from utils.nanochat_optim import DistMuonAdamW, MuonAdamW
        except Exception as exc:
            raise ImportError(
                "nanochat.optim (MuonAdamW) is required for setup_optimizer; "
                "install it or avoid calling setup_optimizer."
            ) from exc

        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Separate out all parameters into groups
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        if self.learned_pos_embed:
            embedding_params += list(self.transformer.wpe.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == len(matrix_params) + len(
            embedding_params
        ) + len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(
            x0_params
        )

        # Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(
            f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}"
        )

        # Build param_groups with all required fields explicit
        param_groups = [
            # AdamW groups (embeddings, lm_head, scalars)
            dict(
                kind="adamw",
                params=lm_head_params,
                lr=unembedding_lr * dmodel_lr_scale,
                betas=adam_betas,
                eps=1e-10,
                weight_decay=0.0,
            ),
            dict(
                kind="adamw",
                params=embedding_params,
                lr=embedding_lr * dmodel_lr_scale,
                betas=adam_betas,
                eps=1e-10,
                weight_decay=0.0,
            ),
            dict(
                kind="adamw",
                params=resid_params,
                lr=scalar_lr * 0.01,
                betas=adam_betas,
                eps=1e-10,
                weight_decay=0.0,
            ),
            dict(
                kind="adamw",
                params=x0_params,
                lr=scalar_lr,
                betas=(0.96, 0.95),
                eps=1e-10,
                weight_decay=0.0,
            ),  # higher beta1 for x0
        ]
        if value_embeds_params:
            param_groups.append(
                dict(
                    kind="adamw",
                    params=value_embeds_params,
                    lr=embedding_lr * dmodel_lr_scale,
                    betas=adam_betas,
                    eps=1e-10,
                    weight_decay=0.0,
                )
            )

        # Muon groups (matrix params, grouped by shape for stacking)
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(
                dict(
                    kind="muon",
                    params=group_params,
                    lr=matrix_lr,
                    momentum=0.95,
                    ns_steps=5,
                    beta2=0.95,
                    weight_decay=weight_decay,
                )
            )

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction="mean"):
        start_pos = 0 if kv_cache is None else kv_cache.get_pos()
        rope = self._get_rope(idx, kv_cache=kv_cache)
        if kv_cache is not None:
            kv_cache.store_tokens(idx, start_pos)

        # Forward the trunk of the Transformer
        x = self.transformer.wte(idx)  # embed current token
        if self.learned_pos_embed:
            wpe_size = self.transformer.wpe.weight.size(0)
            if start_pos + idx.size(1) > wpe_size:
                raise ValueError(
                    f"Position {start_pos + idx.size(1)} exceeds wpe table size {wpe_size}"
                )
            pos = torch.arange(start_pos, start_pos + idx.size(1), device=idx.device)
            x = x + self.transformer.wpe(pos)
        x = norm(x)
        x0 = x  # save initial normalized embedding for x0 residual
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, rope, self.window_sizes[i], kv_cache)
        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15  # smoothly cap the logits to the range [-softcap, softcap]
        logits = self.lm_head(
            x
        )  # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
        logits = logits[..., : self.config.vocab_size]  # slice to remove padding
        logits = logits.float()  # switch to fp32 for logit softcap and loss computation
        logits = softcap * torch.tanh(logits / softcap)  # squash the logits
        self._apply_slot_logits_mask(logits, start_pos=start_pos)
        if targets is None and kv_cache is not None:
            self._apply_monotonic_position_mask(
                logits, idx=idx, start_pos=start_pos, kv_cache=kv_cache
            )

        if targets is not None:
            # training: given the targets, compute and return the loss
            # TODO experiment with chunked cross-entropy?
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=loss_reduction,
            )
            return loss
        else:
            # inference: just return the logits directly
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        raise NotImplementedError(
            "This method should no longer be used. Use the engine instead."
        )

        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)  # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids)  # (B, T, vocab_size)
            logits = logits[:, -1, :]  # (B, vocab_size)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
