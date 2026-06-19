from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import imageio.v3 as iio
import torch

from model.gaussian_gpt import GaussianGPT
from model.gaussian_vqvae import GaussianVQVAE
from utils.pos_tokens import pos_tokens_to_centered_coords
from utils.render import GaussianScene, render

TOPDOWN_FIT_MARGIN = 1.05


def _to_cpu_payload(payload: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu() if torch.is_tensor(value) else value
        for key, value in payload.items()
    }


def _load_checkpoint_from_output(output_dir: Path) -> Optional[str]:
    manifest_path = output_dir / "manifest.pt"
    if manifest_path.exists():
        manifest = torch.load(manifest_path, map_location="cpu", weights_only=False)
        if isinstance(manifest, dict):
            metadata = manifest.get("metadata", {})
            if isinstance(metadata, dict):
                checkpoint = metadata.get("checkpoint")
                if checkpoint:
                    return str(checkpoint)

    shards_dir = output_dir / "shards"
    if shards_dir.exists():
        shard_paths = sorted(shards_dir.glob("rank_*.pt"))
        for shard_path in shard_paths:
            shard = torch.load(shard_path, map_location="cpu", weights_only=False)
            if not isinstance(shard, dict):
                continue
            metadata = shard.get("metadata", {})
            if not isinstance(metadata, dict):
                continue
            checkpoint = metadata.get("checkpoint")
            if checkpoint:
                return str(checkpoint)

    return None


@torch.no_grad()
def _decode_tokens_to_payload(
    gpt: GaussianGPT, tokens: torch.Tensor
) -> Dict[str, torch.Tensor]:
    if tokens.numel() == 0:
        return gpt._empty_scene_dict(tokens.device)

    if gpt.dense_chunks:
        return gpt._decode_dense_tokens(tokens)

    tokens_per_latent = int(gpt.tokens_per_latent)
    usable_len = (tokens.numel() // tokens_per_latent) * tokens_per_latent
    if usable_len == 0:
        return gpt._empty_scene_dict(tokens.device)
    if usable_len != tokens.numel():
        tokens = tokens[:usable_len]

    token_rows = tokens.view(-1, tokens_per_latent)
    num_pos_tokens = int(gpt.num_position_tokens)
    position_vocab_size = int(gpt.position_vocab_size)
    feature_vocab_size = int(gpt.feature_vocab_size)
    feature_offset = int(gpt.feature_token_offset)

    pos_tokens = token_rows[:, :num_pos_tokens]
    feature_ids = token_rows[:, num_pos_tokens:]

    pad_id = gpt.gpt.pad_token_id
    if pad_id is None:
        all_pad = torch.zeros(
            token_rows.shape[0], device=tokens.device, dtype=torch.bool
        )
    else:
        all_pad = (feature_ids == int(pad_id)).all(dim=1)

    pos_invalid = (pos_tokens < 0) | (pos_tokens >= position_vocab_size)
    feature_out_of_range = (feature_ids < feature_offset) | (
        feature_ids >= feature_offset + feature_vocab_size
    )
    feature_invalid = (~all_pad).unsqueeze(1) & feature_out_of_range
    invalid = (~all_pad) & (pos_invalid.any(dim=1) | feature_invalid.any(dim=1))

    if invalid.any():
        token_rows = token_rows.clone()
        pos_tokens = token_rows[:, :num_pos_tokens]
        feature_ids = token_rows[:, num_pos_tokens:]
        pos_tokens = pos_tokens.masked_fill(pos_invalid, 0)
        feature_ids = feature_ids.masked_fill(feature_invalid, feature_offset)
        token_rows[:, :num_pos_tokens] = pos_tokens
        token_rows[:, num_pos_tokens:] = feature_ids
        pos_tokens = token_rows[:, :num_pos_tokens]
        feature_ids = token_rows[:, num_pos_tokens:]

    valid_mask = ~all_pad
    if not valid_mask.any():
        return gpt._empty_scene_dict(tokens.device)

    coords = pos_tokens_to_centered_coords(
        pos_tokens[valid_mask],
        num_pos_tokens,
        int(gpt.position_side_length),
    ).to(dtype=torch.long)
    raw_feature_ids = (feature_ids[valid_mask] - feature_offset).to(dtype=torch.long)
    return gpt.vqvae.decode(coords, raw_feature_ids)


@torch.no_grad()
def _decode_largesampling_sparse_payload(
    gpt: GaussianGPT,
    payload: Dict[str, torch.Tensor],
    *,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    tokens = payload["tokens_sequence"]
    column_lengths = payload["column_token_lengths"]
    if not torch.is_tensor(tokens) or not torch.is_tensor(column_lengths):
        raise ValueError("invalid payload tensors")
    if column_lengths.ndim != 2:
        raise ValueError("column_token_lengths must be 2D")

    tokens = tokens.to(device=device, dtype=torch.long)
    column_lengths = column_lengths.to(device=device, dtype=torch.long)

    scene_cols_x, scene_cols_y = [int(v) for v in column_lengths.shape]
    if gpt.chunk_shape is None or len(gpt.chunk_shape) != 3:
        raise ValueError("Sparse decode requires 3D chunk_shape in GPT checkpoint.")
    z = int(gpt.chunk_shape[2])
    if z < 1:
        raise ValueError(f"Invalid z from chunk_shape: {gpt.chunk_shape}")

    tokens_per_latent = int(gpt.tokens_per_latent)
    num_pos_tokens = int(gpt.num_position_tokens)
    num_feature_tokens = int(tokens_per_latent - num_pos_tokens)
    if num_feature_tokens < 1:
        raise ValueError("Expected at least one feature token per latent.")

    feature_offset = int(gpt.feature_token_offset)
    feature_vocab_size = int(gpt.feature_vocab_size)
    position_vocab_size = int(gpt.position_vocab_size)
    side = int(gpt.position_side_length)
    pos_token_ids = torch.arange(position_vocab_size, device=device, dtype=torch.long)
    lut_z = pos_token_ids % side

    feature_grid = torch.full(
        (scene_cols_x, scene_cols_y, z, num_feature_tokens),
        -1,
        device=device,
        dtype=torch.long,
    )
    cursor = 0

    for x in range(scene_cols_x):
        for y in range(scene_cols_y):
            col_len = int(column_lengths[x, y].item())
            if col_len <= 0:
                continue
            if cursor + col_len > int(tokens.numel()):
                raise ValueError(
                    f"Token stream is shorter than expected from column lengths at ({x}, {y})."
                )

            col_tokens = tokens[cursor : cursor + col_len]
            cursor += col_len

            usable_len = (
                int(col_tokens.numel()) // tokens_per_latent
            ) * tokens_per_latent
            if usable_len <= 0:
                continue
            token_rows = col_tokens[:usable_len].view(-1, tokens_per_latent)
            pos_tokens = token_rows[:, :num_pos_tokens]
            feature_tokens = token_rows[:, num_pos_tokens:]

            pos_invalid = (pos_tokens < 0) | (pos_tokens >= position_vocab_size)
            if num_pos_tokens == 1:
                z_idx = lut_z[pos_tokens[:, 0]]
                z_invalid = pos_invalid[:, 0]
            else:
                local_coords = pos_tokens_to_centered_coords(
                    pos_tokens,
                    num_pos_tokens,
                    int(gpt.position_side_length),
                ).to(dtype=torch.long)
                z_idx = local_coords[:, 2]
                z_invalid = pos_invalid.any(dim=1)

            feature_out_of_range = (feature_tokens < feature_offset) | (
                feature_tokens >= feature_offset + feature_vocab_size
            )
            feature_tokens = feature_tokens.masked_fill(
                feature_out_of_range, feature_offset
            )
            raw_feature_tokens = (feature_tokens - feature_offset).to(dtype=torch.long)

            valid_rows = (~z_invalid) & (z_idx >= 0) & (z_idx < z)
            if not valid_rows.any():
                continue

            for row_idx in torch.nonzero(valid_rows, as_tuple=False).flatten():
                z_val = int(z_idx[row_idx].item())
                feature_grid[x, y, z_val] = raw_feature_tokens[row_idx]

    if cursor != int(tokens.numel()):
        raise ValueError(
            f"Token stream length mismatch: consumed={cursor}, total={int(tokens.numel())}."
        )

    valid_mask = feature_grid[..., 0] >= 0
    if not bool(valid_mask.any().item()):
        return gpt._empty_scene_dict(device)

    coords = valid_mask.nonzero(as_tuple=False).to(dtype=torch.long)
    feature_ids = feature_grid[valid_mask]
    return gpt.vqvae.decode(coords, feature_ids)


@torch.no_grad()
def _decode_scene_payload(
    gpt: GaussianGPT,
    payload: Dict[str, torch.Tensor],
    *,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    if not isinstance(payload, dict) or "tokens_sequence" not in payload:
        raise ValueError("missing tokens_sequence")
    tokens = payload["tokens_sequence"]
    if not torch.is_tensor(tokens):
        raise ValueError("tokens_sequence is not a tensor")

    if (
        not bool(getattr(gpt, "dense_chunks", False))
    ) and "column_token_lengths" in payload:
        return _decode_largesampling_sparse_payload(gpt, payload, device=device)

    tokens = tokens.to(device=device, dtype=torch.long)
    return _decode_tokens_to_payload(gpt, tokens)


def _filter_scene_by_z_quantile(
    payload: Dict[str, torch.Tensor], quantile: float
) -> Tuple[Dict[str, torch.Tensor], float]:
    coords = payload.get("coords")
    if coords is None:
        raise ValueError("Decoded payload has no 'coords' key.")
    if coords.numel() == 0:
        raise ValueError("Decoded scene is empty.")

    threshold = float(torch.quantile(coords[:, 2], float(quantile) / 100.0).item())
    keep_mask = coords[:, 2] <= threshold
    total_points = int(coords.shape[0])
    kept_points = int(keep_mask.sum().item())
    if kept_points == 0:
        raise ValueError("No points remain after z-quantile filtering.")

    filtered: Dict[str, torch.Tensor] = {}
    for key, value in payload.items():
        if (
            torch.is_tensor(value)
            and value.dim() > 0
            and value.shape[0] == total_points
        ):
            filtered[key] = value[keep_mask]
        else:
            filtered[key] = value
    return filtered, threshold


def _center_scene_xy(payload: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    centered = dict(payload)
    coords = centered["coords"].clone()
    min_xy = coords[:, :2].amin(dim=0)
    max_xy = coords[:, :2].amax(dim=0)
    xy_center = 0.5 * (min_xy + max_xy)
    coords[:, 0] -= xy_center[0]
    coords[:, 1] -= xy_center[1]
    centered["coords"] = coords
    return centered


def _build_topdown_view_matrix(
    camera_pos: torch.Tensor,
    look_at: torch.Tensor,
    *,
    up_world: torch.Tensor,
) -> torch.Tensor:
    fwd = torch.nn.functional.normalize(look_at - camera_pos, dim=0)
    right = torch.nn.functional.normalize(torch.cross(fwd, up_world, dim=0), dim=0)
    up_cam = torch.cross(right, fwd, dim=0)
    rot = torch.stack([right, up_cam, fwd], dim=1)
    trans = -(rot.transpose(0, 1) @ camera_pos)

    view = torch.eye(4, device=camera_pos.device, dtype=torch.float32)
    view[:3, :3] = rot.transpose(0, 1)
    view[:3, 3] = trans
    return view


def _camera_for_topdown(
    payload: Dict[str, torch.Tensor],
    resolution: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    coords = payload["coords"]
    device = coords.device
    half_x = float(coords[:, 0].abs().amax().item())
    half_y = float(coords[:, 1].abs().amax().item())
    half_extent = max(half_x, half_y, 1e-6)

    z_min = float(coords[:, 2].amin().item())
    z_max = float(coords[:, 2].amax().item())
    z_look = 0.5 * (z_min + z_max)
    top_clearance = z_max - z_look

    focal = 0.9 * float(resolution)
    principal = 0.5 * float(resolution)
    distance_xy = half_extent * focal / principal
    distance_fit = max(distance_xy * TOPDOWN_FIT_MARGIN, 1e-4)
    camera_z = z_look + top_clearance + distance_fit

    camera_pos = torch.tensor([0.0, 0.0, camera_z], device=device, dtype=torch.float32)
    look_at = torch.tensor([0.0, 0.0, z_look], device=device, dtype=torch.float32)
    up_world = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=torch.float32)

    view = _build_topdown_view_matrix(camera_pos, look_at, up_world=up_world)
    intrinsics = torch.zeros((1, 3, 3), device=device, dtype=torch.float32)
    intrinsics[0, 0, 0] = focal
    intrinsics[0, 1, 1] = focal
    intrinsics[0, 0, 2] = principal
    intrinsics[0, 1, 2] = principal
    intrinsics[0, 2, 2] = 1.0
    return view.unsqueeze(0), intrinsics


def _render_topdown_png(
    decoded_scene: Dict[str, torch.Tensor],
    *,
    output_path: Path,
    quantile: float,
    resolution: int,
    background_color: str,
) -> None:
    filtered, _ = _filter_scene_by_z_quantile(decoded_scene, quantile=quantile)
    centered = _center_scene_xy(filtered)
    scene = GaussianScene.from_dict(centered)
    view_mats, intrinsics = _camera_for_topdown(centered, int(resolution))
    rendered, _ = render(
        scene,
        view_mats,
        intrinsics=intrinsics,
        render_size=(int(resolution), int(resolution)),
        background_color=background_color,
    )
    image = rendered[0].permute(1, 2, 0).mul(255.0).clamp(0, 255).byte().cpu().numpy()
    iio.imwrite(output_path, image)


def _iter_token_files(tokens_root: Path) -> List[Path]:
    return sorted(tokens_root.glob("rank_*/*.pt"))


def _resolve_torchrun_env() -> Tuple[int, int, int]:
    rank_env = os.getenv("RANK")
    local_rank_env = os.getenv("LOCAL_RANK")
    world_size_env = os.getenv("WORLD_SIZE")
    if rank_env is None and local_rank_env is None and world_size_env is None:
        return 0, 0, 1
    if rank_env is None or local_rank_env is None or world_size_env is None:
        raise RuntimeError(
            "Incomplete torchrun environment; expected RANK, LOCAL_RANK, WORLD_SIZE."
        )

    rank = int(rank_env)
    local_rank = int(local_rank_env)
    world_size = int(world_size_env)
    if world_size < 1:
        raise ValueError("WORLD_SIZE must be >= 1.")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"Invalid RANK={rank} for WORLD_SIZE={world_size}.")
    if local_rank < 0:
        raise ValueError(f"Invalid LOCAL_RANK={local_rank}.")
    return rank, local_rank, world_size


def _shard_range(total: int, rank: int, world_size: int) -> Tuple[int, int]:
    base = total // world_size
    extra = total % world_size
    local_n = base + int(rank < extra)
    start = rank * base + min(rank, extra)
    end = start + local_n
    return start, end


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decode all token sidecars from generate_scene.py "
            "outputs into Gaussian scene payloads, overwriting existing decodes."
        )
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Path that contains tokens/, shards/, and optionally manifest.pt.",
    )
    parser.add_argument(
        "--vqvae-checkpoint",
        required=True,
        type=Path,
        help="VQ-VAE checkpoint to use for decoding.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        type=Path,
        help="GPT checkpoint override. Defaults to metadata checkpoint from manifest/shards.",
    )
    parser.add_argument(
        "--device",
        default=None,
        type=str,
        help="Torch device (e.g. cuda, cuda:0, cpu). Default: cuda if available else cpu.",
    )
    parser.add_argument(
        "--log-every",
        default=25,
        type=int,
        help="Print progress every N scenes.",
    )
    parser.add_argument(
        "--render-topdown",
        action="store_true",
        help="Also regenerate topdown PNGs under output_dir/topdown/.",
    )
    parser.add_argument(
        "--topdown-quantile",
        default=75.0,
        type=float,
        help="Keep points with z <= this percentile before rendering.",
    )
    parser.add_argument(
        "--topdown-resolution",
        default=1024,
        type=int,
        help="Square output resolution for topdown PNGs.",
    )
    parser.add_argument(
        "--topdown-background-color",
        default="white",
        choices=["white", "black"],
        type=str,
        help="Topdown render background color.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rank, local_rank, world_size = _resolve_torchrun_env()
    output_dir = args.output_dir.expanduser().resolve(strict=False)
    vqvae_checkpoint = args.vqvae_checkpoint.expanduser().resolve(strict=False)

    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")
    if not vqvae_checkpoint.exists():
        raise FileNotFoundError(f"VQ-VAE checkpoint not found: {vqvae_checkpoint}")
    if args.topdown_quantile < 0.0 or args.topdown_quantile > 100.0:
        raise ValueError("topdown_quantile must be in [0, 100].")
    if args.topdown_resolution < 1:
        raise ValueError("topdown_resolution must be >= 1.")

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        inferred_checkpoint = _load_checkpoint_from_output(output_dir)
        if inferred_checkpoint is None:
            raise ValueError(
                "Could not infer GPT checkpoint from manifest/shards. "
                "Pass --checkpoint explicitly."
            )
        checkpoint_path = Path(inferred_checkpoint)
    checkpoint_path = checkpoint_path.expanduser().resolve(strict=False)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"GPT checkpoint not found: {checkpoint_path}")

    tokens_root = output_dir / "tokens"
    if not tokens_root.exists():
        raise FileNotFoundError(f"Tokens directory not found: {tokens_root}")
    token_files = _iter_token_files(tokens_root)
    if not token_files:
        raise FileNotFoundError(f"No token sidecar files found in: {tokens_root}")
    start, end = _shard_range(len(token_files), rank, world_size)
    local_token_files = token_files[start:end]

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(local_rank)
    else:
        device = torch.device("cpu")

    print(
        f"[rank {rank}] assigned {len(local_token_files)} scene token file(s) "
        f"in global range [{start}, {end}) of {len(token_files)}.",
        flush=True,
    )
    print(f"[rank {rank}] Loading VQ-VAE: {vqvae_checkpoint}", flush=True)
    vqvae = GaussianVQVAE.load_from_checkpoint(str(vqvae_checkpoint)).eval()

    print(f"[rank {rank}] Loading GPT checkpoint: {checkpoint_path}", flush=True)
    gpt = GaussianGPT.load_from_checkpoint(str(checkpoint_path), vqvae=vqvae).eval()
    gpt = gpt.to(device)

    gaussians_root = output_dir / "gaussians"
    gaussians_root.mkdir(parents=True, exist_ok=True)
    topdown_root = output_dir / "topdown"
    if args.render_topdown:
        topdown_root.mkdir(parents=True, exist_ok=True)

    decode_failures: List[str] = []
    topdown_skips: List[str] = []
    total = len(local_token_files)
    print(
        f"[rank {rank}] Decoding {total} scene token file(s) on device={device}.",
        flush=True,
    )

    for idx, token_path in enumerate(local_token_files, start=1):
        rel_path = token_path.relative_to(tokens_root)
        output_path = gaussians_root / rel_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            payload = torch.load(token_path, map_location="cpu", weights_only=False)
            decoded = _decode_scene_payload(gpt, payload, device=device)
            torch.save(_to_cpu_payload(decoded), output_path)
            if args.render_topdown:
                topdown_output_path = topdown_root / rel_path.with_suffix(".png")
                topdown_output_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    _render_topdown_png(
                        decoded,
                        output_path=topdown_output_path,
                        quantile=float(args.topdown_quantile),
                        resolution=int(args.topdown_resolution),
                        background_color=str(args.topdown_background_color),
                    )
                except ValueError as exc:
                    topdown_skips.append(f"{rel_path}: {exc}")
        except Exception as exc:  # pylint: disable=broad-except
            decode_failures.append(f"{rel_path}: {exc}")

        if idx % max(1, int(args.log_every)) == 0 or idx == total:
            print(f"[rank {rank}] Decoded {idx}/{total}", flush=True)

        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.empty_cache()

    if decode_failures:
        print(
            f"[rank {rank}] Finished with {len(decode_failures)} failure(s). First few:",
            flush=True,
        )
        for message in decode_failures[:10]:
            print(f"  - {message}", flush=True)
    else:
        print(f"[rank {rank}] Finished with 0 failures.", flush=True)
    if topdown_skips:
        print(
            f"[rank {rank}] Topdown skipped for {len(topdown_skips)} scene(s). First few:",
            flush=True,
        )
        for message in topdown_skips[:10]:
            print(f"  - {message}", flush=True)


if __name__ == "__main__":
    main()
