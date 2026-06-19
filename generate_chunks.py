from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning
import torch
from omegaconf import DictConfig, OmegaConf

from data.vfront_dataset import VFrontPreprocessedDataModule
from model.gaussian_gpt import GaussianGPT
from model.gaussian_vqvae import GaussianVQVAE
from utils.pos_tokens import pos_tokens_to_centered_coords
from utils.render import (
    GaussianScene,
    center_scene_aabb,
    render_and_save_trajectory,
    render_and_save_trajectory_strip,
)


def _path_matches(left: Optional[str], right: Optional[str]) -> bool:
    if not left or not right:
        return False
    left_norm = str(Path(left).expanduser().resolve(strict=False))
    right_norm = str(Path(right).expanduser().resolve(strict=False))
    return left_norm == right_norm


def _resolve_completion_data_cfg(
    data_cfg: Dict[str, Any],
    *,
    dataset: str,
    vqvae_checkpoint: Optional[str],
) -> Dict[str, Any]:
    if data_cfg.get("dataset_name") == dataset and data_cfg.get("vqvae_path"):
        return data_cfg

    conf_data_dir = Path(__file__).resolve().parent / "conf" / "data"
    if conf_data_dir.exists():
        for config_path in sorted(conf_data_dir.glob("tokenized_*.yaml")):
            candidate = OmegaConf.to_container(
                OmegaConf.load(config_path),
                resolve=True,
            )
            if not isinstance(candidate, dict):
                continue
            if candidate.get("dataset_name") != dataset:
                continue
            if _path_matches(candidate.get("vqvae_path"), vqvae_checkpoint):
                print(
                    f"Completion data config auto-resolved to {config_path.name}.",
                    flush=True,
                )
                return candidate

    raise ValueError(
        "Completion requires tokenized data. Either pass a tokenized data config "
        "(e.g. `data=tokenized_vfront`) or ensure one tokenized config in "
        "`conf/data` has matching `dataset_name` and `vqvae_path`."
    )


def _center_scene_reference(
    scene: GaussianScene, reference: GaussianScene
) -> GaussianScene:
    coords = reference.means
    if coords.numel() == 0:
        return scene
    min_vals, max_vals = torch.aminmax(coords, dim=0)
    center = 0.5 * (min_vals + max_vals)
    return GaussianScene(
        means=scene.means - center,
        sh0=scene.sh0,
        sh=scene.sh,
        opacities=scene.opacities,
        scales=scene.scales,
        quats=scene.quats,
    )


def _decode_tokens_to_scene(
    gpt: GaussianGPT,
    tokens: torch.Tensor,
    *,
    base_side_length: int,
    pad_id: int,
) -> GaussianScene:
    if tokens.numel() == 0:
        return GaussianScene.from_dict(gpt._empty_scene_dict(tokens.device))

    if gpt.dense_chunks:
        decoded = gpt._decode_dense_tokens(tokens)
        return GaussianScene.from_dict(decoded)

    tokens_per_latent = gpt.tokens_per_latent
    usable_len = (tokens.numel() // tokens_per_latent) * tokens_per_latent
    if usable_len == 0:
        return GaussianScene.from_dict(gpt._empty_scene_dict(tokens.device))
    if usable_len != tokens.numel():
        tokens = tokens[:usable_len]

    token_rows = tokens.view(-1, tokens_per_latent)
    num_pos_tokens = gpt.num_position_tokens
    position_vocab_size = gpt.position_vocab_size
    feature_vocab_size = gpt.feature_vocab_size
    feature_offset = gpt.feature_token_offset
    pos_tokens = token_rows[:, :num_pos_tokens]
    feature_ids = token_rows[:, num_pos_tokens:]
    all_pad = (feature_ids == pad_id).all(dim=1)
    pos_invalid = (pos_tokens < 0) | (pos_tokens >= position_vocab_size)
    feature_out_of_range = (feature_ids < feature_offset) | (
        feature_ids >= feature_offset + feature_vocab_size
    )
    feature_invalid = (~all_pad).unsqueeze(1) & feature_out_of_range
    invalid = (~all_pad) & (pos_invalid.any(dim=1) | feature_invalid.any(dim=1))
    if invalid.any():
        print(
            "WARNING: Sparse tokens contain invalid ids; replacing with safe defaults.",
            flush=True,
        )
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
        return GaussianScene.from_dict(gpt._empty_scene_dict(tokens.device))

    coords = pos_tokens_to_centered_coords(
        pos_tokens[valid_mask],
        num_pos_tokens,
        base_side_length,
    ).to(dtype=torch.long)
    raw_feature_ids = (feature_ids[valid_mask] - feature_offset).to(dtype=torch.long)
    decoded = gpt.vqvae.decode(coords, raw_feature_ids)
    return GaussianScene.from_dict(decoded)


def _build_completion_dataset(
    data_cfg: Dict[str, Any],
    gpt: GaussianGPT,
    *,
    split: str,
) -> torch.utils.data.Dataset:
    dataset_name = str(data_cfg.get("dataset_name", "")).lower()
    if dataset_name not in {"vfront", "ase", "photoshape"}:
        raise ValueError(
            "Completion currently supports 'vfront', 'ase', and 'photoshape', "
            f"got '{dataset_name}'."
        )
    if data_cfg.get("sort_latents") is not None:
        raise ValueError(
            "`data.sort_latents` is deprecated outside tokenization; use `model.chunk_order`."
        )
    chunk_shape = getattr(gpt, "chunk_shape", None)
    if chunk_shape is None:
        raise ValueError(
            "Completion requires chunked GPT checkpoints; `model.chunk_shape` is missing."
        )
    if isinstance(chunk_shape, int):
        chunk_shape = [chunk_shape] * 3
    if len(chunk_shape) != 3 or any(int(v) <= 0 for v in chunk_shape):
        raise ValueError(
            "Completion requires `model.chunk_shape` to be 3 positive integers."
        )

    preprocessed_kwargs = dict(
        data_path=data_cfg.get("data_path"),
        train_list_path=data_cfg.get("train_split"),
        val_list_path=data_cfg.get("val_split"),
        dataloader_kwargs={"batch_size": 1},
        overfit_scenes=0,
        overfit_epoch_size=1000,
        verbose=False,
        background_color=data_cfg.get("background_color", "white"),
        num_position_tokens=gpt.num_position_tokens,
        position_vocab_size=getattr(gpt, "position_vocab_size", None),
        codebook_size=gpt.vqvae.autoencoder.vq.codebook_size,
        shared=getattr(gpt, "shared_vocab", False),
        chunk_shape=chunk_shape,
        dense_chunks=getattr(gpt, "dense_chunks", False),
        chunk_order=getattr(gpt, "chunk_order", "xyz"),
        min_chunk_occupancy=data_cfg.get("min_chunk_occupancy", 0.0),
        max_chunk_attempts=data_cfg.get("max_chunk_attempts", 1),
        chunk_origin=data_cfg.get("chunk_origin"),
        load_augmented_tokens=bool(data_cfg.get("load_augmented_tokens", False)),
    )
    preprocessed_subpath = data_cfg.get("preprocessed_subpath")
    if preprocessed_subpath is not None:
        preprocessed_kwargs["preprocessed_subpath"] = preprocessed_subpath

    data_module = VFrontPreprocessedDataModule(**preprocessed_kwargs)
    data_module.setup()
    dataset = data_module.train_dataset if split == "train" else data_module.val_dataset
    if dataset is None:
        raise RuntimeError(f"Failed to build {split} dataset for completion.")
    return dataset


@torch.no_grad()
def sample_and_evaluate(
    dataset: Optional[str],
    output_dir: Path,
    num_samples: int,
    batch_size: int,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
    max_length: Optional[int],
    views: int,
    high_size: Tuple[int, int],
    low_size: Tuple[int, int],
    background_color: str,
    skip_metrics: bool,
    gif_frames: int,
    gif_fps: int,
    vqvae_checkpoint: Optional[str],
    store_samples: bool,
    data_cfg: Dict[str, Any],
    checkpoint_path: Optional[Path] = None,
    remove_checkpoint: bool = False,
    completion_cfg: Optional[Dict[str, Any]] = None,
    gpt: Optional[GaussianGPT] = None,
    vqvae: Optional[GaussianVQVAE] = None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if gpt is None:
        if checkpoint_path is None:
            raise ValueError(
                "checkpoint_path is required when gpt model is not provided."
            )
        # pylint: disable=no-value-for-parameter
        vqvae = None
        if vqvae_checkpoint:
            print(f"Loading VQ-VAE from checkpoint: {vqvae_checkpoint}", flush=True)
            vqvae = GaussianVQVAE.load_from_checkpoint(vqvae_checkpoint).eval()
            vqvae.set_background_color(background_color)

        print(f"Loading GaussianGPT from checkpoint: {checkpoint_path}", flush=True)
        gpt = (
            GaussianGPT.load_from_checkpoint(str(checkpoint_path), vqvae=vqvae)
            .eval()
            .to(device)
        )
        if getattr(gpt, "vqvae", None) is not None:
            gpt.vqvae.set_background_color(background_color)
    else:
        if vqvae is None:
            vqvae = getattr(gpt, "vqvae", None)
        if vqvae is not None:
            vqvae.set_background_color(background_color)
        gpt = gpt.eval().to(device)

    gif_dir = output_dir / "gif"
    gif_dir.mkdir(parents=True, exist_ok=True)

    samples_dir = output_dir / "samples" if store_samples else None
    if samples_dir is not None:
        samples_dir.mkdir(parents=True, exist_ok=True)

    completion_cfg = completion_cfg or {}
    completion_enabled = bool(completion_cfg.get("enabled", False))

    if completion_enabled:
        if dataset is None:
            raise ValueError(
                "Completion requires a dataset name. Set `dataset=<name>` or provide "
                "`data.dataset_name`."
            )
        completion_split = completion_cfg.get("split", "val")
        prompt_fraction = float(completion_cfg.get("prompt_fraction", 0.5))
        completion_data_cfg = _resolve_completion_data_cfg(
            data_cfg,
            dataset=dataset,
            vqvae_checkpoint=vqvae_checkpoint,
        )

        print(
            f"Completing {num_samples} scenes from '{completion_split}' split "
            f"with prompt_fraction={prompt_fraction:.2f} "
            f"as gifs ({gif_frames} frames @ {gif_fps} fps).",
            flush=True,
        )

        completion_dataset = _build_completion_dataset(
            completion_data_cfg,
            gpt,
            split=completion_split,
        )
        total_samples = min(num_samples, len(completion_dataset))
        if total_samples <= 0:
            raise ValueError("No samples available for completion.")

        indices = list(range(len(completion_dataset)))
        indices = indices[:total_samples]

        tokens_per_latent = gpt.tokens_per_latent
        pad_id = gpt.gpt.pad_token_id
        base_side_length = gpt.position_side_length

        total_start = time.perf_counter()
        global_sample_idx = 0

        for data_idx in indices:
            tokens = completion_dataset[data_idx]
            if not torch.is_tensor(tokens):
                tokens = torch.tensor(tokens, dtype=torch.long)
            tokens = tokens.to(device=device, dtype=torch.long)
            full_len = tokens.numel()
            if full_len == 0:
                gt_scene = GaussianScene.from_dict(gpt._empty_scene_dict(tokens.device))
                partial_scene = gt_scene
                completed_scene = gt_scene
            else:
                num_latents = full_len // tokens_per_latent
                prefix_latents = int(num_latents * prompt_fraction)
                prefix_len = prefix_latents * tokens_per_latent
                prefix_tokens = tokens[:prefix_len]
                remaining_len = full_len - prefix_len

                if remaining_len > 0:
                    scene_seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
                    completed_tokens = gpt.gpt.sample_sequence_with_prompt(
                        prefix_tokens,
                        max_new_tokens=remaining_len,
                        num_samples=1,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        stop_on_eos=False,
                        seed=scene_seed,
                    )[0]
                else:
                    completed_tokens = prefix_tokens

                if gpt.dense_chunks:
                    if completed_tokens.numel() > full_len:
                        completed_tokens = completed_tokens[:full_len]
                    elif completed_tokens.numel() < full_len:
                        pad = torch.full(
                            (full_len - completed_tokens.numel(),),
                            pad_id,
                            device=device,
                            dtype=torch.long,
                        )
                        completed_tokens = torch.cat([completed_tokens, pad], dim=0)
                    partial_tokens = tokens.clone()
                    partial_tokens[prefix_len:] = pad_id
                else:
                    partial_tokens = prefix_tokens

                printed = 0
                for i in range(remaining_len):
                    tok_i = tokens[prefix_len + i]
                    com_i = completed_tokens[prefix_len + i]
                    if tok_i != com_i:
                        print(
                            i,
                            prefix_len + i,
                            f"GT token {tok_i.item()} != completed token {com_i.item()}",
                        )
                        printed += 1
                        if printed >= 10:
                            break

                gt_scene = _decode_tokens_to_scene(
                    gpt,
                    tokens,
                    base_side_length=base_side_length,
                    pad_id=pad_id,
                )
                partial_scene = _decode_tokens_to_scene(
                    gpt,
                    partial_tokens,
                    base_side_length=base_side_length,
                    pad_id=pad_id,
                )
                completed_scene = _decode_tokens_to_scene(
                    gpt,
                    completed_tokens,
                    base_side_length=base_side_length,
                    pad_id=pad_id,
                )

            render_gt = _center_scene_reference(gt_scene, gt_scene)
            render_partial = _center_scene_reference(partial_scene, gt_scene)
            render_completed = _center_scene_reference(completed_scene, gt_scene)

            gif_path = gif_dir / f"sample_{global_sample_idx:04d}.gif"
            render_and_save_trajectory_strip(
                [render_partial, render_completed, render_gt],
                str(gif_path),
                num_frames=gif_frames,
                fps=gif_fps,
                background_color=background_color,
            )

            if samples_dir is not None:
                for label, scene in (
                    ("partial", render_partial),
                    ("completed", render_completed),
                    ("gt", render_gt),
                ):
                    sample_stem = f"sample_{global_sample_idx:04d}_{label}.pt"
                    sample_path = samples_dir / sample_stem
                    payload = {
                        key: value.detach().cpu() if torch.is_tensor(value) else value
                        for key, value in scene.to_dict().items()
                    }
                    torch.save(payload, sample_path)

            global_sample_idx += 1

        elapsed = time.perf_counter() - total_start
        print(
            f"Completed {total_samples} scenes in {elapsed:.2f}s "
            f"({elapsed / max(total_samples, 1):.3f}s per scene).",
            flush=True,
        )
    else:
        print(
            f"Sampling {num_samples} scenes "
            f"(batch size {batch_size}, temperature {temperature}, top_k {top_k}, top_p {top_p}) "
            f"as gifs ({gif_frames} frames @ {gif_fps} fps).",
            flush=True,
        )
        total_samples = num_samples
        num_batches = math.ceil(total_samples / batch_size)

        total_start = time.perf_counter()
        global_sample_idx = 0
        avg_batch_time = None

        for batch_idx in range(num_batches):
            current_batch_size = min(batch_size, total_samples - global_sample_idx)
            batch_start = time.perf_counter()
            batch_seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
            batch_samples, lengths = gpt.sample(
                max_length=max_length,
                num_samples=current_batch_size,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                return_lengths=True,
                seed=batch_seed,
            )
            if lengths is not None:
                lengths_list = lengths.detach().cpu().tolist()
                print(f"Generated token lengths: {lengths_list}", flush=True)
            batch_time = time.perf_counter() - batch_start
            avg_batch_time = (
                batch_time
                if avg_batch_time is None
                else 0.9 * avg_batch_time + 0.1 * batch_time
            )
            elapsed = time.perf_counter() - total_start
            remaining_batches = num_batches - (batch_idx + 1)
            eta = remaining_batches * avg_batch_time
            print(
                f"Batch {batch_idx + 1}/{num_batches} "
                f"completed in {batch_time:.2f}s (sampled {current_batch_size} scenes). "
                f"Elapsed {elapsed:.1f}s, ETA ~{eta:.1f}s.",
                flush=True,
            )

            scenes = _normalize_samples(batch_samples)

            for scene in scenes:
                render_scene = center_scene_aabb(scene)
                gif_path = gif_dir / f"sample_{global_sample_idx:04d}.gif"
                render_and_save_trajectory(
                    render_scene,
                    str(gif_path),
                    num_frames=gif_frames,
                    fps=gif_fps,
                    background_color=background_color,
                )

                if samples_dir is not None:
                    sample_stem = f"sample_{global_sample_idx:04d}.pt"
                    sample_path = samples_dir / sample_stem
                    payload = {
                        key: value.detach().cpu() if torch.is_tensor(value) else value
                        for key, value in scene.to_dict().items()
                    }
                    torch.save(payload, sample_path)

                global_sample_idx += 1

        elapsed = time.perf_counter() - total_start
        print(
            f"Generated {total_samples} samples in {elapsed:.2f}s "
            f"({elapsed / max(total_samples, 1):.3f}s per scene).",
            flush=True,
        )

    if skip_metrics:
        print("Skipping metric evaluation (skip_metrics=True).", flush=True)
    else:
        print(
            "Skipping metric evaluation because evaluation renders gifs only.",
            flush=True,
        )

    if remove_checkpoint and checkpoint_path is not None and checkpoint_path.exists():
        checkpoint_path.unlink()
        print(
            f"Removed checkpoint {checkpoint_path} (remove_checkpoint=True).",
            flush=True,
        )


def _split_batched_scene(scene_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    coords = scene_dict.get("coords")
    if not torch.is_tensor(coords) or coords.ndim != 3:
        return [scene_dict]

    batch_size = coords.shape[0]
    if batch_size == 1:
        return [
            {
                key: value.squeeze(0) if torch.is_tensor(value) else value
                for key, value in scene_dict.items()
            }
        ]

    scenes: List[Dict[str, Any]] = []
    for batch_idx in range(batch_size):
        entry: Dict[str, Any] = {}
        for key, value in scene_dict.items():
            if torch.is_tensor(value) and value.shape[0] == batch_size:
                entry[key] = value[batch_idx]
            else:
                entry[key] = value
        scenes.append(entry)
    return scenes


def _normalize_samples(samples: Any) -> List[GaussianScene]:
    if isinstance(samples, (list, tuple)):
        items = list(samples)
    else:
        items = [samples]

    scenes: List[GaussianScene] = []
    for item in items:
        if isinstance(item, GaussianScene):
            scenes.append(item)
            continue
        if isinstance(item, dict):
            for scene_dict in _split_batched_scene(item):
                scenes.append(GaussianScene.from_dict(scene_dict))
            continue
        raise TypeError(f"Unsupported sample type {type(item)}.")
    return scenes


@hydra.main(config_path="conf", config_name="gpt_eval", version_base=None)
def main(cfg: DictConfig) -> None:
    completion_cfg = (
        OmegaConf.to_container(cfg.get("completion"), resolve=True)
        if cfg.get("completion")
        else None
    )
    completion_enabled = bool(completion_cfg and completion_cfg.get("enabled", False))

    data_cfg: Dict[str, Any] = {}
    if cfg.get("data") is not None:
        data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
        if not isinstance(data_cfg, dict):
            raise TypeError("Expected `cfg.data` to convert to a dictionary.")
        if data_cfg.get("sort_latents") is not None:
            raise ValueError(
                "`data.sort_latents` is deprecated outside tokenization; use `model.chunk_order`."
            )
    elif completion_enabled:
        raise ValueError(
            "Completion requires a data config. Pass a tokenized data config via "
            "`data=...`."
        )

    dataset = cfg.get("dataset")
    if dataset is None:
        dataset = data_cfg.get("dataset_name")
    if completion_enabled and dataset is None:
        raise ValueError(
            "Completion requires dataset name via `dataset` or `data.dataset_name`."
        )

    requested_background = cfg.get("background_color")
    background_color = (
        requested_background
        if requested_background is not None
        else data_cfg.get("background_color", "white")
    )

    vqvae_checkpoint = cfg.get("vqvae_checkpoint")
    if not vqvae_checkpoint:
        model_cfg = (
            OmegaConf.to_container(cfg.get("model"), resolve=True)
            if cfg.get("model")
            else {}
        )
        if isinstance(model_cfg, dict):
            vqvae_config = model_cfg.get("vqvae")
            if isinstance(vqvae_config, dict):
                vqvae_checkpoint = vqvae_config.get("checkpoint_path")
    if not vqvae_checkpoint:
        vqvae_checkpoint = data_cfg.get("vqvae_path")
    if vqvae_checkpoint is None:
        raise ValueError(
            "A VQ-VAE checkpoint must be provided via `model.vqvae.checkpoint_path`, "
            "`data.vqvae_path`, or an explicit `vqvae_checkpoint` override."
        )

    checkpoint_path = Path(cfg.checkpoint)
    if not checkpoint_path.exists():
        if cfg.checkpoint_timeout == 0:
            raise FileNotFoundError(f"Checkpoint {checkpoint_path} does not exist.")
        print(
            f"Waiting up to {cfg.checkpoint_timeout}s for checkpoint {checkpoint_path}.",
            flush=True,
        )
        waited = 0
        while not checkpoint_path.exists():
            time.sleep(1)
            waited += 1
            if waited >= cfg.checkpoint_timeout:
                raise TimeoutError(
                    f"Checkpoint {checkpoint_path} was not found after {cfg.checkpoint_timeout}s."
                )
        print(f"Checkpoint found after {waited}s.", flush=True)

    print("Sleeping for 10s to ensure checkpoint write completion.", flush=True)
    time.sleep(10)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lightning.seed_everything(0)

    high_size = (512, 512)
    low_size = (128, 128)

    sample_and_evaluate(
        dataset=dataset,
        output_dir=output_dir,
        num_samples=cfg.num_samples,
        batch_size=cfg.batch_size,
        temperature=cfg.temperature,
        top_k=cfg.get("top_k"),
        top_p=cfg.get("top_p"),
        max_length=cfg.get("max_length"),
        views=cfg.views,
        high_size=high_size,
        low_size=low_size,
        background_color=background_color,
        skip_metrics=cfg.skip_metrics,
        gif_frames=getattr(cfg, "gif_frames", 120),
        gif_fps=getattr(cfg, "gif_fps", 24),
        vqvae_checkpoint=vqvae_checkpoint,
        store_samples=cfg.store_samples,
        data_cfg=data_cfg,
        checkpoint_path=checkpoint_path,
        remove_checkpoint=cfg.remove_checkpoint,
        completion_cfg=completion_cfg,
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
