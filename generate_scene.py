from __future__ import annotations

import os
import time
import zipfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import hydra
import imageio.v3 as iio
import torch
from omegaconf import DictConfig

from model.gaussian_gpt import GaussianGPT
from model.gaussian_vqvae import GaussianVQVAE
from utils.pos_tokens import coords_to_pos_tokens
from utils.render import GaussianScene, render
from utils.sampling import build_token_type_sampling_params_resolver

SparseColumnRows = List[Tuple[int, torch.Tensor]]
SparseRowsGrid = List[List[SparseColumnRows]]
TOPDOWN_FIT_MARGIN = 1.05
MAX_RETRIES = 5

# pylint: disable=E1120,W0212


def _resolve_shard_env() -> Tuple[int, int]:
    # Each array task is fully independent. We only need shard_id (== rank) and
    # num_shards (== world_size) so that _shard_range partitions num_scenes
    # consistently across tasks. No torchrun, no NCCL — tasks need not be
    # aligned in time.
    if "GAUSS_SHARD_ID" not in os.environ or "GAUSS_NUM_SHARDS" not in os.environ:
        raise RuntimeError(
            "generate_scene.py expects GAUSS_SHARD_ID and GAUSS_NUM_SHARDS to be set "
            "in the environment (shard_id == rank, num_shards == world_size). "
            "For a single non-sharded run use GAUSS_SHARD_ID=0 GAUSS_NUM_SHARDS=1."
        )
    shard_id = int(os.environ["GAUSS_SHARD_ID"])
    num_shards = int(os.environ["GAUSS_NUM_SHARDS"])
    if num_shards < 1:
        raise ValueError(f"GAUSS_NUM_SHARDS must be >= 1, got {num_shards}.")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"GAUSS_SHARD_ID={shard_id} out of range [0, {num_shards}).")
    return shard_id, num_shards


def _setup_single_gpu() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for parallel GPT large-scene sampling.")
    # SLURM with --gpus=1 exposes a single device, mapped to cuda:0 inside the job.
    torch.cuda.set_device(0)
    return torch.device("cuda:0")


def _wait_for_checkpoint(checkpoint_path: Path, timeout_s: int) -> None:
    if checkpoint_path.exists():
        return
    if timeout_s == 0:
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} does not exist.")
    print(f"Waiting up to {timeout_s}s for checkpoint {checkpoint_path}.", flush=True)
    waited = 0
    while not checkpoint_path.exists():
        time.sleep(1)
        waited += 1
        if waited >= timeout_s:
            raise TimeoutError(
                f"Checkpoint {checkpoint_path} was not found after {timeout_s}s."
            )
    print(f"Checkpoint found after {waited}s.", flush=True)


def _shard_range(total: int, rank: int, world_size: int) -> Tuple[int, int, int]:
    base = total // world_size
    extra = total % world_size
    local_n = base + int(rank < extra)
    start = rank * base + min(rank, extra)
    end = start + local_n
    return local_n, start, end


def _seed_for_step(
    base_seed: int,
    rank: int,
    scene_idx: int,
    col_idx: int,
    step_idx: int,
) -> int:
    mod = 2**31 - 1
    return int(
        (
            base_seed
            + rank * 1_000_003
            + scene_idx * 97_003
            + col_idx * 8_191
            + step_idx * 101
        )
        % mod
    )


def _temperature_for_retry(base_temperature: float, retry_idx: int) -> float:
    if retry_idx <= 5:
        return float(base_temperature)
    return float(base_temperature * (1.0 + 0.1 * float(retry_idx - 5)))


def _resolve_feature_sampling_params(
    cfg: DictConfig,
) -> Tuple[Optional[float], Optional[int], Optional[float]]:
    feature_temperature = cfg.get("feature_temperature")
    feature_top_k = cfg.get("feature_top_k")
    feature_top_p = cfg.get("feature_top_p")
    return (
        float(feature_temperature) if feature_temperature is not None else None,
        int(feature_top_k) if feature_top_k is not None else None,
        float(feature_top_p) if feature_top_p is not None else None,
    )


def _build_sparse_sampling_params_resolver(
    gpt: GaussianGPT,
    *,
    prompt_token_count: int,
    default_temperature: float,
    default_top_k: Optional[int],
    default_top_p: Optional[float],
    feature_temperature: Optional[float],
    feature_top_k: Optional[int],
    feature_top_p: Optional[float],
):
    return build_token_type_sampling_params_resolver(
        prompt_token_count=prompt_token_count,
        tokens_per_latent=int(gpt.tokens_per_latent),
        num_position_tokens=int(gpt.num_position_tokens),
        default_temperature=default_temperature,
        default_top_k=default_top_k,
        default_top_p=default_top_p,
        feature_temperature=feature_temperature,
        feature_top_k=feature_top_k,
        feature_top_p=feature_top_p,
    )


def _collect_context_cells(
    target_x: int,
    target_y: int,
    *,
    scene_cols_x: int,
    scene_cols_y: int,
    context_chunks_x: int,
    context_chunks_y: int,
) -> List[Tuple[int, int]]:
    start_x = max(0, target_x - context_chunks_x + 1)
    start_y = max(0, target_y - context_chunks_y + 1)
    end_x = min(scene_cols_x, start_x + context_chunks_x)
    end_y = min(scene_cols_y, start_y + context_chunks_y)
    cells: List[Tuple[int, int]] = []
    for x in range(start_x, end_x):
        for y in range(start_y, end_y):
            if x > target_x or (x == target_x and y >= target_y):
                continue
            cells.append((x, y))
    return cells


def _context_window_start(
    target_x: int,
    target_y: int,
    *,
    context_chunks_x: int,
    context_chunks_y: int,
) -> Tuple[int, int]:
    return (
        max(0, target_x - context_chunks_x + 1),
        max(0, target_y - context_chunks_y + 1),
    )


def _outpainting_window_for_target(
    target_x: int,
    target_y: int,
    *,
    scene_cols_x: int,
    scene_cols_y: int,
    context_chunks_x: int,
    context_chunks_y: int,
    x_offset: int,
    y_offset: int,
    y_range: int,
) -> Tuple[int, int, int, int]:
    target_local_x_desired = int((context_chunks_x - 1) - x_offset)
    max_start_x = max(0, scene_cols_x - context_chunks_x)
    raw_start_x = int(target_x - target_local_x_desired)
    start_x = max(0, min(raw_start_x, max_start_x))
    target_local_x = int(target_x - start_x)

    offset_first_desired = int(y_offset + y_range - 1)
    target_local_y_first_desired = int((context_chunks_y - 1) - offset_first_desired)
    max_start_y = max(0, scene_cols_y - context_chunks_y)
    raw_start_y = int(target_y - target_local_y_first_desired)
    start_y = max(0, min(raw_start_y, max_start_y))
    target_local_y = int(target_y - start_y)

    if (
        target_local_x < 0
        or target_local_x >= context_chunks_x
        or target_local_y < 0
        or target_local_y >= context_chunks_y
    ):
        start_x, start_y = _context_window_start(
            target_x,
            target_y,
            context_chunks_x=context_chunks_x,
            context_chunks_y=context_chunks_y,
        )
        target_local_x = int(target_x - start_x)
        target_local_y = int(target_y - start_y)
    return start_x, start_y, target_local_x, target_local_y


def _build_no_empty_position_logits_processor(
    *,
    prompt_len_with_bos: int,
    target_local_x: int,
    target_local_ys: List[int],
    z: int,
    pos_token_to_coord: torch.Tensor,
    position_vocab_size: int,
    feature_offset: int,
    feature_vocab_size: int,
    eos_token_id: int,
    tokens_per_latent: int,
    num_position_tokens: int,
) -> Callable[[torch.Tensor, list, int], None]:
    if not target_local_ys:
        raise ValueError("target_local_ys must not be empty.")
    if tokens_per_latent < 1:
        raise ValueError("tokens_per_latent must be >= 1.")
    if num_position_tokens < 1:
        raise ValueError("num_position_tokens must be >= 1.")

    y_to_index = {int(local_y): idx for idx, local_y in enumerate(target_local_ys)}
    column_pos_ids_cpu: List[torch.Tensor] = []
    for local_y in target_local_ys:
        mask = (
            (pos_token_to_coord[:, 0] == int(target_local_x))
            & (pos_token_to_coord[:, 1] == int(local_y))
            & (pos_token_to_coord[:, 2] >= 0)
            & (pos_token_to_coord[:, 2] < int(z))
        )
        ids = torch.nonzero(mask, as_tuple=False).flatten().to(dtype=torch.long)
        if ids.numel() == 0:
            raise ValueError(
                f"No position tokens for local column (x={target_local_x}, y={local_y})"
            )
        column_pos_ids_cpu.append(ids)

    def _logits_processor(logits: torch.Tensor, row_states: list, _: int) -> None:
        if logits.ndim != 2:
            raise ValueError(
                f"logits must have shape (B, V), got {tuple(logits.shape)}"
            )
        if logits.shape[0] != len(row_states):
            raise ValueError(
                f"Batch mismatch for logits processor: {logits.shape[0]} vs {len(row_states)}"
            )

        device = logits.device
        column_pos_ids = [ids.to(device=device) for ids in column_pos_ids_cpu]

        for row_idx, state in enumerate(row_states):
            seq_len = len(state.current_tokens)
            if seq_len <= 0:
                continue
            slot_idx = int((seq_len - 1) % tokens_per_latent)
            is_position_slot = slot_idx < num_position_tokens

            counts = [0 for _ in target_local_ys]
            # exited[i] becomes True once the model has generated a position
            # token for a y-column strictly after target_local_ys[i], which is
            # the natural signal that it has finished scanning column i.
            exited = [False for _ in target_local_ys]
            if seq_len > prompt_len_with_bos:
                for abs_pos in range(prompt_len_with_bos, seq_len):
                    token_slot = int((abs_pos - 1) % tokens_per_latent)
                    if token_slot >= num_position_tokens:
                        continue
                    token = int(state.current_tokens[abs_pos])
                    if token < 0 or token >= position_vocab_size:
                        continue
                    coord = pos_token_to_coord[token]
                    px, py, pz = [int(v) for v in coord.tolist()]
                    if px != target_local_x:
                        continue
                    # Mark every earlier started column as exited when the
                    # model moves to a later y-column (column boundary cross).
                    col_idx = y_to_index.get(py)
                    if col_idx is None:
                        # Position is for a y outside our target set — the
                        # model has exited all started columns.
                        for k in range(len(exited)):
                            if counts[k] > 0:
                                exited[k] = True
                    else:
                        for k in range(col_idx):
                            if counts[k] > 0:
                                exited[k] = True
                    # Count completed pos+feat pairs (requires valid feature
                    # token in the immediately following slot).
                    if pz < 0 or pz >= z:
                        continue
                    if col_idx is None:
                        continue
                    next_pos = abs_pos + 1
                    if next_pos >= seq_len:
                        continue
                    next_slot = int((next_pos - 1) % tokens_per_latent)
                    if next_slot < num_position_tokens:
                        continue
                    feature_token = int(state.current_tokens[next_pos])
                    if (
                        feature_token < feature_offset
                        or feature_token >= feature_offset + feature_vocab_size
                    ):
                        continue
                    counts[col_idx] += 1

            # A column is "done" when it has enough latents (>= 2, to cover
            # at least a floor and ceiling surface) AND has been exited (model
            # crossed to a later y-column).
            # Exception: the last target column has no subsequent y-column to
            # cross into (the position-token constraint forbids it), so for the
            # last column the latent count alone is sufficient — the model
            # signals completion there via EOS rather than a boundary crossing.
            first_empty = 0
            while first_empty < len(counts) and counts[first_empty] >= 2:
                is_last_col = first_empty == len(counts) - 1
                if not is_last_col and not exited[first_empty]:
                    break
                first_empty += 1

            # Constrain which position tokens are allowed (position slots only).
            if is_position_slot:
                if first_empty >= len(counts):
                    allowed_col_indices = [len(counts) - 1]
                elif first_empty < len(counts) - 1:
                    # Allow current and next column so the model can emit the
                    # "exit signal" (a token for column first_empty+1) that
                    # lets exited[first_empty] be set and first_empty advance.
                    allowed_col_indices = [first_empty, first_empty + 1]
                else:  # first_empty is the last column
                    allowed_col_indices = [first_empty]

                allowed_ids = torch.cat(
                    [column_pos_ids[idx] for idx in allowed_col_indices], dim=0
                ).unique()

                row_pos_logits = logits[row_idx, :position_vocab_size]
                kept_values = row_pos_logits[allowed_ids].clone()
                row_pos_logits.fill_(float("-inf"))
                row_pos_logits[allowed_ids] = kept_values

            # Block EOS at both position and feature slots while columns remain
            # unfilled.  Previously this only ran for position slots, so a model
            # that sampled EOS at a feature slot could silently discard the
            # preceding (orphaned) position token via the usable_len strip.
            if first_empty < len(counts) and 0 <= eos_token_id < int(logits.shape[1]):
                logits[row_idx, eos_token_id] = float("-inf")

    return _logits_processor


def _build_sparse_prompt_tokens(
    rows_grid: SparseRowsGrid,
    target_x: int,
    target_y: int,
    *,
    scene_cols_x: int,
    scene_cols_y: int,
    context_chunks_x: int,
    context_chunks_y: int,
    num_position_tokens: int,
    base_side_length: int,
) -> torch.Tensor:
    start_x, start_y = _context_window_start(
        target_x,
        target_y,
        context_chunks_x=context_chunks_x,
        context_chunks_y=context_chunks_y,
    )
    rows: List[torch.Tensor] = []
    for gx, gy in _collect_context_cells(
        target_x,
        target_y,
        scene_cols_x=scene_cols_x,
        scene_cols_y=scene_cols_y,
        context_chunks_x=context_chunks_x,
        context_chunks_y=context_chunks_y,
    ):
        for z_idx, feature_tokens in rows_grid[gx][gy]:
            local_coord = torch.tensor(
                [[gx - start_x, gy - start_y, int(z_idx)]],
                dtype=torch.long,
            )
            pos_tokens = coords_to_pos_tokens(
                local_coord,
                num_position_tokens,
                base_side_length,
            ).squeeze(0)
            feat = feature_tokens.to(dtype=torch.long)
            if feat.ndim == 0:
                feat = feat.unsqueeze(0)
            rows.append(torch.cat((pos_tokens, feat), dim=0))

    if not rows:
        return torch.empty(0, dtype=torch.long)
    return torch.cat(rows, dim=0)


def _collect_context_cells_from_window(
    target_x: int,
    target_y: int,
    *,
    start_x: int,
    start_y: int,
    scene_cols_x: int,
    scene_cols_y: int,
    context_chunks_x: int,
    context_chunks_y: int,
) -> List[Tuple[int, int]]:
    end_x = min(scene_cols_x, start_x + context_chunks_x)
    end_y = min(scene_cols_y, start_y + context_chunks_y)
    cells: List[Tuple[int, int]] = []
    for x in range(start_x, end_x):
        for y in range(start_y, end_y):
            if x > target_x or (x == target_x and y >= target_y):
                continue
            cells.append((x, y))
    return cells


def _build_sparse_prompt_tokens_from_window(
    rows_grid: SparseRowsGrid,
    *,
    target_x: int,
    target_y: int,
    start_x: int,
    start_y: int,
    scene_cols_x: int,
    scene_cols_y: int,
    context_chunks_x: int,
    context_chunks_y: int,
    num_position_tokens: int,
    base_side_length: int,
) -> torch.Tensor:
    rows: List[torch.Tensor] = []
    for gx, gy in _collect_context_cells_from_window(
        target_x,
        target_y,
        start_x=start_x,
        start_y=start_y,
        scene_cols_x=scene_cols_x,
        scene_cols_y=scene_cols_y,
        context_chunks_x=context_chunks_x,
        context_chunks_y=context_chunks_y,
    ):
        for z_idx, feature_tokens in rows_grid[gx][gy]:
            local_coord = torch.tensor(
                [[gx - start_x, gy - start_y, int(z_idx)]],
                dtype=torch.long,
            )
            pos_tokens = coords_to_pos_tokens(
                local_coord,
                num_position_tokens,
                base_side_length,
            ).squeeze(0)
            feat = feature_tokens.to(dtype=torch.long)
            if feat.ndim == 0:
                feat = feat.unsqueeze(0)
            rows.append(torch.cat((pos_tokens, feat), dim=0))

    if not rows:
        return torch.empty(0, dtype=torch.long)
    return torch.cat(rows, dim=0)


@torch.no_grad()
def _accept_sparse_rows_for_target(
    new_tokens: torch.Tensor,
    *,
    target_local_x: int,
    target_local_y: int,
    z: int,
    position_vocab_size: int,
    feature_offset: int,
    feature_vocab_size: int,
    pos_token_to_coord: torch.Tensor,
    debug_column_context: bool,
    rank: int,
    scene_idx: int,
    col_idx: int,
) -> Tuple[torch.Tensor, SparseColumnRows]:
    accepted_token_rows: List[torch.Tensor] = []
    accepted_rows: SparseColumnRows = []

    usable_len = (new_tokens.numel() // 2) * 2
    if usable_len <= 0:
        if debug_column_context:
            stop_reason = (
                "eos_before_any" if int(new_tokens.numel()) == 0 else "odd_token_count"
            )
            print(
                f"[rank {rank}] scene {scene_idx} col={col_idx} sample_debug "
                f"sampled_new_tokens={int(new_tokens.numel())} sampled_rows=0 accepted_rows=0 "
                f"rejected_rows=0 stop_reason={stop_reason}.",
                flush=True,
            )
        return torch.empty(0, dtype=torch.long), []
    token_rows = new_tokens[:usable_len].view(-1, 2).detach().cpu().to(dtype=torch.long)

    stop_reason = "all_rows_accepted"
    for row_tokens in token_rows:
        pos_token = int(row_tokens[0].item())
        feature_token = int(row_tokens[1].item())

        if pos_token < 0 or pos_token >= position_vocab_size:
            stop_reason = "invalid_pos_id"
            break
        if (
            feature_token < feature_offset
            or feature_token >= feature_offset + feature_vocab_size
        ):
            stop_reason = "invalid_feature_id"
            break

        local_coord = pos_token_to_coord[pos_token]
        px, py, pz = [int(v) for v in local_coord.tolist()]
        if px != target_local_x or py != target_local_y:
            stop_reason = "wrong_xy"
            break
        if pz < 0 or pz >= z:
            stop_reason = "z_oob"
            break

        accepted_token_rows.append(row_tokens)
        accepted_rows.append((pz, torch.tensor([feature_token], dtype=torch.long)))

    if debug_column_context:
        sampled_rows = int(token_rows.shape[0])
        accepted_count = len(accepted_rows)
        print(
            f"[rank {rank}] scene {scene_idx} col={col_idx} sample_debug "
            f"sampled_new_tokens={int(new_tokens.numel())} sampled_rows={sampled_rows} "
            f"accepted_rows={accepted_count} rejected_rows={sampled_rows - accepted_count} "
            f"stop_reason={stop_reason}.",
            flush=True,
        )

    if not accepted_token_rows:
        return torch.empty(0, dtype=torch.long), []
    return torch.cat(accepted_token_rows, dim=0), accepted_rows


@torch.no_grad()
def _sample_sparse_column_tokenwise(
    gpt: GaussianGPT,
    base_prompt: torch.Tensor,
    *,
    target_local_x: int,
    target_local_y: int,
    z: int,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
    feature_temperature: Optional[float],
    feature_top_k: Optional[int],
    feature_top_p: Optional[float],
    base_seed: int,
    rank: int,
    scene_idx: int,
    col_idx: int,
    step_idx: int = 0,
    no_empty_columns: bool = False,
    pos_token_to_coord: torch.Tensor,
    debug_column_context: bool,
) -> Tuple[torch.Tensor, SparseColumnRows]:
    position_vocab_size = int(gpt.position_vocab_size)
    feature_offset = int(gpt.feature_token_offset)
    feature_vocab_size = int(gpt.feature_vocab_size)

    prompt_tokens = base_prompt.detach().cpu().to(dtype=torch.long)
    seed = _seed_for_step(base_seed, rank, scene_idx, col_idx, step_idx)
    sampling_params_resolver = _build_sparse_sampling_params_resolver(
        gpt,
        prompt_token_count=int(prompt_tokens.numel()),
        default_temperature=temperature,
        default_top_k=top_k,
        default_top_p=top_p,
        feature_temperature=feature_temperature,
        feature_top_k=feature_top_k,
        feature_top_p=feature_top_p,
    )
    logits_processor = None
    if no_empty_columns:
        logits_processor = _build_no_empty_position_logits_processor(
            prompt_len_with_bos=1 + int(prompt_tokens.numel()),
            target_local_x=target_local_x,
            target_local_ys=[int(target_local_y)],
            z=z,
            pos_token_to_coord=pos_token_to_coord,
            position_vocab_size=position_vocab_size,
            feature_offset=feature_offset,
            feature_vocab_size=feature_vocab_size,
            eos_token_id=int(gpt.gpt.eos_token_id),
            tokens_per_latent=int(gpt.tokens_per_latent),
            num_position_tokens=int(gpt.num_position_tokens),
        )
    sampled = gpt.gpt.sample_sequence_with_prompt(
        prompt_tokens,
        max_new_tokens=int(2 * z),
        num_samples=1,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        stop_on_eos=True,
        seed=seed,
        logits_processor=logits_processor,
        sampling_params_resolver=sampling_params_resolver,
    )[0]
    prompt_len = int(prompt_tokens.numel())
    new_tokens = sampled[prompt_len:]
    return _accept_sparse_rows_for_target(
        new_tokens,
        target_local_x=target_local_x,
        target_local_y=target_local_y,
        z=z,
        position_vocab_size=position_vocab_size,
        feature_offset=feature_offset,
        feature_vocab_size=feature_vocab_size,
        pos_token_to_coord=pos_token_to_coord,
        debug_column_context=debug_column_context,
        rank=rank,
        scene_idx=scene_idx,
        col_idx=col_idx,
    )


@torch.no_grad()
def _sample_sparse_column_group_tokenwise(
    gpt: GaussianGPT,
    base_prompt: torch.Tensor,
    *,
    target_local_x: int,
    target_local_ys: List[int],
    z: int,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
    feature_temperature: Optional[float],
    feature_top_k: Optional[int],
    feature_top_p: Optional[float],
    base_seed: int,
    rank: int,
    scene_idx: int,
    col_idx: int,
    step_idx: int = 0,
    no_empty_columns: bool = False,
    pos_token_to_coord: torch.Tensor,
    debug_column_context: bool,
) -> Tuple[List[Tuple[torch.Tensor, SparseColumnRows]], int]:
    if not target_local_ys:
        return [], 0

    position_vocab_size = int(gpt.position_vocab_size)
    feature_offset = int(gpt.feature_token_offset)
    feature_vocab_size = int(gpt.feature_vocab_size)
    prompt_tokens = base_prompt.detach().cpu().to(dtype=torch.long)
    seed = _seed_for_step(base_seed, rank, scene_idx, col_idx, step_idx)
    sampling_params_resolver = _build_sparse_sampling_params_resolver(
        gpt,
        prompt_token_count=int(prompt_tokens.numel()),
        default_temperature=temperature,
        default_top_k=top_k,
        default_top_p=top_p,
        feature_temperature=feature_temperature,
        feature_top_k=feature_top_k,
        feature_top_p=feature_top_p,
    )
    logits_processor = None
    if no_empty_columns:
        logits_processor = _build_no_empty_position_logits_processor(
            prompt_len_with_bos=1 + int(prompt_tokens.numel()),
            target_local_x=target_local_x,
            target_local_ys=[int(v) for v in target_local_ys],
            z=z,
            pos_token_to_coord=pos_token_to_coord,
            position_vocab_size=position_vocab_size,
            feature_offset=feature_offset,
            feature_vocab_size=feature_vocab_size,
            eos_token_id=int(gpt.gpt.eos_token_id),
            tokens_per_latent=int(gpt.tokens_per_latent),
            num_position_tokens=int(gpt.num_position_tokens),
        )
    sampled = gpt.gpt.sample_sequence_with_prompt(
        prompt_tokens,
        max_new_tokens=int(2 * z * len(target_local_ys)),
        num_samples=1,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        stop_on_eos=True,
        seed=seed,
        logits_processor=logits_processor,
        sampling_params_resolver=sampling_params_resolver,
    )[0]
    prompt_len = int(prompt_tokens.numel())
    new_tokens = sampled[prompt_len:]
    usable_len = (new_tokens.numel() // 2) * 2
    token_rows = (
        new_tokens[:usable_len].view(-1, 2).detach().cpu().to(dtype=torch.long)
        if usable_len > 0
        else torch.empty((0, 2), dtype=torch.long)
    )

    outputs: List[Tuple[torch.Tensor, SparseColumnRows]] = []
    cursor = 0
    hard_stop = False

    for group_idx, target_local_y in enumerate(target_local_ys):
        if group_idx > 0 and cursor >= int(token_rows.shape[0]):
            break

        accepted_token_rows: List[torch.Tensor] = []
        accepted_rows: SparseColumnRows = []
        stop_reason = "stream_exhausted"

        while cursor < int(token_rows.shape[0]):
            row_tokens = token_rows[cursor]
            pos_token = int(row_tokens[0].item())
            feature_token = int(row_tokens[1].item())

            if pos_token < 0 or pos_token >= position_vocab_size:
                hard_stop = True
                stop_reason = "invalid_pos_id"
                break
            if (
                feature_token < feature_offset
                or feature_token >= feature_offset + feature_vocab_size
            ):
                hard_stop = True
                stop_reason = "invalid_feature_id"
                break

            local_coord = pos_token_to_coord[pos_token]
            px, py, pz = [int(v) for v in local_coord.tolist()]
            if pz < 0 or pz >= z:
                hard_stop = True
                stop_reason = "z_oob"
                break
            if px == target_local_x and py == target_local_y:
                accepted_token_rows.append(row_tokens)
                accepted_rows.append(
                    (pz, torch.tensor([feature_token], dtype=torch.long))
                )
                cursor += 1
                stop_reason = "all_rows_accepted"
                continue

            stop_reason = "next_column_boundary"
            break

        if accepted_token_rows:
            col_tokens = torch.cat(accepted_token_rows, dim=0)
        else:
            col_tokens = torch.empty(0, dtype=torch.long)
        outputs.append((col_tokens, accepted_rows))

        if debug_column_context:
            print(
                f"[rank {rank}] scene {scene_idx} col={col_idx} group_debug "
                f"group_idx={group_idx} target_local_y={target_local_y} "
                f"accepted_rows={len(accepted_rows)} stop_reason={stop_reason} "
                f"cursor_rows={cursor}/{int(token_rows.shape[0])}.",
                flush=True,
            )

        if hard_stop:
            break

    if not outputs:
        outputs.append((torch.empty(0, dtype=torch.long), []))

    return outputs, len(outputs)


@torch.no_grad()
def _bootstrap_initial_context_block_like_parallel_sampling(
    gpt: GaussianGPT,
    *,
    scene_cols_x: int,
    scene_cols_y: int,
    context_chunks_x: int,
    context_chunks_y: int,
    z: int,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
    feature_temperature: Optional[float],
    feature_top_k: Optional[int],
    feature_top_p: Optional[float],
    base_seed: int,
    rank: int,
    scene_idx: int,
    pos_token_to_coord: torch.Tensor,
    debug_column_context: bool,
) -> Dict[Tuple[int, int], Tuple[torch.Tensor, SparseColumnRows]]:
    # NOTE: no_empty_columns is intentionally NOT applied here.  Bootstrap
    # samples a free-running token stream that is parsed post-hoc into columns;
    # the constraint requires a known target-column set at logit-processor time,
    # which is incompatible with that streaming approach.  Empty columns
    # produced here are handled downstream by resample_empty_columns.
    bootstrap_cols_x = min(int(scene_cols_x), int(context_chunks_x))
    bootstrap_cols_y = min(int(scene_cols_y), int(context_chunks_y))
    if bootstrap_cols_x <= 0 or bootstrap_cols_y <= 0:
        return {}

    total_bootstrap_columns = int(bootstrap_cols_x * bootstrap_cols_y)
    max_new_tokens = int(2 * z * total_bootstrap_columns)
    position_vocab_size = int(gpt.position_vocab_size)
    feature_offset = int(gpt.feature_token_offset)
    feature_vocab_size = int(gpt.feature_vocab_size)
    seed = _seed_for_step(base_seed, rank, scene_idx, 0, 0)
    sampling_params_resolver = _build_sparse_sampling_params_resolver(
        gpt,
        prompt_token_count=0,
        default_temperature=temperature,
        default_top_k=top_k,
        default_top_p=top_p,
        feature_temperature=feature_temperature,
        feature_top_k=feature_top_k,
        feature_top_p=feature_top_p,
    )
    sampled = gpt.gpt.sample_sequence(
        max_new_tokens,
        batch_size=1,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        stop_on_eos=True,
        seed=seed,
        sampling_params_resolver=sampling_params_resolver,
    )[0]
    stream = sampled.detach().cpu().to(dtype=torch.long)
    cursor = 0
    bootstrapped: Dict[Tuple[int, int], Tuple[torch.Tensor, SparseColumnRows]] = {}
    hard_stop = False
    hard_stop_reason = "none"

    for bx in range(bootstrap_cols_x):
        for by in range(bootstrap_cols_y):
            accepted_rows: SparseColumnRows = []
            accepted_token_rows: List[torch.Tensor] = []
            stop_reason = "stream_exhausted"

            while cursor + 1 < int(stream.numel()):
                pos_token = int(stream[cursor].item())
                feature_token = int(stream[cursor + 1].item())

                if pos_token < 0 or pos_token >= position_vocab_size:
                    stop_reason = "invalid_pos_id"
                    hard_stop = True
                    hard_stop_reason = stop_reason
                    break
                if (
                    feature_token < feature_offset
                    or feature_token >= feature_offset + feature_vocab_size
                ):
                    stop_reason = "invalid_feature_id"
                    hard_stop = True
                    hard_stop_reason = stop_reason
                    break

                local_coord = pos_token_to_coord[pos_token]
                px, py, pz = [int(v) for v in local_coord.tolist()]
                if pz < 0 or pz >= z:
                    stop_reason = "z_oob"
                    hard_stop = True
                    hard_stop_reason = stop_reason
                    break
                if px == bx and py == by:
                    accepted_token_rows.append(
                        torch.tensor([pos_token, feature_token], dtype=torch.long)
                    )
                    accepted_rows.append(
                        (pz, torch.tensor([feature_token], dtype=torch.long))
                    )
                    cursor += 2
                    stop_reason = "all_rows_accepted"
                    continue

                stop_reason = "next_column_boundary"
                break

            if accepted_token_rows:
                col_tokens = torch.cat(accepted_token_rows, dim=0)
            else:
                col_tokens = torch.empty(0, dtype=torch.long)
            bootstrapped[(bx, by)] = (col_tokens, accepted_rows)

            if debug_column_context:
                print(
                    f"[rank {rank}] scene {scene_idx} bootstrap col=({bx},{by}) "
                    f"accepted_rows={len(accepted_rows)} stop_reason={stop_reason} "
                    f"cursor={cursor}/{int(stream.numel())}.",
                    flush=True,
                )

            if hard_stop:
                break
        if hard_stop:
            break

    if debug_column_context:
        print(
            f"[rank {rank}] scene {scene_idx} bootstrap summary: "
            f"cols_x={bootstrap_cols_x}, cols_y={bootstrap_cols_y}, "
            f"stream_tokens={int(stream.numel())}, cursor={cursor}, hard_stop={hard_stop}, "
            f"hard_stop_reason={hard_stop_reason}.",
            flush=True,
        )

    return bootstrapped


@torch.no_grad()
def _bootstrap_shifted_x_slices_with_y0_start(
    gpt: GaussianGPT,
    *,
    scene_cols_x: int,
    scene_cols_y: int,
    context_chunks_x: int,
    context_chunks_y: int,
    x_offset: int,
    z: int,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
    feature_temperature: Optional[float],
    feature_top_k: Optional[int],
    feature_top_p: Optional[float],
    base_seed: int,
    rank: int,
    scene_idx: int,
    pos_token_to_coord: torch.Tensor,
    initial_bootstrap_columns: Dict[
        Tuple[int, int], Tuple[torch.Tensor, SparseColumnRows]
    ],
    debug_column_context: bool,
) -> Dict[Tuple[int, int], Tuple[torch.Tensor, SparseColumnRows]]:
    # NOTE: no_empty_columns is intentionally NOT applied here — same reason as
    # _bootstrap_initial_context_block_like_parallel_sampling: the stream is
    # parsed post-hoc so per-column logit constraints cannot be applied.
    bootstrap_cols_x = min(int(scene_cols_x), int(context_chunks_x))
    bootstrap_cols_y = min(int(scene_cols_y), int(context_chunks_y))
    if bootstrap_cols_x <= 0 or bootstrap_cols_y <= 0:
        return {}
    if scene_cols_x <= bootstrap_cols_x:
        return {}

    working_rows: SparseRowsGrid = [
        [[] for _ in range(bootstrap_cols_y)] for _ in range(scene_cols_x)
    ]
    for bx in range(bootstrap_cols_x):
        for by in range(bootstrap_cols_y):
            if (bx, by) in initial_bootstrap_columns:
                working_rows[bx][by] = initial_bootstrap_columns[(bx, by)][1]

    position_vocab_size = int(gpt.position_vocab_size)
    feature_offset = int(gpt.feature_token_offset)
    feature_vocab_size = int(gpt.feature_vocab_size)
    num_position_tokens = int(gpt.num_position_tokens)
    base_side_length = int(gpt.position_side_length)
    bootstrapped: Dict[Tuple[int, int], Tuple[torch.Tensor, SparseColumnRows]] = {}
    hard_stop = False
    hard_stop_reason = "none"
    target_local_x_desired = int((context_chunks_x - 1) - x_offset)
    max_start_x = max(0, scene_cols_x - context_chunks_x)

    for target_x in range(bootstrap_cols_x, scene_cols_x):
        raw_start_x = int(target_x - target_local_x_desired)
        start_x = max(0, min(raw_start_x, max_start_x))
        target_local_x = int(target_x - start_x)
        if start_x <= 0:
            continue

        prompt_rows: List[torch.Tensor] = []
        for gx in range(start_x, target_x):
            for gy in range(bootstrap_cols_y):
                for z_idx, feature_tokens in working_rows[gx][gy]:
                    local_coord = torch.tensor(
                        [[gx - start_x, gy, int(z_idx)]],
                        dtype=torch.long,
                    )
                    pos_tokens = coords_to_pos_tokens(
                        local_coord,
                        num_position_tokens,
                        base_side_length,
                    ).squeeze(0)
                    feat = feature_tokens.to(dtype=torch.long)
                    if feat.ndim == 0:
                        feat = feat.unsqueeze(0)
                    prompt_rows.append(torch.cat((pos_tokens, feat), dim=0))
        if prompt_rows:
            prompt_tokens = torch.cat(prompt_rows, dim=0)
        else:
            prompt_tokens = torch.empty(0, dtype=torch.long)

        seed = _seed_for_step(base_seed, rank, scene_idx, target_x, 1)
        sampling_params_resolver = _build_sparse_sampling_params_resolver(
            gpt,
            prompt_token_count=int(prompt_tokens.numel()),
            default_temperature=temperature,
            default_top_k=top_k,
            default_top_p=top_p,
            feature_temperature=feature_temperature,
            feature_top_k=feature_top_k,
            feature_top_p=feature_top_p,
        )
        sampled = gpt.gpt.sample_sequence_with_prompt(
            prompt_tokens,
            max_new_tokens=int(2 * z * bootstrap_cols_y),
            num_samples=1,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stop_on_eos=True,
            seed=seed,
            sampling_params_resolver=sampling_params_resolver,
        )[0]
        new_tokens = (
            sampled[int(prompt_tokens.numel()) :].detach().cpu().to(dtype=torch.long)
        )
        cursor = 0

        for by in range(bootstrap_cols_y):
            accepted_rows: SparseColumnRows = []
            accepted_token_rows: List[torch.Tensor] = []
            stop_reason = "stream_exhausted"

            while cursor + 1 < int(new_tokens.numel()):
                pos_token = int(new_tokens[cursor].item())
                feature_token = int(new_tokens[cursor + 1].item())

                if pos_token < 0 or pos_token >= position_vocab_size:
                    stop_reason = "invalid_pos_id"
                    hard_stop = True
                    hard_stop_reason = stop_reason
                    break
                if (
                    feature_token < feature_offset
                    or feature_token >= feature_offset + feature_vocab_size
                ):
                    stop_reason = "invalid_feature_id"
                    hard_stop = True
                    hard_stop_reason = stop_reason
                    break

                local_coord = pos_token_to_coord[pos_token]
                px, py, pz = [int(v) for v in local_coord.tolist()]
                if pz < 0 or pz >= z:
                    stop_reason = "z_oob"
                    hard_stop = True
                    hard_stop_reason = stop_reason
                    break
                if px == target_local_x and py == by:
                    accepted_token_rows.append(
                        torch.tensor([pos_token, feature_token], dtype=torch.long)
                    )
                    accepted_rows.append(
                        (pz, torch.tensor([feature_token], dtype=torch.long))
                    )
                    cursor += 2
                    stop_reason = "all_rows_accepted"
                    continue

                stop_reason = "next_column_boundary"
                break

            if accepted_token_rows:
                col_tokens = torch.cat(accepted_token_rows, dim=0)
            else:
                col_tokens = torch.empty(0, dtype=torch.long)
            bootstrapped[(target_x, by)] = (col_tokens, accepted_rows)
            working_rows[target_x][by] = accepted_rows

            if debug_column_context:
                print(
                    f"[rank {rank}] scene {scene_idx} bootstrap2 col=({target_x},{by}) "
                    f"accepted_rows={len(accepted_rows)} stop_reason={stop_reason} "
                    f"cursor={cursor}/{int(new_tokens.numel())} start_x={start_x}.",
                    flush=True,
                )

            if hard_stop:
                break
        if hard_stop:
            break

    if debug_column_context:
        print(
            f"[rank {rank}] scene {scene_idx} bootstrap2 summary: "
            f"x_from={bootstrap_cols_x}, x_to={scene_cols_x - 1}, cols_y={bootstrap_cols_y}, "
            f"prepared_cols={len(bootstrapped)}, hard_stop={hard_stop}, "
            f"hard_stop_reason={hard_stop_reason}.",
            flush=True,
        )

    return bootstrapped


@torch.no_grad()
def _generate_sparse_scene(
    gpt: GaussianGPT,
    *,
    scene_cols_x: int,
    scene_cols_y: int,
    context_chunks_x: int,
    context_chunks_y: int,
    z: int,
    outpainting_temperature: float,
    outpainting_top_k: Optional[int],
    outpainting_top_p: Optional[float],
    bootstrap_temperature: float,
    bootstrap_top_k: Optional[int],
    bootstrap_top_p: Optional[float],
    bootstrap2_temperature: float,
    bootstrap2_top_k: Optional[int],
    bootstrap2_top_p: Optional[float],
    feature_temperature: Optional[float],
    feature_top_k: Optional[int],
    feature_top_p: Optional[float],
    x_offset: int,
    y_offset: int,
    y_range: int,
    no_empty_columns: bool,
    resample_empty_columns: bool,
    enable_bootstrap: bool,
    early_stop_min_occupancy: Optional[float],
    max_empty_column_retries: int,
    base_seed: int,
    rank: int,
    scene_idx: int,
    pos_token_to_coord: torch.Tensor,
    debug_column_context: bool,
) -> Tuple[torch.Tensor, torch.Tensor, SparseRowsGrid]:
    scene_start = time.perf_counter()
    token_pieces: List[torch.Tensor] = []
    column_token_lengths = torch.zeros(
        (scene_cols_x, scene_cols_y),
        dtype=torch.long,
    )
    rows_grid: SparseRowsGrid = [
        [[] for _ in range(scene_cols_y)] for _ in range(scene_cols_x)
    ]

    total_columns = scene_cols_x * scene_cols_y
    next_progress_pct = 10
    accepted_rows_total = 0
    bootstrap_columns: Dict[Tuple[int, int], Tuple[torch.Tensor, SparseColumnRows]] = {}
    bootstrap_cols_x = min(int(scene_cols_x), int(context_chunks_x))
    bootstrap_cols_y = min(int(scene_cols_y), int(context_chunks_y))
    if enable_bootstrap:
        total_bootstrap_cols = int(bootstrap_cols_x * bootstrap_cols_y)
        # Bootstrap phases generate the first context block unconstrainedly:
        # no_empty_columns is intentionally NOT applied here because the
        # constraint requires a fixed target-column set, while bootstrap samples
        # a free-running stream that is parsed post-hoc into columns.
        # Empty bootstrap columns are handled downstream by resample_empty_columns.
        print(
            f"[rank {rank}] scene {scene_idx} bootstrap start: "
            f"mode=initial_context_block cols_x={bootstrap_cols_x} cols_y={bootstrap_cols_y} "
            f"total_cols={total_bootstrap_cols}"
            + (
                " no_empty_columns=True does not apply to bootstrap"
                if no_empty_columns
                else ""
            )
            + ".",
            flush=True,
        )
        bootstrap_columns = _bootstrap_initial_context_block_like_parallel_sampling(
            gpt,
            scene_cols_x=scene_cols_x,
            scene_cols_y=scene_cols_y,
            context_chunks_x=context_chunks_x,
            context_chunks_y=context_chunks_y,
            z=z,
            temperature=bootstrap_temperature,
            top_k=bootstrap_top_k,
            top_p=bootstrap_top_p,
            feature_temperature=feature_temperature,
            feature_top_k=feature_top_k,
            feature_top_p=feature_top_p,
            base_seed=base_seed,
            rank=rank,
            scene_idx=scene_idx,
            pos_token_to_coord=pos_token_to_coord,
            debug_column_context=debug_column_context,
        )
        bootstrap_rows = sum(len(rows) for _, rows in bootstrap_columns.values())
        bootstrap_nonempty_cols = sum(
            1 for _, rows in bootstrap_columns.values() if len(rows) > 0
        )
        bootstrap_tokens = sum(
            int(tokens.numel()) for tokens, _ in bootstrap_columns.values()
        )
        print(
            f"[rank {rank}] scene {scene_idx} bootstrap done: "
            f"prepared_cols={len(bootstrap_columns)} nonempty_cols={bootstrap_nonempty_cols} "
            f"accepted_rows={bootstrap_rows} accepted_tokens={bootstrap_tokens}.",
            flush=True,
        )

        print(
            f"[rank {rank}] scene {scene_idx} bootstrap2 start: "
            f"mode=shifted_x_slices_y0_start x_from={bootstrap_cols_x} cols_y={bootstrap_cols_y}"
            + (
                " no_empty_columns=True does not apply to bootstrap"
                if no_empty_columns
                else ""
            )
            + ".",
            flush=True,
        )
        bootstrap2_columns = _bootstrap_shifted_x_slices_with_y0_start(
            gpt,
            scene_cols_x=scene_cols_x,
            scene_cols_y=scene_cols_y,
            context_chunks_x=context_chunks_x,
            context_chunks_y=context_chunks_y,
            x_offset=x_offset,
            z=z,
            temperature=bootstrap2_temperature,
            top_k=bootstrap2_top_k,
            top_p=bootstrap2_top_p,
            feature_temperature=feature_temperature,
            feature_top_k=feature_top_k,
            feature_top_p=feature_top_p,
            base_seed=base_seed,
            rank=rank,
            scene_idx=scene_idx,
            pos_token_to_coord=pos_token_to_coord,
            initial_bootstrap_columns=bootstrap_columns,
            debug_column_context=debug_column_context,
        )
        bootstrap2_rows = sum(len(rows) for _, rows in bootstrap2_columns.values())
        bootstrap2_nonempty_cols = sum(
            1 for _, rows in bootstrap2_columns.values() if len(rows) > 0
        )
        bootstrap2_tokens = sum(
            int(tokens.numel()) for tokens, _ in bootstrap2_columns.values()
        )
        bootstrap_columns.update(bootstrap2_columns)
        print(
            f"[rank {rank}] scene {scene_idx} bootstrap2 done: "
            f"prepared_cols={len(bootstrap2_columns)} nonempty_cols={bootstrap2_nonempty_cols} "
            f"accepted_rows={bootstrap2_rows} accepted_tokens={bootstrap2_tokens}.",
            flush=True,
        )
    else:
        print(
            f"[rank {rank}] scene {scene_idx} bootstrap disabled by config.",
            flush=True,
        )

    col_idx = 0
    done_columns = 0
    early_stopped = False
    while col_idx < total_columns:
        x = col_idx // scene_cols_y
        y = col_idx % scene_cols_y
        sampled_columns: List[Tuple[int, torch.Tensor, SparseColumnRows]] = []
        advance_cols = 1
        group_window: Optional[Tuple[int, int, int]] = None

        if (x, y) in bootstrap_columns:
            if debug_column_context:
                print(
                    f"[rank {rank}] scene {scene_idx} col={col_idx} bootstrap_mode=precomputed",
                    flush=True,
                )
            column_tokens, column_rows = bootstrap_columns[(x, y)]
            sampled_columns.append((y, column_tokens, column_rows))
        else:
            (
                start_x,
                start_y,
                target_local_x,
                _,
            ) = _outpainting_window_for_target(
                x,
                y,
                scene_cols_x=scene_cols_x,
                scene_cols_y=scene_cols_y,
                context_chunks_x=context_chunks_x,
                context_chunks_y=context_chunks_y,
                x_offset=x_offset,
                y_offset=y_offset,
                y_range=y_range,
            )

            y_targets: List[int] = []
            target_local_ys: List[int] = []
            for y_step in range(y_range):
                target_y = int(y + y_step)
                if target_y >= scene_cols_y:
                    break
                if (x, target_y) in bootstrap_columns:
                    break
                target_local_y = int(target_y - start_y)
                if target_local_y < 0 or target_local_y >= context_chunks_y:
                    break
                offset = int((context_chunks_y - 1) - target_local_y)
                if offset < y_offset or offset >= y_offset + y_range:
                    break
                y_targets.append(target_y)
                target_local_ys.append(target_local_y)

            if not y_targets:
                # Edge fallback: keep grouped sampling under the clamped shifted
                # window, but relax desired offset constraints when infeasible.
                relaxed_y_targets: List[int] = []
                relaxed_target_local_ys: List[int] = []
                for y_step in range(y_range):
                    target_y = int(y + y_step)
                    if target_y >= scene_cols_y:
                        break
                    if (x, target_y) in bootstrap_columns:
                        break
                    target_local_y = int(target_y - start_y)
                    if target_local_y < 0 or target_local_y >= context_chunks_y:
                        break
                    relaxed_y_targets.append(target_y)
                    relaxed_target_local_ys.append(target_local_y)

                if relaxed_y_targets:
                    y_targets = relaxed_y_targets
                    target_local_ys = relaxed_target_local_ys
                else:
                    # Safety fallback (should rarely trigger): default edge-history mode.
                    start_x, start_y = _context_window_start(
                        x,
                        y,
                        context_chunks_x=context_chunks_x,
                        context_chunks_y=context_chunks_y,
                    )
                    target_local_x = int(x - start_x)
                    target_local_ys = [int(y - start_y)]
                    y_targets = [y]

            prompt = _build_sparse_prompt_tokens_from_window(
                rows_grid,
                target_x=x,
                target_y=y_targets[0],
                start_x=start_x,
                start_y=start_y,
                scene_cols_x=scene_cols_x,
                scene_cols_y=scene_cols_y,
                context_chunks_x=context_chunks_x,
                context_chunks_y=context_chunks_y,
                num_position_tokens=int(gpt.num_position_tokens),
                base_side_length=int(gpt.position_side_length),
            )
            group_window = (int(start_x), int(start_y), int(target_local_x))

            if debug_column_context:
                context_cells = _collect_context_cells_from_window(
                    x,
                    y_targets[0],
                    start_x=start_x,
                    start_y=start_y,
                    scene_cols_x=scene_cols_x,
                    scene_cols_y=scene_cols_y,
                    context_chunks_x=context_chunks_x,
                    context_chunks_y=context_chunks_y,
                )
                context_rows = sum(
                    int(len(rows_grid[gx][gy])) for gx, gy in context_cells
                )
                context_unsorted_cells = 0
                for gx, gy in context_cells:
                    z_seq = [int(z_idx) for z_idx, _ in rows_grid[gx][gy]]
                    if any(
                        z_seq[idx] > z_seq[idx + 1] for idx in range(len(z_seq) - 1)
                    ):
                        context_unsorted_cells += 1
                print(
                    f"[rank {rank}] scene {scene_idx} col={col_idx} "
                    f"group_len_target={len(y_targets)} "
                    f"global_xy=({x},{y_targets[0]}) model_xy=({target_local_x},{target_local_ys[0]}) "
                    f"context_start_xy=({start_x},{start_y}) "
                    f"context_cols={len(context_cells)} context_rows={context_rows} "
                    f"context_tokens={int(prompt.numel())} "
                    f"context_unsorted_cells={context_unsorted_cells}.",
                    flush=True,
                )

            if len(target_local_ys) == 1:
                column_tokens, column_rows = _sample_sparse_column_tokenwise(
                    gpt,
                    prompt,
                    target_local_x=target_local_x,
                    target_local_y=target_local_ys[0],
                    z=z,
                    temperature=outpainting_temperature,
                    top_k=outpainting_top_k,
                    top_p=outpainting_top_p,
                    feature_temperature=feature_temperature,
                    feature_top_k=feature_top_k,
                    feature_top_p=feature_top_p,
                    base_seed=base_seed,
                    rank=rank,
                    scene_idx=scene_idx,
                    col_idx=col_idx,
                    no_empty_columns=no_empty_columns,
                    pos_token_to_coord=pos_token_to_coord,
                    debug_column_context=debug_column_context,
                )
                sampled_columns.append((y_targets[0], column_tokens, column_rows))
                advance_cols = 1
            else:
                group_outputs, generated_count = _sample_sparse_column_group_tokenwise(
                    gpt,
                    prompt,
                    target_local_x=target_local_x,
                    target_local_ys=target_local_ys,
                    z=z,
                    temperature=outpainting_temperature,
                    top_k=outpainting_top_k,
                    top_p=outpainting_top_p,
                    feature_temperature=feature_temperature,
                    feature_top_k=feature_top_k,
                    feature_top_p=feature_top_p,
                    base_seed=base_seed,
                    rank=rank,
                    scene_idx=scene_idx,
                    col_idx=col_idx,
                    no_empty_columns=no_empty_columns,
                    pos_token_to_coord=pos_token_to_coord,
                    debug_column_context=debug_column_context,
                )
                advance_cols = max(1, min(int(generated_count), len(y_targets)))
                for sample_idx in range(advance_cols):
                    column_tokens, column_rows = group_outputs[sample_idx]
                    sampled_columns.append(
                        (y_targets[sample_idx], column_tokens, column_rows)
                    )

        group_retry_idx = 1
        write_idx = 0
        while write_idx < len(sampled_columns):
            target_y, column_tokens, column_rows = sampled_columns[write_idx]
            if resample_empty_columns and len(column_rows) == 0:
                remaining_cols = len(sampled_columns) - write_idx
                if group_window is not None and remaining_cols > 1:
                    if group_retry_idx > max_empty_column_retries:
                        print(
                            f"WARNING: reached max grouped retries ({max_empty_column_retries}): col_idx={col_idx} "
                            f"x={x} target_y={target_y}. Keeping current sampled suffix.",
                            flush=True,
                        )
                    else:
                        retry_temperature = _temperature_for_retry(
                            outpainting_temperature, group_retry_idx
                        )
                        if group_retry_idx % 5 == 4:
                            print(
                                f"WARNING: grouped retry {group_retry_idx}: col_idx={col_idx} "
                                f"x={x} target_y={target_y} retry_temperature={retry_temperature}",
                                flush=True,
                            )
                        group_start_x, group_start_y, group_target_local_x = (
                            group_window
                        )
                        suffix_targets = [
                            int(sampled_columns[idx][0])
                            for idx in range(write_idx, len(sampled_columns))
                        ]
                        suffix_target_local_ys = [
                            int(target - group_start_y) for target in suffix_targets
                        ]
                        retry_prompt = _build_sparse_prompt_tokens_from_window(
                            rows_grid,
                            target_x=x,
                            target_y=suffix_targets[0],
                            start_x=group_start_x,
                            start_y=group_start_y,
                            scene_cols_x=scene_cols_x,
                            scene_cols_y=scene_cols_y,
                            context_chunks_x=context_chunks_x,
                            context_chunks_y=context_chunks_y,
                            num_position_tokens=int(gpt.num_position_tokens),
                            base_side_length=int(gpt.position_side_length),
                        )
                        group_outputs, generated_count = (
                            _sample_sparse_column_group_tokenwise(
                                gpt,
                                retry_prompt,
                                target_local_x=group_target_local_x,
                                target_local_ys=suffix_target_local_ys,
                                z=z,
                                temperature=retry_temperature,
                                top_k=outpainting_top_k,
                                top_p=outpainting_top_p,
                                feature_temperature=feature_temperature,
                                feature_top_k=feature_top_k,
                                feature_top_p=feature_top_p,
                                base_seed=base_seed,
                                rank=rank,
                                scene_idx=scene_idx,
                                col_idx=col_idx,
                                step_idx=int(
                                    2000 + suffix_targets[0] * 131 + group_retry_idx
                                ),
                                no_empty_columns=no_empty_columns,
                                pos_token_to_coord=pos_token_to_coord,
                                debug_column_context=debug_column_context,
                            )
                        )
                        regenerated_suffix: List[
                            Tuple[int, torch.Tensor, SparseColumnRows]
                        ] = []
                        suffix_len = len(suffix_targets)
                        for suffix_idx in range(suffix_len):
                            if suffix_idx < int(generated_count):
                                res_tokens, res_rows = group_outputs[suffix_idx]
                            else:
                                res_tokens = torch.empty(0, dtype=torch.long)
                                res_rows = []
                            regenerated_suffix.append(
                                (suffix_targets[suffix_idx], res_tokens, res_rows)
                            )
                        sampled_columns = (
                            sampled_columns[:write_idx] + regenerated_suffix
                        )
                        group_retry_idx += 1
                        continue
                else:
                    retry_idx = 1
                    while (
                        len(column_rows) == 0 and retry_idx <= max_empty_column_retries
                    ):
                        retry_temperature = _temperature_for_retry(
                            outpainting_temperature, retry_idx
                        )
                        if retry_idx % 5 == 4:
                            print(
                                f"WARNING: retrying column for {retry_idx}th time:",
                                "col_idx",
                                col_idx,
                                "x",
                                x,
                                "target_y",
                                target_y,
                                "retry_temperature",
                                retry_temperature,
                                flush=True,
                            )

                        (
                            retry_start_x,
                            retry_start_y,
                            retry_target_local_x,
                            retry_target_local_y,
                        ) = _outpainting_window_for_target(
                            x,
                            target_y,
                            scene_cols_x=scene_cols_x,
                            scene_cols_y=scene_cols_y,
                            context_chunks_x=context_chunks_x,
                            context_chunks_y=context_chunks_y,
                            x_offset=x_offset,
                            y_offset=y_offset,
                            y_range=y_range,
                        )
                        retry_prompt = _build_sparse_prompt_tokens_from_window(
                            rows_grid,
                            target_x=x,
                            target_y=target_y,
                            start_x=retry_start_x,
                            start_y=retry_start_y,
                            scene_cols_x=scene_cols_x,
                            scene_cols_y=scene_cols_y,
                            context_chunks_x=context_chunks_x,
                            context_chunks_y=context_chunks_y,
                            num_position_tokens=int(gpt.num_position_tokens),
                            base_side_length=int(gpt.position_side_length),
                        )
                        column_tokens, column_rows = _sample_sparse_column_tokenwise(
                            gpt,
                            retry_prompt,
                            target_local_x=retry_target_local_x,
                            target_local_y=retry_target_local_y,
                            z=z,
                            temperature=retry_temperature,
                            top_k=outpainting_top_k,
                            top_p=outpainting_top_p,
                            feature_temperature=feature_temperature,
                            feature_top_k=feature_top_k,
                            feature_top_p=feature_top_p,
                            base_seed=base_seed,
                            rank=rank,
                            scene_idx=scene_idx,
                            col_idx=col_idx,
                            step_idx=int(1000 + target_y * 131 + retry_idx),
                            no_empty_columns=no_empty_columns,
                            pos_token_to_coord=pos_token_to_coord,
                            debug_column_context=debug_column_context,
                        )
                        retry_idx += 1
                    if len(column_rows) == 0 and retry_idx > max_empty_column_retries:
                        print(
                            f"WARNING: reached max retries ({max_empty_column_retries}): col_idx={col_idx} "
                            f"x={x} target_y={target_y}. Keeping empty column.",
                            flush=True,
                        )

            rows_grid[x][target_y] = column_rows
            column_token_lengths[x, target_y] = int(column_tokens.numel())
            accepted_rows_total += int(column_tokens.numel() // 2)
            if column_tokens.numel() > 0:
                token_pieces.append(column_tokens)
            done_columns += 1
            if len(column_rows) > 0:
                group_retry_idx = 1
            write_idx += 1
            progress_pct = int((100 * done_columns) / total_columns)
            while progress_pct >= next_progress_pct and next_progress_pct <= 100:
                elapsed_s = time.perf_counter() - scene_start
                cols_per_s = done_columns / max(elapsed_s, 1e-6)
                remaining_cols = total_columns - done_columns
                eta_s = remaining_cols / max(cols_per_s, 1e-6)
                occupancy_frac = float(accepted_rows_total) / float(
                    max(done_columns * z, 1)
                )
                occupancy_pct = 100.0 * occupancy_frac
                print(
                    f"[rank {rank}] scene {scene_idx} sampling progress: "
                    f"{next_progress_pct}% ({done_columns}/{total_columns} columns), "
                    f"elapsed={elapsed_s:.1f}s, col/s={cols_per_s:.2f}, eta={eta_s:.1f}s, "
                    f"occ={occupancy_frac:.4f} ({occupancy_pct:.2f}%).",
                    flush=True,
                )
                if (
                    early_stop_min_occupancy is not None
                    and occupancy_frac < early_stop_min_occupancy
                ):
                    print(
                        f"[rank {rank}] scene {scene_idx} early stop: "
                        f"occ={occupancy_frac:.4f} ({occupancy_pct:.2f}%) < "
                        f"threshold={early_stop_min_occupancy:.4f} ({100.0 * early_stop_min_occupancy:.2f}%) "
                        f"at {next_progress_pct}% progress.",
                        flush=True,
                    )
                    early_stopped = True
                    break
                next_progress_pct += 10
            if early_stopped:
                break

        if early_stopped:
            break

        col_idx += int(max(1, advance_cols))

    if token_pieces:
        tokens_sequence = torch.cat(token_pieces, dim=0)
    else:
        tokens_sequence = torch.empty(0, dtype=torch.long)

    return tokens_sequence, column_token_lengths, rows_grid


@torch.no_grad()
def _decode_sparse_rows_grid(
    gpt: GaussianGPT,
    rows_grid: SparseRowsGrid,
    *,
    z: int,
) -> Dict[str, torch.Tensor]:
    device = next(gpt.parameters()).device
    feature_offset = int(gpt.feature_token_offset)
    feature_vocab_size = int(gpt.feature_vocab_size)
    scene_cols_x = len(rows_grid)
    scene_cols_y = len(rows_grid[0]) if scene_cols_x > 0 else 0

    # Explicit sparse occupancy grid to guarantee value placement before decode.
    feature_grid = torch.full(
        (scene_cols_x, scene_cols_y, z),
        -1,
        device=device,
        dtype=torch.long,
    )
    for x, cols in enumerate(rows_grid):
        for y, rows in enumerate(cols):
            for z_idx, feature_tokens in rows:
                if z_idx < 0 or z_idx >= z:
                    continue
                feat = feature_tokens.to(device=device, dtype=torch.long)
                if feat.numel() < 1:
                    continue
                raw_feat = int(feat.view(-1)[0].item()) - feature_offset
                if raw_feat < 0 or raw_feat >= feature_vocab_size:
                    raw_feat = 0
                feature_grid[x, y, z_idx] = int(raw_feat)

    valid_mask = feature_grid >= 0
    if not bool(valid_mask.any().item()):
        return gpt._empty_scene_dict(device)

    coords = valid_mask.nonzero(as_tuple=False).to(dtype=torch.long)
    feature_ids = feature_grid[valid_mask].unsqueeze(1)
    return gpt.vqvae.decode(coords, feature_ids)


def _filter_scene_by_z_quantile(
    payload: Dict[str, torch.Tensor], quantile: float
) -> Tuple[Dict[str, torch.Tensor], float]:
    coords = payload.get("coords")
    if coords is None:
        raise ValueError("Decoded payload has no 'coords' key.")
    if coords.numel() == 0:
        raise ValueError("Decoded scene is empty.")

    # TODO quantile fails with large inputs - use a workaround instead.
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


def _add_dir_to_zip(zip_file: zipfile.ZipFile, directory: Path, arc_root: str) -> None:
    if not directory.exists():
        return
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            rel = path.relative_to(directory)
            zip_file.write(path, arcname=str(Path(arc_root) / rel))


def _to_cpu_payload(payload: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu() if torch.is_tensor(value) else value
        for key, value in payload.items()
    }


def _is_cuda_oom_error(exc: RuntimeError) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg and "cuda" in msg


def _validate_and_derive_geometry(gpt: GaussianGPT) -> Tuple[List[int], int, int, int]:
    if bool(getattr(gpt, "dense_chunks", False)):
        raise ValueError("generate_scene only supports sparse checkpoints.")
    if str(getattr(gpt, "chunk_order", "")) != "xyz":
        raise ValueError("generate_scene requires chunk_order='xyz'.")
    if gpt.chunk_shape is None or len(gpt.chunk_shape) != 3:
        raise ValueError("Sparse checkpoint must provide a 3D chunk_shape.")
    if int(gpt.num_position_tokens) != 1:
        raise ValueError("This script requires num_position_tokens == 1.")
    num_feature_tokens = int(gpt.vqvae.autoencoder.vq.num_tokens)
    if num_feature_tokens != 1:
        raise ValueError("This script requires num_feature_tokens == 1.")
    if int(gpt.tokens_per_latent) != 2:
        raise ValueError("This script requires tokens_per_latent == 2.")

    chunk_shape = [int(v) for v in gpt.chunk_shape]
    context_chunks_x = int(chunk_shape[0])
    context_chunks_y = int(chunk_shape[1])
    z = int(chunk_shape[2])
    if context_chunks_x < 1 or context_chunks_y < 1 or z < 1:
        raise ValueError(f"Invalid non-positive chunk dimensions: {chunk_shape}.")

    required_n_ctx = (
        context_chunks_x * context_chunks_y * z * int(gpt.tokens_per_latent)
    )
    model_n_ctx = int(gpt.model_config.n_ctx)
    if model_n_ctx < required_n_ctx:
        raise ValueError(
            f"Model context is too small for full chunk context. "
            f"Need n_ctx >= {required_n_ctx}, got n_ctx={model_n_ctx}."
        )

    return chunk_shape, context_chunks_x, context_chunks_y, z


def _build_pos_token_to_coord_lut(gpt: GaussianGPT) -> torch.Tensor:
    side = int(gpt.position_side_length)
    vocab = int(gpt.position_vocab_size)
    ids = torch.arange(vocab, dtype=torch.long)
    z = ids % side
    y = (ids // side) % side
    x = (ids // (side * side)) % side
    return torch.stack([x, y, z], dim=1)


def _resolve_stage_sampling_params(
    cfg: DictConfig,
    *,
    stage_prefix: str,
    fallback_temperature: float,
    fallback_top_k: Optional[int],
    fallback_top_p: Optional[float],
) -> Tuple[float, Optional[int], Optional[float]]:
    stage_temperature = cfg.get(f"{stage_prefix}_temperature")
    stage_top_k = cfg.get(f"{stage_prefix}_top_k")
    stage_top_p = cfg.get(f"{stage_prefix}_top_p")
    resolved_temperature = (
        float(stage_temperature)
        if stage_temperature is not None
        else float(fallback_temperature)
    )
    resolved_top_k = int(stage_top_k) if stage_top_k is not None else fallback_top_k
    resolved_top_p = float(stage_top_p) if stage_top_p is not None else fallback_top_p
    return resolved_temperature, resolved_top_k, resolved_top_p


@hydra.main(
    config_path="conf",
    config_name="generate_scene",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    rank, world_size = _resolve_shard_env()
    device = _setup_single_gpu()
    print(
        f"[shard {rank}/{world_size}] device={device}",
        flush=True,
    )

    try:

        checkpoint_path = Path(str(cfg.checkpoint))
        _wait_for_checkpoint(checkpoint_path, int(cfg.checkpoint_timeout))

        vqvae_checkpoint = cfg.get("vqvae_checkpoint")
        if vqvae_checkpoint:
            print(
                f"[rank {rank}] Loading VQ-VAE checkpoint: {vqvae_checkpoint}",
                flush=True,
            )
            vqvae = GaussianVQVAE.load_from_checkpoint(str(vqvae_checkpoint)).eval()
            gpt = GaussianGPT.load_from_checkpoint(str(checkpoint_path), vqvae=vqvae)
        else:
            try:
                gpt = GaussianGPT.load_from_checkpoint(str(checkpoint_path))
            except Exception as exc:
                raise RuntimeError(
                    "Failed to load GaussianGPT checkpoint without explicit VQ-VAE. "
                    "Provide `vqvae_checkpoint=...`."
                ) from exc
        gpt = gpt.eval().to(device)

        num_scenes = int(cfg.num_scenes)
        scene_cols_x = int(cfg.scene_cols_x)
        scene_cols_y = int(cfg.scene_cols_y)
        if num_scenes < 0:
            raise ValueError("num_scenes must be >= 0.")
        if scene_cols_x < 1 or scene_cols_y < 1:
            raise ValueError("scene_cols_x and scene_cols_y must be >= 1.")

        chunk_shape, context_chunks_x, context_chunks_y, z = (
            _validate_and_derive_geometry(gpt)
        )
        x_offset = int(cfg.get("x_offset", 0))
        y_offset = int(cfg.get("y_offset", 0))
        y_range = int(cfg.get("y_range", 1))
        raw_enable_bootstrap = cfg.get("enable_bootstrap", True)
        if not isinstance(raw_enable_bootstrap, bool):
            raise ValueError("enable_bootstrap must be a boolean.")
        enable_bootstrap = bool(raw_enable_bootstrap)
        raw_resample_empty_columns = cfg.get("resample_empty_columns", False)
        if not isinstance(raw_resample_empty_columns, bool):
            raise ValueError("resample_empty_columns must be a boolean.")
        resample_empty_columns = bool(raw_resample_empty_columns)
        raw_no_empty_columns = cfg.get("no_empty_columns", False)
        if not isinstance(raw_no_empty_columns, bool):
            raise ValueError("no_empty_columns must be a boolean.")
        no_empty_columns = bool(raw_no_empty_columns)
        if no_empty_columns and resample_empty_columns:
            print(
                "[rank "
                + str(rank)
                + "] no_empty_columns=true disables resample_empty_columns.",
                flush=True,
            )
            resample_empty_columns = False
        if x_offset < 0:
            raise ValueError("x_offset must be >= 0.")
        if y_offset < 0:
            raise ValueError("y_offset must be >= 0.")
        if y_range < 1:
            raise ValueError("y_range must be >= 1.")
        if x_offset > context_chunks_x - 1:
            raise ValueError("x_offset must be <= context_chunks_x - 1.")
        if y_offset + y_range - 1 > context_chunks_y - 1:
            raise ValueError("y_offset + y_range - 1 must be <= context_chunks_y - 1.")
        raw_max_empty_column_retries = cfg.get("max_empty_column_retries", MAX_RETRIES)
        if not isinstance(raw_max_empty_column_retries, int) or isinstance(
            raw_max_empty_column_retries, bool
        ):
            raise ValueError("max_empty_column_retries must be an int.")
        max_empty_column_retries = int(raw_max_empty_column_retries)
        if max_empty_column_retries < 1:
            raise ValueError("max_empty_column_retries must be >= 1.")
        raw_early_stop_min_occupancy = cfg.get("early_stop_min_occupancy", None)
        early_stop_min_occupancy: Optional[float]
        if raw_early_stop_min_occupancy is None:
            early_stop_min_occupancy = None
        else:
            early_stop_min_occupancy = float(raw_early_stop_min_occupancy)
            if early_stop_min_occupancy < 0.0 or early_stop_min_occupancy > 1.0:
                raise ValueError("early_stop_min_occupancy must be in [0, 1] or null.")

        pos_token_to_coord = _build_pos_token_to_coord_lut(gpt)
        print(
            f"[rank {rank}] sparse large-scene geometry from checkpoint: "
            f"chunk_shape={chunk_shape}, context=({context_chunks_x}, {context_chunks_y}), z={z}",
            flush=True,
        )

        debug_column_context = bool(cfg.get("debug_column_context", False))
        seed = int(cfg.seed)
        base_temperature = float(cfg.temperature)
        base_top_k = int(cfg.top_k) if cfg.get("top_k") is not None else None
        base_top_p = float(cfg.top_p) if cfg.get("top_p") is not None else None
        feature_temperature, feature_top_k, feature_top_p = (
            _resolve_feature_sampling_params(cfg)
        )
        outpainting_temperature, outpainting_top_k, outpainting_top_p = (
            _resolve_stage_sampling_params(
                cfg,
                stage_prefix="outpainting",
                fallback_temperature=base_temperature,
                fallback_top_k=base_top_k,
                fallback_top_p=base_top_p,
            )
        )
        bootstrap_temperature, bootstrap_top_k, bootstrap_top_p = (
            _resolve_stage_sampling_params(
                cfg,
                stage_prefix="bootstrap",
                fallback_temperature=outpainting_temperature,
                fallback_top_k=outpainting_top_k,
                fallback_top_p=outpainting_top_p,
            )
        )
        bootstrap2_temperature, bootstrap2_top_k, bootstrap2_top_p = (
            _resolve_stage_sampling_params(
                cfg,
                stage_prefix="bootstrap2",
                fallback_temperature=outpainting_temperature,
                fallback_top_k=outpainting_top_k,
                fallback_top_p=outpainting_top_p,
            )
        )

        local_n, global_start, global_end = _shard_range(num_scenes, rank, world_size)
        global_scene_indices = torch.arange(global_start, global_end, dtype=torch.long)
        decode_outputs = bool(cfg.get("decode_outputs", False))
        render_topdown = bool(cfg.get("render_topdown", False))
        topdown_quantile = float(cfg.get("topdown_quantile", 75.0))
        topdown_resolution = int(cfg.get("topdown_resolution", 1024))
        topdown_background_color = str(cfg.get("topdown_background_color", "white"))
        if render_topdown and not decode_outputs:
            raise ValueError("render_topdown requires decode_outputs=true.")
        if topdown_quantile < 0.0 or topdown_quantile > 100.0:
            raise ValueError("topdown_quantile must be in [0, 100].")
        if topdown_resolution < 1:
            raise ValueError("topdown_resolution must be >= 1.")
        if topdown_background_color not in ("white", "black"):
            raise ValueError("topdown_background_color must be 'white' or 'black'.")

        print(
            f"[rank {rank}] assigned {local_n} scenes in global range "
            f"[{global_start}, {global_end}).",
            flush=True,
        )
        print(
            f"[rank {rank}] sampling params: "
            f"outpainting=(temperature={outpainting_temperature}, top_k={outpainting_top_k}, top_p={outpainting_top_p}), "
            f"bootstrap=(temperature={bootstrap_temperature}, top_k={bootstrap_top_k}, top_p={bootstrap_top_p}), "
            f"bootstrap2=(temperature={bootstrap2_temperature}, top_k={bootstrap2_top_k}, top_p={bootstrap2_top_p}), "
            f"feature_overrides=(temperature={feature_temperature}, top_k={feature_top_k}, top_p={feature_top_p}), "
            f"x_offset={x_offset}, y_offset={y_offset}, y_range={y_range}, "
            f"no_empty_columns={no_empty_columns}, "
            f"resample_empty_columns={resample_empty_columns}, "
            f"max_empty_column_retries={max_empty_column_retries}, "
            f"enable_bootstrap={enable_bootstrap}, "
            f"early_stop_min_occupancy={early_stop_min_occupancy}.",
            flush=True,
        )

        output_dir = Path(str(cfg.output_dir))
        shards_dir = output_dir / "shards"
        tokens_rank_dir = output_dir / "tokens" / f"rank_{rank:04d}"
        gaussians_rank_dir = output_dir / "gaussians" / f"rank_{rank:04d}"
        topdown_rank_dir = output_dir / "topdown" / f"rank_{rank:04d}"
        shards_dir.mkdir(parents=True, exist_ok=True)
        tokens_rank_dir.mkdir(parents=True, exist_ok=True)
        if decode_outputs:
            gaussians_rank_dir.mkdir(parents=True, exist_ok=True)
        if render_topdown:
            topdown_rank_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.perf_counter()
        local_effective_tokens = 0
        decode_oom_scene_indices: List[int] = []
        scene_token_filenames: List[str] = []

        def _find_existing_outputs(scene_idx: int):
            # Scan ALL prior rank dirs (any past shard layout), not just the
            # current one — resumes with a different world_size still skip work.
            png_name = f"scene_{scene_idx:07d}.png"
            pt_name = f"scene_{scene_idx:07d}.pt"
            tokens_root = output_dir / "tokens"
            gaussians_root = output_dir / "gaussians"
            topdown_root = output_dir / "topdown"
            tok = (
                next(iter(tokens_root.glob(f"rank_*/{pt_name}")), None)
                if tokens_root.exists()
                else None
            )
            gau = (
                next(iter(gaussians_root.glob(f"rank_*/{pt_name}")), None)
                if gaussians_root.exists()
                else None
            )
            top = (
                next(iter(topdown_root.glob(f"rank_*/{png_name}")), None)
                if topdown_root.exists()
                else None
            )
            return tok, gau, top

        for local_scene_idx in range(local_n):
            global_scene_idx = global_start + local_scene_idx

            scene_token_filename = f"scene_{global_scene_idx:07d}.pt"
            scene_token_path = tokens_rank_dir / scene_token_filename
            existing_tok, existing_gau, existing_top = _find_existing_outputs(
                global_scene_idx
            )
            tok_ok = existing_tok is not None
            gau_ok = (not decode_outputs) or existing_gau is not None
            top_ok = (not render_topdown) or existing_top is not None
            if tok_ok and gau_ok and top_ok:
                try:
                    existing_tokens = int(
                        torch.load(
                            existing_tok,
                            map_location="cpu",
                            weights_only=False,
                        )["tokens_sequence"].numel()
                    )
                except Exception as exc:  # corrupt sidecar — fall through to re-sample
                    print(
                        f"[rank {rank}] scene {global_scene_idx} sidecar unreadable "
                        f"({exc!r}); re-sampling.",
                        flush=True,
                    )
                else:
                    scene_token_filenames.append(scene_token_filename)
                    local_effective_tokens += existing_tokens
                    print(
                        f"[rank {rank}] scene {local_scene_idx + 1}/{local_n} skipped "
                        f"(global_idx={global_scene_idx}, tokens={existing_tokens}, "
                        f"found_at={existing_tok.parent.name}).",
                        flush=True,
                    )
                    continue

            scene_tokens, column_token_lengths, rows_grid = _generate_sparse_scene(
                gpt,
                scene_cols_x=scene_cols_x,
                scene_cols_y=scene_cols_y,
                context_chunks_x=context_chunks_x,
                context_chunks_y=context_chunks_y,
                z=z,
                outpainting_temperature=outpainting_temperature,
                outpainting_top_k=outpainting_top_k,
                outpainting_top_p=outpainting_top_p,
                bootstrap_temperature=bootstrap_temperature,
                bootstrap_top_k=bootstrap_top_k,
                bootstrap_top_p=bootstrap_top_p,
                bootstrap2_temperature=bootstrap2_temperature,
                bootstrap2_top_k=bootstrap2_top_k,
                bootstrap2_top_p=bootstrap2_top_p,
                feature_temperature=feature_temperature,
                feature_top_k=feature_top_k,
                feature_top_p=feature_top_p,
                x_offset=x_offset,
                y_offset=y_offset,
                y_range=y_range,
                no_empty_columns=no_empty_columns,
                resample_empty_columns=resample_empty_columns,
                enable_bootstrap=enable_bootstrap,
                early_stop_min_occupancy=early_stop_min_occupancy,
                max_empty_column_retries=max_empty_column_retries,
                base_seed=seed,
                rank=rank,
                scene_idx=global_scene_idx,
                pos_token_to_coord=pos_token_to_coord,
                debug_column_context=debug_column_context,
            )
            scene_token_count = int(scene_tokens.numel())
            local_effective_tokens += scene_token_count

            scene_token_payload = {
                "scene_idx": int(global_scene_idx),
                "tokens_sequence": scene_tokens.detach().cpu().to(dtype=torch.long),
                "column_token_lengths": column_token_lengths.detach()
                .cpu()
                .to(dtype=torch.long),
            }
            torch.save(scene_token_payload, scene_token_path)
            scene_token_filenames.append(scene_token_filename)

            if decode_outputs:
                decoded = None
                try:
                    decoded = _decode_sparse_rows_grid(gpt, rows_grid, z=z)
                    if render_topdown:
                        try:
                            _render_topdown_png(
                                decoded,
                                output_path=topdown_rank_dir
                                / f"scene_{global_scene_idx:07d}.png",
                                quantile=topdown_quantile,
                                resolution=topdown_resolution,
                                background_color=topdown_background_color,
                            )
                        except ValueError as exc:
                            print(
                                f"[rank {rank}] topdown skipped for scene {global_scene_idx}: {exc}",
                                flush=True,
                            )
                    torch.save(
                        _to_cpu_payload(decoded),
                        gaussians_rank_dir / f"scene_{global_scene_idx:07d}.pt",
                    )
                except RuntimeError as exc:
                    if _is_cuda_oom_error(exc):
                        decode_oom_scene_indices.append(int(global_scene_idx))
                        print(
                            f"[rank {rank}] decode OOM for scene {global_scene_idx}; "
                            "keeping sampled tokens and continuing.",
                            flush=True,
                        )
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    else:
                        raise
                finally:
                    if decoded is not None:
                        del decoded

            del scene_tokens, column_token_lengths, rows_grid, scene_token_payload
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(
                f"[rank {rank}] scene {local_scene_idx + 1}/{local_n} done "
                f"(global_idx={global_scene_idx}, tokens={scene_token_count}).",
                flush=True,
            )

        elapsed = time.perf_counter() - start_time
        scenes_payload = [
            torch.load(tokens_rank_dir / fname, map_location="cpu", weights_only=False)
            for fname in scene_token_filenames
        ]

        shard_payload = {
            "metadata": {
                "rank": rank,
                "world_size": world_size,
                "global_start_scene_idx": global_start,
                "global_end_scene_idx_exclusive": global_end,
                "num_local_scenes": local_n,
                "checkpoint": str(checkpoint_path),
                "vqvae_checkpoint": str(vqvae_checkpoint) if vqvae_checkpoint else None,
                "dense_chunks": False,
                "chunk_order": "xyz",
                "chunk_shape": chunk_shape,
                "context_chunks_x": context_chunks_x,
                "context_chunks_y": context_chunks_y,
                "z": z,
                "tokens_per_latent": int(gpt.tokens_per_latent),
                "num_position_tokens": int(gpt.num_position_tokens),
                "num_feature_tokens": int(gpt.vqvae.autoencoder.vq.num_tokens),
                "model_n_ctx": int(gpt.model_config.n_ctx),
                "temperature": outpainting_temperature,
                "top_k": outpainting_top_k,
                "top_p": outpainting_top_p,
                "outpainting_temperature": outpainting_temperature,
                "outpainting_top_k": outpainting_top_k,
                "outpainting_top_p": outpainting_top_p,
                "bootstrap_temperature": bootstrap_temperature,
                "bootstrap_top_k": bootstrap_top_k,
                "bootstrap_top_p": bootstrap_top_p,
                "bootstrap2_temperature": bootstrap2_temperature,
                "bootstrap2_top_k": bootstrap2_top_k,
                "bootstrap2_top_p": bootstrap2_top_p,
                "feature_temperature": feature_temperature,
                "feature_top_k": feature_top_k,
                "feature_top_p": feature_top_p,
                "x_offset": x_offset,
                "y_offset": y_offset,
                "y_range": y_range,
                "no_empty_columns": no_empty_columns,
                "resample_empty_columns": resample_empty_columns,
                "max_empty_column_retries": max_empty_column_retries,
                "enable_bootstrap": enable_bootstrap,
                "early_stop_min_occupancy": early_stop_min_occupancy,
                "debug_column_context": debug_column_context,
                "bootstrap_initial_context_block_like_parallel_sampling": enable_bootstrap,
                "bootstrap_second_tier_x0_shifted_y_slices": enable_bootstrap,
                "seed": seed,
                "scene_cols_x": scene_cols_x,
                "scene_cols_y": scene_cols_y,
                "decode_outputs": decode_outputs,
                "render_topdown": render_topdown,
                "topdown_quantile": topdown_quantile,
                "topdown_resolution": topdown_resolution,
                "topdown_background_color": topdown_background_color,
                "scene_token_sidecars_dir": str(tokens_rank_dir),
                "decode_oom_count": len(decode_oom_scene_indices),
                "decode_oom_scene_indices": decode_oom_scene_indices,
                "local_effective_tokens": local_effective_tokens,
                "elapsed_seconds": elapsed,
            },
            "global_scene_indices": global_scene_indices,
            "scenes": scenes_payload,
        }
        shard_path = shards_dir / f"rank_{rank:04d}.pt"
        torch.save(shard_payload, shard_path)

        print(
            f"[rank {rank}] wrote {shard_path} in {elapsed:.2f}s "
            f"(scenes={local_n}, effective_tokens={local_effective_tokens}).",
            flush=True,
        )

        if bool(cfg.get("zip_outputs", False)):
            zip_path = output_dir / str(
                cfg.get(
                    "zip_filename",
                    f"generate_scene_shard_{rank:04d}.zip",
                )
            )
            include_decoded = bool(cfg.get("zip_include_decoded", True))
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.write(shard_path, arcname=f"shards/{shard_path.name}")
                _add_dir_to_zip(zf, tokens_rank_dir, f"tokens/rank_{rank:04d}")
                if include_decoded:
                    _add_dir_to_zip(
                        zf, gaussians_rank_dir, f"gaussians/rank_{rank:04d}"
                    )
                    _add_dir_to_zip(zf, topdown_rank_dir, f"topdown/rank_{rank:04d}")
            print(f"[shard {rank}] wrote zip archive: {zip_path}", flush=True)

    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
