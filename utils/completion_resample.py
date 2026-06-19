"""Empty-column retry for dense-chunk GaussianGPT completion / generation.

Ports the resample-empty-columns concept from
``generate_scene.py`` (where each (x, y) latent column
is sampled as a separate column-token stream and retried up to MAX_RETRIES
when it comes out empty) to the dense per-chunk path used by:

- ``complete_chunks.py`` (per-chunk completion via tokenized DataModule)
- ``gpt.sample()`` consumers (per-chunk unconditional generation)

After the initial whole-chunk sample, reshape the completion region into
(num_cols, cz * num_features) and re-sample any (x, y) column whose feature
slots are all pad. Each retry conditions on everything emitted before that
column in the chunk_order emission order; up to MAX_RETRIES tries per column
with the same temperature ramp helper as the large-sampling code.

Limited to chunk_order='xyz' — that's the only ordering in which one (x, y)
column's tokens form a contiguous emission block, which is what makes
localized retry cheap and well-defined.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from model.gaussian_gpt import GaussianGPT
from utils.pos_tokens import pos_tokens_to_centered_coords

DEFAULT_MAX_RETRIES = 5
# Matches generate_scene._temperature_for_retry: only
# ramp temperature after we've already burned 5 same-temperature retries.
_TEMP_RAMP_AFTER = 5


def _temperature_for_retry(base_temperature: float, retry_idx: int) -> float:
    if retry_idx <= _TEMP_RAMP_AFTER:
        return float(base_temperature)
    return float(base_temperature * (1.0 + 0.1 * float(retry_idx - _TEMP_RAMP_AFTER)))


def _retry_seed(base_seed: int, col_emission_idx: int, retry_idx: int) -> int:
    mod = 2**31 - 1
    return int(
        (
            int(base_seed) * 1_000_003
            + int(col_emission_idx) * 8_191
            + int(retry_idx) * 101
        )
        % mod
    )


@dataclass
class ColumnRetryStats:
    chunk_order: str
    total_completion_columns: int
    initial_empty_columns: int
    retries_performed: int
    columns_filled_by_retry: int
    unresolved_empty_columns: int


def _column_is_empty(
    column_tokens: torch.Tensor, *, num_features: int, pad_id: int
) -> bool:
    if column_tokens.numel() % num_features != 0:
        raise ValueError(
            f"column_tokens.numel()={column_tokens.numel()} not divisible by num_features={num_features}"
        )
    feats = column_tokens.view(-1, num_features)
    return bool((feats == pad_id).all().item())


def _chunk_geometry(gpt: GaussianGPT) -> tuple[int, int, int, int]:
    """Returns ``(cx, cy, cz, num_features)``; raises for non-dense or non-3D shapes."""
    if not getattr(gpt, "dense_chunks", False):
        raise ValueError("Column retry requires dense_chunks=True.")
    if gpt.chunk_shape is None or len(gpt.chunk_shape) != 3:
        raise ValueError(f"chunk_shape must be 3D, got {gpt.chunk_shape}")
    cx, cy, cz = [int(v) for v in gpt.chunk_shape]
    return cx, cy, cz, int(gpt.num_feature_tokens)


@torch.no_grad()
def retry_empty_columns(
    gpt: GaussianGPT,
    sampled: torch.Tensor,
    *,
    prefix_len: int,
    pad_id: int,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
    seed: int,
    max_retries: int = DEFAULT_MAX_RETRIES,
    verbose: bool = False,
) -> tuple[torch.Tensor, ColumnRetryStats]:
    """Run the empty-column retry pass on an already-sampled dense chunk.

    ``sampled`` is a 1D tensor of length ``cx*cy*cz*num_features``. Columns at
    emission positions ``>= ceil(prefix_len / tokens_per_column)`` whose feature
    slots are all pad get re-sampled (in xyz emission order) by conditioning on
    everything emitted before them.
    """
    cx, cy, cz, num_features = _chunk_geometry(gpt)
    full_len = cx * cy * cz * num_features
    if sampled.numel() != full_len:
        raise ValueError(
            f"sampled length {sampled.numel()} does not match full_len {full_len}"
        )
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0.")
    chunk_order = str(getattr(gpt, "chunk_order", "xyz"))
    device = next(gpt.parameters()).device
    sampled = sampled.to(device=device, dtype=torch.long).clone()

    if chunk_order != "xyz":
        # Column retry only works when one (x, y) column = one contiguous emission block.
        if verbose:
            print(
                f"[resample] chunk_order={chunk_order!r} - column retry not supported; "
                f"returning single sample.",
                flush=True,
            )
        return sampled, ColumnRetryStats(
            chunk_order=chunk_order,
            total_completion_columns=0,
            initial_empty_columns=0,
            retries_performed=0,
            columns_filled_by_retry=0,
            unresolved_empty_columns=0,
        )

    tokens_per_column = cz * num_features
    # Only retry columns whose tokens are fully within the completion region —
    # a partial column at the prefix/completion boundary contains conditioning
    # tokens we must not overwrite.
    if prefix_len % tokens_per_column == 0:
        completion_start_col = prefix_len // tokens_per_column
    else:
        completion_start_col = prefix_len // tokens_per_column + 1
    total_columns = cx * cy

    initial_empty = 0
    retries_performed = 0
    columns_filled_by_retry = 0
    unresolved_empty = 0

    for col_emission_idx in range(completion_start_col, total_columns):
        col_start = col_emission_idx * tokens_per_column
        col_end = col_start + tokens_per_column
        column_tokens = sampled[col_start:col_end]
        if not _column_is_empty(
            column_tokens, num_features=num_features, pad_id=pad_id
        ):
            continue

        initial_empty += 1
        if max_retries == 0:
            unresolved_empty += 1
            continue

        filled_this_column = False
        for retry_idx in range(1, max_retries + 1):
            retry_temperature = _temperature_for_retry(temperature, retry_idx)
            retry_seed = _retry_seed(seed, col_emission_idx, retry_idx)
            retry_prompt = sampled[:col_start]
            retried = gpt.gpt.sample_sequence_with_prompt(
                retry_prompt,
                max_new_tokens=tokens_per_column,
                num_samples=1,
                temperature=retry_temperature,
                top_k=top_k,
                top_p=top_p,
                stop_on_eos=False,
                seed=int(retry_seed),
            )[0]
            retries_performed += 1
            new_column = retried[col_start:col_end]
            if new_column.numel() < tokens_per_column:
                pad = torch.full(
                    (tokens_per_column - new_column.numel(),),
                    pad_id,
                    device=device,
                    dtype=torch.long,
                )
                new_column = torch.cat([new_column, pad], dim=0)
            elif new_column.numel() > tokens_per_column:
                new_column = new_column[:tokens_per_column]
            sampled[col_start:col_end] = new_column
            if not _column_is_empty(
                new_column, num_features=num_features, pad_id=pad_id
            ):
                filled_this_column = True
                if verbose:
                    print(
                        f"[resample] col {col_emission_idx} filled after retry {retry_idx} "
                        f"(T={retry_temperature:.3f}).",
                        flush=True,
                    )
                break

        if filled_this_column:
            columns_filled_by_retry += 1
        else:
            unresolved_empty += 1
            if verbose:
                print(
                    f"[resample] col {col_emission_idx} still empty after "
                    f"{max_retries} retries; keeping empty.",
                    flush=True,
                )

    stats = ColumnRetryStats(
        chunk_order=chunk_order,
        total_completion_columns=total_columns - completion_start_col,
        initial_empty_columns=initial_empty,
        retries_performed=retries_performed,
        columns_filled_by_retry=columns_filled_by_retry,
        unresolved_empty_columns=unresolved_empty,
    )
    return sampled, stats


@torch.no_grad()
def complete_with_empty_column_retry(
    gpt: GaussianGPT,
    prefix_tokens: torch.Tensor,
    *,
    full_len: int,
    pad_id: int,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
    seed: int,
    max_retries: int = DEFAULT_MAX_RETRIES,
    verbose: bool = False,
) -> tuple[torch.Tensor, ColumnRetryStats]:
    """Sample a dense chunk completion and re-sample any empty (x, y) column.

    Args:
        gpt: dense-chunks GaussianGPT.
        prefix_tokens: 1D tensor of the conditioning prefix. Length 0 == unconditional.
        full_len: ``cx*cy*cz*num_features`` — total tokens in one chunk.
        pad_id: GPT pad token id (``gpt.gpt.pad_token_id``).
        temperature/top_k/top_p: sampling params (also used for retries, with the
            same temperature ramp helper as the large-sampling code).
        seed: base seed for the initial sample; retries use deterministic
            transforms of (seed, col_emission_idx, retry_idx).
        max_retries: max retries per empty column; 0 disables retry (initial sample only).
        verbose: print per-column retry outcomes.

    Returns:
        (sampled, stats):
            sampled: 1D tensor of length ``full_len``.
            stats: ColumnRetryStats summary (for surfacing in run manifests).
    """
    cx, cy, cz, num_features = _chunk_geometry(gpt)
    expected_full_len = cx * cy * cz * num_features
    if full_len != expected_full_len:
        raise ValueError(
            f"full_len={full_len} does not match chunk_shape*num_features={expected_full_len}"
        )

    device = next(gpt.parameters()).device
    if not torch.is_tensor(prefix_tokens):
        prefix_tokens = torch.tensor(prefix_tokens, dtype=torch.long, device=device)
    prefix_tokens = prefix_tokens.to(device=device, dtype=torch.long)
    prefix_len = int(prefix_tokens.numel())
    remaining_len = full_len - prefix_len

    if remaining_len <= 0:
        return prefix_tokens.clone(), ColumnRetryStats(
            chunk_order=str(getattr(gpt, "chunk_order", "xyz")),
            total_completion_columns=0,
            initial_empty_columns=0,
            retries_performed=0,
            columns_filled_by_retry=0,
            unresolved_empty_columns=0,
        )

    # Initial sample — same call shape as the existing eval drivers.
    if prefix_len > 0:
        initial = gpt.gpt.sample_sequence_with_prompt(
            prefix_tokens,
            max_new_tokens=remaining_len,
            num_samples=1,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stop_on_eos=False,
            seed=int(seed),
        )[0]
    else:
        initial = gpt.gpt.sample_sequence(
            full_len,
            batch_size=1,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stop_on_eos=False,
            seed=int(seed),
        )[0]

    if initial.numel() > full_len:
        initial = initial[:full_len]
    elif initial.numel() < full_len:
        pad = torch.full(
            (full_len - initial.numel(),), pad_id, device=device, dtype=torch.long
        )
        initial = torch.cat([initial, pad], dim=0)

    return retry_empty_columns(
        gpt,
        initial,
        prefix_len=prefix_len,
        pad_id=pad_id,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        seed=seed,
        max_retries=max_retries,
        verbose=verbose,
    )


# ============================================================================
# Sparse-mode column retry. The GPT emits (pos, feat) rows in chunk_order; an
# "empty column" is an (x_lat, y_lat) pair that has no row in the completion
# region. Retry by re-sampling with a prompt that ends at the right chunk_order
# position and filtering accepted rows by target (x, y).
# ============================================================================


def _build_pos_token_to_coord(gpt: GaussianGPT) -> torch.Tensor:
    """Return a (position_vocab_size, 3) LongTensor mapping pos_token_id -> (x, y, z)
    in latent coords. Only supports ``num_position_tokens == 1`` (the vfront ggpt
    setup); raises otherwise so the caller can fall back."""
    if int(gpt.num_position_tokens) != 1:
        raise NotImplementedError(
            f"Sparse column retry currently supports num_position_tokens=1 only; "
            f"got {int(gpt.num_position_tokens)}"
        )
    vocab = int(gpt.position_vocab_size)
    side = int(gpt.position_side_length)
    ids = torch.arange(vocab, dtype=torch.long).unsqueeze(-1)  # [V, 1]
    coords = pos_tokens_to_centered_coords(ids, 1, side).to(dtype=torch.long)  # [V, 3]
    return coords


def _chunk_order_key(coords: torch.Tensor, chunk_order: str, side: int) -> torch.Tensor:
    """Sort key for rows in chunk_order. coords: (N, 3). Returns (N,) long."""
    if chunk_order == "xyz":
        return coords[:, 0] * (side * side) + coords[:, 1] * side + coords[:, 2]
    if chunk_order == "xzy":
        return coords[:, 0] * (side * side) + coords[:, 2] * side + coords[:, 1]
    # Hilbert / Z-order: import on demand and use the same encoder
    from serialization import encode  # noqa: WPS433

    depth = int(max(coords.max().item() + 1, 1)).bit_length()
    return encode(coords, depth=depth, order=chunk_order)


def _parse_sparse_rows(
    tokens: torch.Tensor,
    *,
    tokens_per_latent: int,
    num_position_tokens: int,
    position_vocab_size: int,
    feature_offset: int,
    feature_vocab_size: int,
    pad_id: int,
    pos_token_to_coord: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Parse a flat token stream into (token_rows, coords, valid_mask).

    token_rows: (R, tokens_per_latent) cpu long
    coords:     (R, 3) cpu long  (decoded from pos_tokens; junk rows -> -1)
    valid_mask: (R,)   cpu bool  (True iff non-pad + pos/feat in-vocab)
    """
    usable_len = (tokens.numel() // tokens_per_latent) * tokens_per_latent
    if usable_len <= 0:
        empty_rows = torch.empty(0, tokens_per_latent, dtype=torch.long)
        empty_coords = torch.empty(0, 3, dtype=torch.long)
        empty_mask = torch.empty(0, dtype=torch.bool)
        return empty_rows, empty_coords, empty_mask
    token_rows = tokens[:usable_len].detach().cpu().view(-1, tokens_per_latent).long()
    pos_tokens = token_rows[:, :num_position_tokens]
    feature_ids = token_rows[:, num_position_tokens:]
    non_pad = ~(feature_ids == pad_id).all(dim=1)
    pos_valid = ((pos_tokens >= 0) & (pos_tokens < position_vocab_size)).all(dim=1)
    feat_valid = (
        (feature_ids >= feature_offset)
        & (feature_ids < feature_offset + feature_vocab_size)
    ).all(dim=1)
    valid = non_pad & pos_valid & feat_valid
    coords = torch.full((token_rows.shape[0], 3), -1, dtype=torch.long)
    if valid.any():
        coords[valid] = pos_token_to_coord[pos_tokens[valid, 0]]
    return token_rows, coords, valid


@torch.no_grad()
def _retry_sample_sparse_column(
    gpt: GaussianGPT,
    prompt_tokens: torch.Tensor,
    *,
    target_local_x: int,
    target_local_y: int,
    z: int,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
    seed: int,
    pos_token_to_coord: torch.Tensor,
) -> torch.Tensor:
    """Sample up to 2*z new tokens after ``prompt_tokens`` and return ONLY the
    accepted rows whose pos_token decodes to (target_local_x, target_local_y, *).
    Returns a flat (rows*tokens_per_latent,) tensor; empty if nothing accepted."""
    position_vocab_size = int(gpt.position_vocab_size)
    feature_offset = int(gpt.feature_token_offset)
    feature_vocab_size = int(gpt.feature_vocab_size)

    prompt_cpu = prompt_tokens.detach().cpu().to(dtype=torch.long)
    sampled = gpt.gpt.sample_sequence_with_prompt(
        prompt_cpu,
        max_new_tokens=int(2 * z),
        num_samples=1,
        temperature=float(temperature),
        top_k=top_k,
        top_p=top_p,
        stop_on_eos=True,
        seed=int(seed),
    )[0]
    prompt_len = int(prompt_cpu.numel())
    new_tokens = sampled[prompt_len:]
    if new_tokens.numel() < 2:
        return torch.empty(0, dtype=torch.long)
    usable_len = (new_tokens.numel() // 2) * 2
    rows = new_tokens[:usable_len].detach().cpu().view(-1, 2).long()
    accepted: List[torch.Tensor] = []
    for row in rows:
        pos_id = int(row[0].item())
        feat_id = int(row[1].item())
        if pos_id < 0 or pos_id >= position_vocab_size:
            break
        if feat_id < feature_offset or feat_id >= feature_offset + feature_vocab_size:
            break
        coord = pos_token_to_coord[pos_id]
        px, py, pz = int(coord[0]), int(coord[1]), int(coord[2])
        if px != target_local_x or py != target_local_y:
            break
        if pz < 0 or pz >= z:
            break
        accepted.append(row)
    if not accepted:
        return torch.empty(0, dtype=torch.long)
    return torch.cat(accepted, dim=0)


@torch.no_grad()
def complete_sparse_with_empty_column_retry(
    gpt: GaussianGPT,
    prefix_tokens: torch.Tensor,
    *,
    pad_id: int,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
    seed: int,
    initial_max_new_tokens: int,
    max_retries: int = DEFAULT_MAX_RETRIES,
    verbose: bool = False,
) -> tuple[torch.Tensor, ColumnRetryStats]:
    """Sparse-mode per-chunk completion with first-hole resample.

    Algorithm:
      1. Do one whole-shot ``sample_sequence_with_prompt`` (the fast path).
      2. Parse + sort the initial completion rows by chunk_order.
      3. Walk the completion region in chunk_order. As long as the initial
         sample emitted at least one row at the current (tx, ty), accept those
         rows. At the first hole, DISCARD the entire rest of the initial
         completion -- those rows were sampled under a context that included
         "no Gaussian at this (tx, ty)", and refilling the hole invalidates
         everything that came after.
      4. From the cutoff onwards, sample column-by-column with retry. For each
         (tx, ty) in chunk_order:
           - prompt = prefix_tokens + flatten(accepted_rows so far)
           - sample 2*cz tokens with stop_on_eos=True, filter accepted rows by
             (tx, ty, z in [0, cz)). Empty -> retry up to MAX_RETRIES with
             same-temp-then-ramp (same _temperature_for_retry as the source).
           - If still empty after MAX_RETRIES: accept the empty column, move on.

    Limited to ``chunk_order='xyz'`` (other orders fall back to bare initial
    sample, no retry).
    """
    if getattr(gpt, "dense_chunks", False):
        raise ValueError(
            "complete_sparse_with_empty_column_retry requires sparse mode."
        )
    if gpt.chunk_shape is None or len(gpt.chunk_shape) != 3:
        raise ValueError(f"chunk_shape must be 3D, got {gpt.chunk_shape}")
    cx, cy, cz = [int(v) for v in gpt.chunk_shape]
    chunk_order = str(getattr(gpt, "chunk_order", "xyz"))

    device = next(gpt.parameters()).device
    if not torch.is_tensor(prefix_tokens):
        prefix_tokens = torch.tensor(prefix_tokens, dtype=torch.long, device=device)
    prefix_tokens = prefix_tokens.to(device=device, dtype=torch.long)
    prefix_len = int(prefix_tokens.numel())

    # Initial sample (the original bare-completion behavior).
    if initial_max_new_tokens <= 0:
        initial_sampled = prefix_tokens.clone()
    else:
        sampled = gpt.gpt.sample_sequence_with_prompt(
            prefix_tokens,
            max_new_tokens=int(initial_max_new_tokens),
            num_samples=1,
            temperature=float(temperature),
            top_k=top_k,
            top_p=top_p,
            stop_on_eos=False,
            seed=int(seed),
        )[0]
        initial_sampled = sampled.to(device=device, dtype=torch.long)

    tokens_per_latent = int(gpt.tokens_per_latent)
    num_position_tokens = int(gpt.num_position_tokens)
    position_vocab_size = int(gpt.position_vocab_size)
    side = int(gpt.position_side_length)
    feature_offset = int(gpt.feature_token_offset)
    feature_vocab_size = int(gpt.feature_vocab_size)

    try:
        pos_token_to_coord = _build_pos_token_to_coord(gpt)
    except NotImplementedError as exc:
        if verbose:
            print(f"[resample-sparse] skipping retry: {exc}", flush=True)
        return initial_sampled, ColumnRetryStats(
            chunk_order=chunk_order,
            total_completion_columns=0,
            initial_empty_columns=0,
            retries_performed=0,
            columns_filled_by_retry=0,
            unresolved_empty_columns=0,
        )

    if chunk_order != "xyz":
        if verbose:
            print(
                f"[resample-sparse] chunk_order={chunk_order!r} not supported "
                f"for first-hole resample; returning bare initial sample.",
                flush=True,
            )
        return initial_sampled, ColumnRetryStats(
            chunk_order=chunk_order,
            total_completion_columns=0,
            initial_empty_columns=0,
            retries_performed=0,
            columns_filled_by_retry=0,
            unresolved_empty_columns=0,
        )

    # Parse prefix to get prefix_max_x (boundary between prefix and completion
    # region) and prefix-occupied (x, y) coverage.
    _prefix_rows, prefix_coords, prefix_valid = _parse_sparse_rows(
        prefix_tokens,
        tokens_per_latent=tokens_per_latent,
        num_position_tokens=num_position_tokens,
        position_vocab_size=position_vocab_size,
        feature_offset=feature_offset,
        feature_vocab_size=feature_vocab_size,
        pad_id=pad_id,
        pos_token_to_coord=pos_token_to_coord,
    )
    if prefix_valid.any():
        prefix_max_x = int(prefix_coords[prefix_valid, 0].max().item())
    else:
        prefix_max_x = -1
    completion_x_min = prefix_max_x + 1

    # Parse + sort initial-completion rows by chunk_order (xyz).
    init_compl_tokens = initial_sampled[prefix_len:]
    compl_rows, compl_coords, compl_valid = _parse_sparse_rows(
        init_compl_tokens,
        tokens_per_latent=tokens_per_latent,
        num_position_tokens=num_position_tokens,
        position_vocab_size=position_vocab_size,
        feature_offset=feature_offset,
        feature_vocab_size=feature_vocab_size,
        pad_id=pad_id,
        pos_token_to_coord=pos_token_to_coord,
    )
    if compl_valid.any():
        valid_rows = compl_rows[compl_valid]
        valid_coords = compl_coords[compl_valid]
        keys = _chunk_order_key(valid_coords, chunk_order, side)
        order = torch.argsort(keys)
        valid_rows = valid_rows.index_select(0, order)
        valid_coords = valid_coords.index_select(0, order)
        # Group rows by (x, y) -> [row_idx, ...]
        init_by_xy: dict[Tuple[int, int], List[torch.Tensor]] = {}
        for i in range(valid_rows.shape[0]):
            xy = (int(valid_coords[i, 0].item()), int(valid_coords[i, 1].item()))
            init_by_xy.setdefault(xy, []).append(valid_rows[i])
    else:
        init_by_xy = {}

    # Completion region in xyz order: x then y. accepted_rows is a list of
    # (tokens_per_latent,) tensors; we'll cat at the end.
    completion_xy_ordered: List[Tuple[int, int]] = [
        (x, y) for x in range(completion_x_min, cx) for y in range(cy)
    ]
    total_completion_cols = len(completion_xy_ordered)

    accepted_rows: List[torch.Tensor] = []
    first_hole_idx: Optional[int] = None
    for col_idx, xy in enumerate(completion_xy_ordered):
        rows_at_xy = init_by_xy.get(xy)
        if rows_at_xy:
            accepted_rows.extend(rows_at_xy)
        else:
            first_hole_idx = col_idx
            break

    initial_empty_cols = (
        total_completion_cols - first_hole_idx if first_hole_idx is not None else 0
    )

    retries_performed = 0
    columns_filled_by_retry = 0
    unresolved_empty = 0

    # If retry is disabled OR no hole was hit, keep the initial sample as-is.
    if max_retries == 0 or first_hole_idx is None:
        return initial_sampled, ColumnRetryStats(
            chunk_order=chunk_order,
            total_completion_columns=total_completion_cols,
            initial_empty_columns=initial_empty_cols,
            retries_performed=0,
            columns_filled_by_retry=0,
            unresolved_empty_columns=initial_empty_cols if max_retries == 0 else 0,
        )

    # From first_hole_idx onwards: regenerate every column. Anything in the
    # initial completion that came after the first hole has been discarded
    # (we never put it into accepted_rows).
    prefix_cpu = prefix_tokens.cpu()
    for col_idx in range(first_hole_idx, len(completion_xy_ordered)):
        tx, ty = completion_xy_ordered[col_idx]
        if accepted_rows:
            accepted_flat = torch.stack(accepted_rows, dim=0).reshape(-1)
            retry_prompt = torch.cat([prefix_cpu, accepted_flat], dim=0)
        else:
            retry_prompt = prefix_cpu

        filled = False
        for retry_idx in range(1, max_retries + 1):
            retry_temperature = _temperature_for_retry(temperature, retry_idx)
            retry_seed = _retry_seed(int(seed), col_idx, retry_idx)
            new_rows_flat = _retry_sample_sparse_column(
                gpt,
                retry_prompt,
                target_local_x=int(tx),
                target_local_y=int(ty),
                z=cz,
                temperature=retry_temperature,
                top_k=top_k,
                top_p=top_p,
                seed=retry_seed,
                pos_token_to_coord=pos_token_to_coord,
            )
            retries_performed += 1
            if new_rows_flat.numel() > 0:
                new_rows = new_rows_flat.view(-1, tokens_per_latent)
                for i in range(new_rows.shape[0]):
                    accepted_rows.append(new_rows[i])
                filled = True
                if verbose:
                    print(
                        f"[resample-sparse] col ({tx},{ty}) filled after retry "
                        f"{retry_idx} (T={retry_temperature:.3f}, "
                        f"rows={new_rows.shape[0]}).",
                        flush=True,
                    )
                break
        if filled:
            columns_filled_by_retry += 1
        else:
            unresolved_empty += 1
            if verbose:
                print(
                    f"[resample-sparse] col ({tx},{ty}) accepted empty after "
                    f"{max_retries} retries.",
                    flush=True,
                )

    # Assemble final tokens.
    if accepted_rows:
        accepted_flat = torch.stack(accepted_rows, dim=0).reshape(-1)
        final_tokens = torch.cat([prefix_cpu, accepted_flat], dim=0)
    else:
        final_tokens = prefix_cpu.clone()
    final_tokens = final_tokens.to(device=device, dtype=torch.long)

    stats = ColumnRetryStats(
        chunk_order=chunk_order,
        total_completion_columns=total_completion_cols,
        initial_empty_columns=initial_empty_cols,
        retries_performed=retries_performed,
        columns_filled_by_retry=columns_filled_by_retry,
        unresolved_empty_columns=unresolved_empty,
    )
    return final_tokens, stats
