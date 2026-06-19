import torch

from serialization import encode


def threed_to_oned_indices(indices_3d: torch.Tensor, side_length: int) -> torch.Tensor:
    """Convert (..., 3) indices to (...,) indices."""
    return (
        indices_3d[..., 0] * (side_length**2)
        + indices_3d[..., 1] * side_length
        + indices_3d[..., 2]
    )


def oned_to_threed_indices(indices_1d: torch.Tensor, side_length: int) -> torch.Tensor:
    """Convert (...,) indices to (..., 3) indices."""
    z = indices_1d % side_length
    y = (indices_1d // side_length) % side_length
    x = (indices_1d // (side_length**2)) % side_length
    return torch.stack([x, y, z], dim=-1)


def coords_to_pos_tokens(
    centered_coords: torch.Tensor,
    num_position_tokens: int,
    base_side_length: int,
) -> torch.Tensor:
    pos_idxs = torch.zeros(
        (centered_coords.shape[0], num_position_tokens),
        dtype=torch.long,
        device=centered_coords.device,
    )
    residual = centered_coords.clone()
    for t in range(num_position_tokens):
        div = base_side_length ** (num_position_tokens - t - 1)
        curr_coords = residual // div
        pos_idxs[:, t] = threed_to_oned_indices(
            curr_coords % base_side_length, base_side_length
        )
        residual = residual - (curr_coords * div)
    return pos_idxs


def pos_tokens_to_centered_coords(
    pos_idxs: torch.Tensor, num_position_tokens: int, base_side_length: int
) -> torch.Tensor:
    coords = torch.zeros(
        (pos_idxs.shape[0], 3), dtype=torch.float32, device=pos_idxs.device
    )
    for t in range(num_position_tokens):
        div = base_side_length ** (num_position_tokens - t - 1)
        curr_coords = oned_to_threed_indices(pos_idxs[:, t], base_side_length)
        coords += curr_coords * div
    return coords


def dense_chunk_coords(
    chunk_shape: list[int] | tuple[int, int, int],
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.long,
) -> torch.Tensor:
    """
    Return dense grid coords in row-major xyz order.
    """
    if len(chunk_shape) != 3:
        raise ValueError("chunk_shape must have 3 elements.")
    xs = torch.arange(chunk_shape[0], device=device, dtype=dtype)
    ys = torch.arange(chunk_shape[1], device=device, dtype=dtype)
    zs = torch.arange(chunk_shape[2], device=device, dtype=dtype)
    grid_x, grid_y, grid_z = torch.meshgrid(xs, ys, zs, indexing="ij")
    return torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)


def dense_chunk_order_indices(
    chunk_shape: list[int] | tuple[int, int, int],
    chunk_order: str,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    if len(chunk_shape) != 3:
        raise ValueError("chunk_shape must have 3 elements.")
    if chunk_order == "xyz":
        num_voxels = int(chunk_shape[0] * chunk_shape[1] * chunk_shape[2])
        return torch.arange(num_voxels, device=device, dtype=torch.long)
    if chunk_order == "xzy":
        num_voxels = int(chunk_shape[0] * chunk_shape[1] * chunk_shape[2])
        order = torch.arange(num_voxels, device=device, dtype=torch.long)
        order = order.view(chunk_shape[0], chunk_shape[1], chunk_shape[2])
        return order.permute(0, 2, 1).reshape(-1)
    if chunk_order not in {"z", "z-trans", "hilbert", "hilbert-trans"}:
        raise ValueError(
            "Unsupported chunk_order. Expected 'xyz', 'xzy', 'z', 'z-trans', "
            "'hilbert', or 'hilbert-trans'."
        )
    coords = dense_chunk_coords(chunk_shape, device="cpu", dtype=torch.long)
    depth = int(max(chunk_shape) - 1).bit_length()
    codes = encode(coords, depth=depth, order=chunk_order)
    order = torch.argsort(codes.reshape(-1))
    if device is not None:
        order = order.to(device)
    return order


def dense_chunk_inverse_order_indices(
    chunk_shape: list[int] | tuple[int, int, int],
    chunk_order: str,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    order = dense_chunk_order_indices(chunk_shape, chunk_order, device=device)
    inv = torch.empty_like(order)
    inv[order] = torch.arange(order.numel(), device=order.device)
    return inv


def dense_chunk_token_positions(
    chunk_shape: list[int] | tuple[int, int, int],
    num_features: int,
    chunk_order: str = "xyz",
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.long,
) -> torch.Tensor:
    coords = dense_chunk_coords(chunk_shape, device=device, dtype=dtype)
    order = dense_chunk_order_indices(chunk_shape, chunk_order, device=coords.device)
    if order.numel():
        coords = coords[order]
    if num_features < 1:
        raise ValueError("num_features must be >= 1.")
    if num_features == 1:
        return coords
    return coords.repeat_interleave(int(num_features), dim=0)
