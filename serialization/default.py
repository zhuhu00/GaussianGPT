"""
Taken from Pointcept / Point Transformer V3 (https://github.com/pointcept/pointcept)
"""

import torch

from .hilbert import encode as hilbert_encode_
from .z_order import xyz2key as z_order_encode_


@torch.inference_mode()
def encode(grid_coord, batch=None, depth=16, order="z"):
    assert order in {"z", "z-trans", "hilbert", "hilbert-trans", "xzy"}
    if order == "z":
        code = z_order_encode(grid_coord, depth=depth)
    elif order == "z-trans":
        code = z_order_encode(grid_coord[:, [1, 0, 2]], depth=depth)
    elif order == "hilbert":
        code = hilbert_encode(grid_coord, depth=depth)
    elif order == "hilbert-trans":
        code = hilbert_encode(grid_coord[:, [1, 0, 2]], depth=depth)
    elif order == "xzy":
        side = 1 << depth
        x = grid_coord[:, 0].long()
        y = grid_coord[:, 1].long()
        z = grid_coord[:, 2].long()
        code = x * (side * side) + z * side + y
    else:
        raise NotImplementedError
    if batch is not None:
        batch = batch.long()
        code = batch << depth * 3 | code
    return code


def z_order_encode(grid_coord: torch.Tensor, depth: int = 16):
    x, y, z = grid_coord[:, 0].long(), grid_coord[:, 1].long(), grid_coord[:, 2].long()
    code = z_order_encode_(x, y, z, b=None, depth=depth)
    return code


def hilbert_encode(grid_coord: torch.Tensor, depth: int = 16):
    return hilbert_encode_(grid_coord, num_dims=3, num_bits=depth)
