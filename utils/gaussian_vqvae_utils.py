import math
from typing import List, Tuple

import torch
import torch.nn.functional as F

from conf.dataclasses import ImageKeys, PCKeys


def int_cube_root(n: int) -> int:
    """Returns the largest integer x such that x^3 <= n."""
    preliminary = math.floor(n ** (1 / 3))
    while (preliminary + 1) ** 3 <= n:
        preliminary += 1
    return preliminary


def compute_loss_coords(
    out_cls: List, targets: List, device  # [(M, 1), (N, 1)]  # [(M), (N)], bool
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """
    args:
        - out_cls --> a list of me.sparse_tensors [M], consisting of coordinates and features which are classification logits, which sparse voxels should stay
        - targets --> list of bool tensors [M], specifies which of these voxels are actually close to the true point cloud
    """
    num_layers, loss_coords = len(out_cls), torch.zeros((), device=device)
    total = torch.zeros((), device=device)
    correct = torch.zeros((), device=device)
    tp_total = torch.zeros((), device=device)
    tn_total = torch.zeros((), device=device)
    fp_total = torch.zeros((), device=device)
    fn_total = torch.zeros((), device=device)
    metrics = {"accuracy": [], "false_negative_rate": [], "false_positive_rate": []}
    for out_cl, target in zip(out_cls, targets):
        logits = out_cl.F.squeeze()
        target = target.to(device=device)
        target_float = target.to(dtype=logits.dtype)
        target_bool = target.bool()

        curr_loss = F.binary_cross_entropy_with_logits(logits, target_float)
        loss_coords += curr_loss / num_layers

        pred = logits > 0.0
        tp = (pred & target_bool).sum()
        tn = (~pred & ~target_bool).sum()
        fp = (pred & ~target_bool).sum()
        fn = (~pred & target_bool).sum()

        layer_total = tp + tn + fp + fn
        layer_total_f = layer_total.to(dtype=logits.dtype)
        correct += (tp + tn).to(dtype=logits.dtype)
        total += layer_total_f
        tp_total += tp
        tn_total += tn
        fp_total += fp
        fn_total += fn

        acc = (tp + tn).to(dtype=logits.dtype) / layer_total_f
        fn_denom = (fn + tp).to(dtype=logits.dtype)
        fp_denom = (fp + tn).to(dtype=logits.dtype)
        fnr = torch.where(fn_denom > 0, fn.to(dtype=logits.dtype) / fn_denom, fn_denom)
        fpr = torch.where(fp_denom > 0, fp.to(dtype=logits.dtype) / fp_denom, fp_denom)

        metrics["accuracy"].append(acc)
        metrics["false_negative_rate"].append(fnr)
        metrics["false_positive_rate"].append(fpr)

    accuracy = correct / total
    fn_total_f = (fn_total + tp_total).to(dtype=accuracy.dtype)
    fp_total_f = (fp_total + tn_total).to(dtype=accuracy.dtype)
    metrics["false_negative_rate_total"] = torch.where(
        fn_total_f > 0, fn_total.to(dtype=accuracy.dtype) / fn_total_f, fn_total_f
    )
    metrics["false_positive_rate_total"] = torch.where(
        fp_total_f > 0, fp_total.to(dtype=accuracy.dtype) / fp_total_f, fp_total_f
    )

    return loss_coords, accuracy, metrics


def split_batch_dict(batch: dict, device=None) -> tuple[dict, dict]:
    """
    Split a batch dictionary into point cloud and image dictionaries.
    """
    point_dict = {k: v for k, v in batch.items() if k in iter(PCKeys)}
    image_dict = {k: v for k, v in batch.items() if k in iter(ImageKeys)}

    if device is not None:
        point_dict = {
            k: v.to(device)
            for k, v in point_dict.items()
            if isinstance(v, torch.Tensor)
        }
        image_dict = {
            k: v.to(device)
            for k, v in image_dict.items()
            if isinstance(v, torch.Tensor)
        }

    return point_dict, image_dict
