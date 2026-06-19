from typing import Dict, Tuple

import numpy as np


def sparsify(
    points: np.array,
    g_data: Dict,
    voxel_size: float,
    level: int,
    sh_degree: int,
    min_voxel: int,
    stride: int = 4,
) -> Tuple[np.array, Dict, np.array]:
    """
    returns:
        sparsifide points [M, 3]
        sparsified feauture dict
        mask [M]
    """

    N = points.shape[0]

    off = g_data["offset"]  # [N, 3]
    opa = g_data["opacity"]  # [N, 1]
    sca = g_data["scale"]  # [N, 3]
    color = g_data["rgb"]  # [N, 3]
    rot = g_data["rotation"]  # [N, 4]

    features = np.concatenate(((points + off), opa, sca, color, rot), axis=-1)
    if sh_degree > 0:
        sh = g_data["sh"]  # [N, D]
        features = np.concatenate((features, sh), axis=-1)

    xyz_voxel_all = points / voxel_size
    xyz_voxel_all_int = np.round(xyz_voxel_all).astype(np.int32)
    xyz_voxel_all_int -= min_voxel
    xyz_voxel_int = xyz_voxel_all_int // (stride**level)

    unique_voxels, inv = np.unique(
        xyz_voxel_int, axis=0, return_inverse=True
    )  # [M, 3], [N]

    sparsified_feats = []
    for i in range(unique_voxels.shape[0]):
        all_voxel_feats = features[inv == i]  # [K, D]
        max_opa_idx = all_voxel_feats[:, 3].argmax()  # int
        sparsified_feats.append(all_voxel_feats[max_opa_idx])  # [D]

    sparsified_points = (
        unique_voxels.astype(np.float32) * stride**level
        + min_voxel
        + (stride**level - 1) / 2
    ) * voxel_size  # [M, 3]
    sparsified_feats = np.stack(sparsified_feats, axis=0)  # [M, D]
    sparsified_feats[:, :3] -= sparsified_points

    out_g_data = {
        "offset": sparsified_feats[:, 0:3],
        "opacity": sparsified_feats[:, 3:4],
        "scale": sparsified_feats[:, 4:7],
        "rgb": sparsified_feats[:, 7:10],
        "rotation": sparsified_feats[:, 10:14],
    }
    if sh_degree > 0:
        out_g_data["sh"] = sparsified_feats[:, 14:]

    return sparsified_points, out_g_data
