from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from plyfile import PlyData, PlyElement

from data.vfront import (
    np_sigmoid,
    safe_exists,
    safe_isdir,
    safe_listdir,
    safe_read_lines,
    safe_torch_load,
)


def get_default_data_split_photoshape(
    data_path: str,
    train_list_path: Optional[str] = None,
    val_list_path: Optional[str] = None,
    gaussian_subpath: str = "point_cloud/iteration_30000/point_cloud.ply",
    require_files: bool = True,
    verbose: bool = False,
) -> Tuple[List[str], List[str]]:
    """
    Build train/val scene-directory lists for Photoshape.
    Scene list files are expected to contain scene IDs, one per line.
    """

    root = Path(data_path)
    if not safe_exists(root):
        raise FileNotFoundError(f"Gaussian root {data_path} does not exist.")

    all_entries = sorted(
        name for name in safe_listdir(str(root)) if not name.startswith(".")
    )

    scene_dirs = [name for name in all_entries if safe_isdir(root / name)]
    uses_scene_directories = len(scene_dirs) > 0
    all_scenes = scene_dirs if uses_scene_directories else all_entries
    if len(all_scenes) == 0:
        raise RuntimeError(f"No scene entries found under {data_path}.")

    def _load_list(file_path: Optional[str]) -> Optional[List[str]]:
        if not file_path:
            return None
        path = Path(file_path)
        if not safe_exists(path):
            raise FileNotFoundError(f"Scene list {file_path} does not exist.")
        scenes = [line.strip() for line in safe_read_lines(str(path)) if line.strip()]
        if (
            not uses_scene_directories
            and len(all_scenes) > 0
            and all_scenes[0].endswith(".pt")
        ):
            scenes = [
                f"{scene}.pt" if not scene.endswith(".pt") else scene
                for scene in scenes
            ]
        missing = [s for s in scenes if s not in all_scenes]
        if missing:
            print(
                f"WARNING: Scene list {file_path} references scenes not present "
                f"under {data_path}: {missing}. These will be ignored."
            )
            scenes = [s for s in scenes if s in all_scenes]
        return scenes

    train_scenes = _load_list(train_list_path)
    val_scenes = _load_list(val_list_path)

    if train_scenes is None and val_scenes is None:
        split_idx = int(len(all_scenes) * 0.9)
        train_scenes = all_scenes[:split_idx] or all_scenes
        val_scenes = all_scenes[split_idx:] or train_scenes[-1:]
    elif train_scenes is None and val_scenes is not None:
        val_set = set(val_scenes)
        train_scenes = [s for s in all_scenes if s not in val_set]
    elif train_scenes is not None and val_scenes is None:
        val_start = int(len(train_scenes) * 0.9)
        val_scenes = train_scenes[val_start:] or train_scenes[-1:]
        train_scenes = train_scenes[:val_start] or val_scenes

    train_scenes = list(train_scenes or [])
    val_scenes = list(val_scenes or [])

    entry_kind = "scene directories" if uses_scene_directories else "files"
    if verbose:
        print(
            f"INFO: Photoshape split found {len(train_scenes)} train {entry_kind} and "
            f"{len(val_scenes)} val {entry_kind}."
        )

    def _scene_path(scene: str) -> str:
        scene_path = root / scene
        if require_files and uses_scene_directories:
            gaussian_path = scene_path / gaussian_subpath
            if not safe_exists(gaussian_path):
                raise FileNotFoundError(
                    f"Expected gaussian file {gaussian_path} for scene '{scene}' "
                    "but it was not found."
                )
        elif require_files and not safe_exists(scene_path):
            raise FileNotFoundError(f"Expected data file {scene_path} was not found.")
        return str(scene_path)

    train_files = [_scene_path(scene) for scene in train_scenes]
    val_files = [_scene_path(scene) for scene in val_scenes]
    return train_files, val_files


def load_inria_data(path: str, sh_degree: int = 0):
    if sh_degree not in (0, 1):
        raise ValueError(
            f"Photoshape only supports sh_degree in {{0, 1}}, got {sh_degree}."
        )

    plydata = PlyData.read(path)

    xyz = np.stack(
        (
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"]),
        ),
        axis=1,
    )
    opacities = np.asarray(plydata.elements[0]["opacity"], dtype=np.float32)

    features_dc = np.stack(
        (
            np.asarray(plydata.elements[0]["f_dc_0"], dtype=np.float32),
            np.asarray(plydata.elements[0]["f_dc_1"], dtype=np.float32),
            np.asarray(plydata.elements[0]["f_dc_2"], dtype=np.float32),
        ),
        axis=1,
    )

    scale_names = [
        p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")
    ]
    scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)), dtype=np.float32)
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [
        p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
    ]
    rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)), dtype=np.float32)
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rots = rots / (np.linalg.norm(rots, axis=-1, keepdims=True) + 1e-8)
    opacities = np.clip(opacities, -10.0, 10.0)

    means = torch.tensor(xyz, dtype=torch.float32)
    sh0 = torch.tensor(features_dc, dtype=torch.float32)
    opacities = torch.tensor(opacities, dtype=torch.float32)
    scales = torch.tensor(scales, dtype=torch.float32)
    quats = torch.tensor(rots, dtype=torch.float32)
    if sh_degree == 0:
        return means, sh0, opacities, scales, quats

    extra_f_names = [
        p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")
    ]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
    if len(extra_f_names) % 3 != 0:
        raise RuntimeError(
            f"Invalid SH feature layout in {path}: found {len(extra_f_names)} "
            "f_rest_* entries, expected a multiple of 3."
        )

    num_basis_functions = (sh_degree + 1) ** 2 - 1
    required_entries = 3 * num_basis_functions
    if len(extra_f_names) < required_entries:
        raise RuntimeError(
            f"Missing SH coefficients in {path}: need at least {required_entries} "
            f"f_rest_* entries for sh_degree={sh_degree}, found {len(extra_f_names)}."
        )

    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)), dtype=np.float32)
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])

    num_basis_available = len(extra_f_names) // 3
    sh = features_extra.reshape(xyz.shape[0], 3, num_basis_available)
    sh = sh[:, :, :num_basis_functions]
    sh = np.transpose(sh, (0, 2, 1))
    sh = sh.reshape(xyz.shape[0], -1)
    sh = np.clip(sh, -5.0, 5.0)

    return means, sh0, opacities, scales, quats, torch.tensor(sh, dtype=torch.float32)


def _as_numpy_feature(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def _as_point_feature(x, name: str, dim: Optional[int] = None) -> np.ndarray:
    arr = _as_numpy_feature(x)
    if arr.ndim == 3:
        if arr.shape[0] != 1:
            raise ValueError(
                f"Expected {name} with optional batch size 1, got shape {arr.shape}."
            )
        arr = arr[0]
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"Expected {name} with shape (N, D), got {arr.shape}.")
    if dim is not None and arr.shape[1] != dim:
        raise ValueError(f"Expected {name} with shape (N, {dim}), got {arr.shape}.")
    return arr


def _as_opacity_feature(x) -> np.ndarray:
    arr = _as_numpy_feature(x)
    if arr.ndim == 3:
        if arr.shape[0] != 1:
            raise ValueError(
                f"Expected opacities with optional batch size 1, got {arr.shape}."
            )
        arr = arr[0]
    if arr.ndim == 2:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[1] == 1:
            arr = arr[:, 0]
    if arr.ndim != 1:
        raise ValueError(
            f"Expected opacities with shape (N,), (N, 1), or batched equivalents, got {arr.shape}."
        )
    return arr


def save_inria_ply(
    output_path: str,
    coords,
    sh0,
    opacities,
    scales,
    quats,
    sh=None,
) -> int:
    xyz = _as_point_feature(coords, "coords", dim=3)
    sh0_arr = _as_point_feature(sh0, "sh0", dim=3)
    scales_arr = _as_point_feature(scales, "scales")
    quats_arr = _as_point_feature(quats, "quats")
    if quats_arr.shape[1] != 4:
        raise ValueError(f"Expected quats with shape (N, 4), got {quats_arr.shape}.")

    op_arr = _as_opacity_feature(opacities)

    n_points = xyz.shape[0]
    if sh0_arr.shape[0] != n_points:
        raise ValueError(
            f"coords/sh0 point count mismatch: {n_points} vs {sh0_arr.shape[0]}."
        )
    if op_arr.shape[0] != n_points:
        raise ValueError(
            f"coords/opacities point count mismatch: {n_points} vs {op_arr.shape[0]}."
        )
    if scales_arr.shape[0] != n_points:
        raise ValueError(
            f"coords/scales point count mismatch: {n_points} vs {scales_arr.shape[0]}."
        )
    if quats_arr.shape[0] != n_points:
        raise ValueError(
            f"coords/quats point count mismatch: {n_points} vs {quats_arr.shape[0]}."
        )

    rest_arr = None
    if sh is not None:
        sh_arr = _as_point_feature(sh, "sh")
        if sh_arr.shape[0] != n_points:
            raise ValueError(
                f"coords/sh point count mismatch: {n_points} vs {sh_arr.shape[0]}."
            )
        if sh_arr.shape[1] % 3 != 0:
            raise ValueError(
                f"Expected sh with channel count divisible by 3, got {sh_arr.shape[1]}."
            )
        n_basis = sh_arr.shape[1] // 3
        rest_arr = np.transpose(
            sh_arr.reshape(n_points, n_basis, 3), (0, 2, 1)
        ).reshape(n_points, -1)

    dtype_fields = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("f_dc_0", "f4"),
        ("f_dc_1", "f4"),
        ("f_dc_2", "f4"),
        ("opacity", "f4"),
    ]
    dtype_fields.extend((f"scale_{i}", "f4") for i in range(scales_arr.shape[1]))
    dtype_fields.extend((f"rot_{i}", "f4") for i in range(quats_arr.shape[1]))
    if rest_arr is not None:
        dtype_fields.extend((f"f_rest_{i}", "f4") for i in range(rest_arr.shape[1]))

    vertices = np.empty(n_points, dtype=dtype_fields)
    vertices["x"] = xyz[:, 0]
    vertices["y"] = xyz[:, 1]
    vertices["z"] = xyz[:, 2]
    vertices["f_dc_0"] = sh0_arr[:, 0]
    vertices["f_dc_1"] = sh0_arr[:, 1]
    vertices["f_dc_2"] = sh0_arr[:, 2]
    vertices["opacity"] = op_arr

    for i in range(scales_arr.shape[1]):
        vertices[f"scale_{i}"] = scales_arr[:, i]
    for i in range(quats_arr.shape[1]):
        vertices[f"rot_{i}"] = quats_arr[:, i]
    if rest_arr is not None:
        for i in range(rest_arr.shape[1]):
            vertices[f"f_rest_{i}"] = rest_arr[:, i]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ply_element = PlyElement.describe(vertices, "vertex")
    PlyData([ply_element], text=False).write(str(output))
    return int(n_points)


def convert_gaussian_pt_to_inria_ply(
    input_path: str,
    output_path: str,
    voxel_size: float = 0.025,
    max_height_quantile: Optional[float] = None,
) -> int:
    payload = safe_torch_load(input_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(
            f"Expected dict payload in {input_path}, got {type(payload).__name__}."
        )
    if max_height_quantile is not None and not (0.0 <= max_height_quantile <= 1.0):
        raise ValueError(
            f"max_height_quantile must be in [0, 1], got {max_height_quantile}."
        )

    scene_keys = {"coords", "sh0", "opacities", "scales", "quats"}
    sparse_keys = {"anchor", "offset", "opacity", "scale", "rotation", "f_dc"}
    if scene_keys.issubset(payload.keys()):
        coords = _as_point_feature(payload["coords"], "coords", dim=3)
        sh0 = _as_point_feature(payload["sh0"], "sh0", dim=3)
        opacities = _as_opacity_feature(payload["opacities"])
        scales = _as_point_feature(payload["scales"], "scales")
        quats = _as_point_feature(payload["quats"], "quats", dim=4)
        sh = payload.get("sh")
        if sh is not None:
            sh = _as_point_feature(sh, "sh")

        if max_height_quantile is not None:
            threshold = float(np.quantile(coords[:, 2], max_height_quantile))
            keep = coords[:, 2] <= threshold
            coords = coords[keep]
            sh0 = sh0[keep]
            opacities = opacities[keep]
            scales = scales[keep]
            quats = quats[keep]
            if sh is not None:
                sh = sh[keep]

        return save_inria_ply(
            output_path=output_path,
            coords=coords,
            sh0=sh0,
            opacities=opacities,
            scales=scales,
            quats=quats,
            sh=sh,
        )

    if sparse_keys.issubset(payload.keys()):
        anchor = _as_point_feature(payload["anchor"], "anchor", dim=3)
        offset = _as_point_feature(payload["offset"], "offset", dim=3)
        coords = anchor + (1.5 * float(voxel_size) * np.tanh(offset))
        scales = (
            2.0
            * float(voxel_size)
            * np_sigmoid(_as_point_feature(payload["scale"], "scale"))
        )
        opacities = _as_opacity_feature(payload["opacity"])
        sh = payload.get("f_rest")
        if sh is not None:
            sh = _as_point_feature(sh, "f_rest")
            if sh.shape[1] % 3 != 0:
                raise ValueError(
                    f"Expected f_rest with channel count divisible by 3, got {sh.shape[1]}."
                )
            n_basis = sh.shape[1] // 3
            sh = np.transpose(sh.reshape(sh.shape[0], 3, n_basis), (0, 2, 1)).reshape(
                sh.shape[0], -1
            )

        if max_height_quantile is not None:
            threshold = float(np.quantile(coords[:, 2], max_height_quantile))
            keep = coords[:, 2] <= threshold
            coords = coords[keep]
            opacities = opacities[keep]
            scales = scales[keep]
            sh = None if sh is None else sh[keep]
            f_dc = _as_point_feature(payload["f_dc"], "f_dc", dim=3)[keep]
            rotation = _as_point_feature(payload["rotation"], "rotation", dim=4)[keep]
        else:
            f_dc = _as_point_feature(payload["f_dc"], "f_dc", dim=3)
            rotation = _as_point_feature(payload["rotation"], "rotation", dim=4)
        return save_inria_ply(
            output_path=output_path,
            coords=coords,
            sh0=f_dc,
            opacities=opacities,
            scales=np.log(scales.clip(min=1e-10)),
            quats=rotation,
            sh=sh,
        )

    expected_scene = ", ".join(sorted(scene_keys))
    expected_sparse = ", ".join(sorted(sparse_keys))
    raise ValueError(
        f"{input_path} is not a supported Gaussian payload. Expected either scene keys "
        f"({expected_scene}) or sparse checkpoint keys ({expected_sparse})."
    )
