from __future__ import annotations

import os
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from plyfile import PlyData

from data.common import BaseSceneDataModule, BaseSceneDataset
from data.vfront import (
    focal2fov,
    safe_exists,
    safe_isdir,
    safe_listdir,
    safe_open_json,
    safe_read_lines,
)


def get_default_data_split_ase(
    data_path: str,
    train_list_path: Optional[str] = None,
    val_list_path: Optional[str] = None,
    gaussian_subpath: str = "ckpts/point_cloud_30000.ply",
    require_files: bool = True,
    skip_missing_files: bool = True,
    verbose: bool = False,
) -> Tuple[List[str], List[str]]:
    """
    Build train/val file lists for ASE.
    Scene list files are expected to contain scene IDs, one per line.
    """

    root = Path(data_path)
    if not safe_exists(root):
        raise FileNotFoundError(f"Gaussian root {data_path} does not exist.")

    all_scenes = sorted(
        name
        for name in safe_listdir(str(root))
        if safe_isdir(root / name) and not name.startswith(".")
    )
    if len(all_scenes) == 0:
        raise RuntimeError(f"No scene directories found under {data_path}.")

    def _load_list(file_path: Optional[str]) -> Optional[List[str]]:
        if not file_path:
            return None
        path = Path(file_path)
        if not safe_exists(path):
            raise FileNotFoundError(f"Scene list {file_path} does not exist.")
        scenes = [line.strip() for line in safe_read_lines(str(path)) if line.strip()]
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

    if verbose:
        print(
            f"INFO: ASE split found {len(train_scenes)} train scenes and {len(val_scenes)} val scenes."
        )

    missing_paths: List[str] = []

    def _gaussian_path(scene: str) -> Optional[str]:
        path = root / scene / gaussian_subpath
        exists = safe_exists(path)
        if require_files and not skip_missing_files and not exists:
            raise FileNotFoundError(
                f"Expected gaussian file {path} for scene '{scene}' but it was not found."
            )
        if require_files and skip_missing_files and not exists:
            missing_paths.append(str(path))
            return None
        return str(path)

    train_files: List[str] = []
    for scene in train_scenes:
        resolved = _gaussian_path(scene)
        if resolved is not None:
            train_files.append(resolved)

    val_files: List[str] = []
    for scene in val_scenes:
        resolved = _gaussian_path(scene)
        if resolved is not None:
            val_files.append(resolved)

    if skip_missing_files and missing_paths:
        unique_missing_paths = list(dict.fromkeys(missing_paths))
        print(
            f"WARNING: Skipping {len(unique_missing_paths)} missing ASE files before dataset creation:"
        )
        for path in unique_missing_paths:
            print(f"  - {path}")

    return train_files, val_files


def _load_ase_ply_gaussians(path: str) -> Dict[str, np.ndarray]:
    plydata = PlyData.read(path)

    xyz = np.stack(
        (
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"]),
        ),
        axis=1,
    )
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [
        p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")
    ]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
    if len(extra_f_names) > 0:
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        max_sh_degree = int(round(np.sqrt(len(extra_f_names) / 3 + 1) - 1))
        assert len(extra_f_names) == 3 * (max_sh_degree + 1) ** 2 - 3
        _ = features_extra.reshape(
            (features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1)
        )

    scale_names = [
        p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")
    ]
    scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [
        p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
    ]
    rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
    rots = rots / (np.linalg.norm(rots, axis=-1, keepdims=True) + 1e-8)

    opacities = np.clip(opacities[:, 0], -10.0, 10.0)

    return {
        "coords": xyz,
        "sh0": features_dc[:, :, 0],
        "opacities": opacities,
        "scales": scales,
        "quats": rots,
    }


def _read_ase_cameras_from_transforms(
    transforms_path: str,
) -> List[
    Tuple[
        np.ndarray,
        np.ndarray,
        np.float32,
        np.float32,
        int,
        int,
        np.float32,
        np.float32,
        str,
    ]
]:
    contents = safe_open_json(transforms_path)

    frames = contents.get("frames", [])
    if len(frames) == 0:
        raise RuntimeError(f"No frames found in {transforms_path}.")

    width = int(round(float(contents["width"])))
    height = int(round(float(contents["height"])))

    if "fx" in contents:
        fx = float(contents["fx"])
    elif "fl_x" in contents:
        fx = float(contents["fl_x"])
    else:
        raise RuntimeError(
            f"Missing focal length x in {transforms_path}. Expected fx or fl_x."
        )

    if "fy" in contents:
        fy = float(contents["fy"])
    elif "fl_y" in contents:
        fy = float(contents["fl_y"])
    else:
        fy = fx

    fovx = np.float32(focal2fov(fx, width))
    fovy = np.float32(focal2fov(fy, height))
    cx = np.float32(contents.get("cx", width / 2))
    cy = np.float32(contents.get("cy", height / 2))

    cams = []
    for frame in frames:
        c2w = np.array(frame["transform_matrix"], dtype=np.float64)

        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3]).astype(np.float32)
        T = w2c[:3, 3].astype(np.float32)
        image_path = str(frame.get("file_path", ""))
        cams.append((R, T, fovx, fovy, width, height, cx, cy, image_path))

    return cams


class ASEGaussianDataset(BaseSceneDataset):
    def __init__(
        self,
        paths: Sequence[str],
        split: str,
        transforms_root: Optional[str],
        transforms_filename: str = "transforms_train.json",
        **kwargs,
    ):
        # We can supervise via camera parameters only, without loading image files.
        if kwargs.get("n_images", 0) > 0 and kwargs.get("img_path") is None:
            kwargs = dict(kwargs)
            kwargs["img_path"] = "__ase_camera_only__"

        super().__init__(
            paths,
            split=split,
            **kwargs,
        )

        self.transforms_root = transforms_root
        self.transforms_filename = transforms_filename

    def load_gaussians(self, path: str) -> Dict[str, torch.Tensor]:
        data = _load_ase_ply_gaussians(path)

        return {
            "coords": torch.tensor(data["coords"], dtype=torch.float32),
            "sh0": torch.tensor(data["sh0"], dtype=torch.float32),
            "opacities": torch.tensor(data["opacities"], dtype=torch.float32),
            "scales": torch.tensor(data["scales"], dtype=torch.float32),
            "quats": torch.tensor(data["quats"], dtype=torch.float32),
        }

    def load_images(
        self, path: str, item_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        if self.n_images <= 0 or self.transforms_root is None:
            return {}

        scene_id = self.get_scene_id(path)
        transforms_path = os.path.join(
            self.transforms_root, scene_id, self.transforms_filename
        )
        if not safe_exists(transforms_path):
            raise FileNotFoundError(
                f"Transforms file {transforms_path} not found for scene {scene_id}."
            )

        cam_infos = _read_ase_cameras_from_transforms(transforms_path)
        candidate_indices = np.arange(len(cam_infos), dtype=np.int64)
        chunk_bounds_world = item_dict.get("_chunk_bounds_world")
        if chunk_bounds_world is not None:
            area_ratios = np.asarray(
                [
                    _ase_camera_chunk_projected_area_ratio(cam, chunk_bounds_world)
                    for cam in cam_infos
                ],
                dtype=np.float32,
            )
            overlap_indices = np.flatnonzero(area_ratios > 0.0).astype(np.int64)
            min_ratio = max(0.0, float(self.camera_chunk_min_area_ratio))
            if min_ratio > 0.0:
                preferred_indices = np.flatnonzero(area_ratios >= min_ratio).astype(
                    np.int64
                )
                if preferred_indices.size > 0:
                    candidate_indices = preferred_indices
                elif overlap_indices.size > 0:
                    print(
                        "INFO[chunks]: camera threshold fallback "
                        f"(scene={item_dict.get('scene_id', 'unknown')}, "
                        f"min_ratio={min_ratio:.4f}, "
                        f"overlap={overlap_indices.size}/{len(cam_infos)})."
                    )
                    candidate_indices = overlap_indices
                else:
                    print(
                        "INFO[chunks]: camera overlap fallback to all cameras "
                        f"(scene={item_dict.get('scene_id', 'unknown')}, "
                        f"min_ratio={min_ratio:.4f})."
                    )
            elif overlap_indices.size > 0:
                candidate_indices = overlap_indices

        replace = len(candidate_indices) < self.n_images
        camera_idxs = np.random.choice(
            candidate_indices,
            size=self.n_images,
            replace=replace,
        )

        Rs, Ts = [], []
        fovxs, fovys = [], []
        widths, heights = [], []
        cxs, cys = [], []
        image_paths: List[str] = []

        for cam_idx in camera_idxs:
            R, T, fovx, fovy, width, height, cx, cy, image_path = cam_infos[cam_idx]
            _, width, height, cx, cy = self._downsample_image_and_camera(
                None,
                width,
                height,
                cx,
                cy,
            )
            Rs.append(R)
            Ts.append(T)
            fovxs.append(fovx)
            fovys.append(fovy)
            widths.append(width)
            heights.append(height)
            cxs.append(cx)
            cys.append(cy)
            image_paths.append(image_path)

        return {
            "cameras_R": torch.from_numpy(np.stack(Rs)).to(torch.float32),
            "cameras_T": torch.from_numpy(np.stack(Ts)).to(torch.float32),
            "cameras_FovX": torch.from_numpy(np.asarray(fovxs, dtype=np.float32)),
            "cameras_FovY": torch.from_numpy(np.asarray(fovys, dtype=np.float32)),
            "cameras_W": torch.tensor(widths),
            "cameras_H": torch.tensor(heights),
            "cameras_cx": torch.from_numpy(np.asarray(cxs, dtype=np.float32)),
            "cameras_cy": torch.from_numpy(np.asarray(cys, dtype=np.float32)),
            "camera_idxs": torch.tensor(camera_idxs, dtype=torch.long),
            "cameras_image_path": image_paths,
        }

    def get_scene_id(self, path: str) -> str:
        p = Path(path)
        try:
            return p.parents[1].name
        except IndexError as exc:
            raise ValueError(f"Cannot infer scene id from path {path}") from exc


class ASEDataModule(BaseSceneDataModule):
    def __init__(
        self,
        *,
        data_path: str,
        transforms_path: str,
        train_list_path: Optional[str],
        val_list_path: Optional[str],
        img_path: Optional[str],
        dataloader_kwargs: Optional[Dict] = None,
        deterministic_sampling: bool = False,
        overfit_scenes: int = 0,
        overfit_epoch_size: int = 1000,
        overfit_min_val_scenes: int = 10,
        max_points: Optional[int] = None,
        max_batch_points: Optional[int] = None,
        n_images: int = 0,
        load_normals: bool = False,
        verbose: bool = False,
        augmentations: Optional[Iterable] = None,
        preload: bool = False,
        gaussian_subpath: str = "ckpts/point_cloud_30000.ply",
        transforms_filename: str = "transforms_train.json",
        background_color: str = "white",
        center_sample: bool = False,
        frustum_subsample: bool = False,
        frustum_subsample_margin: float = 0.0,
        chunk_subsample: bool = False,
        chunk_shape: Optional[Sequence[int]] = None,
        chunk_voxel_size: Optional[float] = None,
        min_chunk_occupancy: float = 0.0,
        max_chunk_attempts: int = 1,
        chunk_origin: Optional[Sequence[Optional[int]]] = None,
        camera_chunk_min_area_ratio: float = 0.0,
        image_downsample_factor: int = 1,
    ):
        super().__init__(
            data_path=data_path,
            train_list_path=train_list_path,
            val_list_path=val_list_path,
            img_path=img_path,
            overfit_scenes=overfit_scenes,
            overfit_epoch_size=overfit_epoch_size,
            overfit_min_val_scenes=overfit_min_val_scenes,
            max_points=max_points,
            max_batch_points=max_batch_points,
            n_images=n_images,
            load_normals=load_normals,
            dataloader_kwargs=dataloader_kwargs,
            deterministic_sampling=deterministic_sampling,
            verbose=verbose,
            background_color=background_color,
            center_sample=center_sample,
            frustum_subsample=frustum_subsample,
            frustum_subsample_margin=frustum_subsample_margin,
            chunk_subsample=chunk_subsample,
            chunk_shape=chunk_shape,
            chunk_voxel_size=chunk_voxel_size,
            min_chunk_occupancy=min_chunk_occupancy,
            max_chunk_attempts=max_chunk_attempts,
            chunk_origin=chunk_origin,
            camera_chunk_min_area_ratio=camera_chunk_min_area_ratio,
            image_downsample_factor=image_downsample_factor,
        )

        self.transforms_path = transforms_path
        self.transforms_filename = transforms_filename
        self.augmentations = list(augmentations or [])
        self.preload = preload
        self.gaussian_subpath = gaussian_subpath

    def build_file_lists(self) -> Tuple[List[str], List[str]]:
        return get_default_data_split_ase(
            self.data_path,
            train_list_path=self.train_list_path,
            val_list_path=self.val_list_path,
            gaussian_subpath=self.gaussian_subpath,
            skip_missing_files=True,
            verbose=self.verbose,
        )

    def get_dataset_class(self, split: str):
        return ASEGaussianDataset

    def get_dataset_kwargs(self, split: str, file_list: Sequence[str]) -> Dict:
        kwargs = super().get_dataset_kwargs(split, file_list)
        augmentations = None
        if split == "train":
            if self.n_images > 0:
                augmentations = [
                    aug
                    for aug in self.augmentations
                    if getattr(aug, "supports_images", False)
                ]
                if self.verbose and len(augmentations) < len(self.augmentations):
                    print(
                        "INFO: Skipping train augmentations that do not support camera/image-consistent transforms."
                    )
            else:
                augmentations = self.augmentations
        kwargs.update(
            transforms_root=self.transforms_path,
            transforms_filename=self.transforms_filename,
            augmentations=augmentations,
            preload=self.preload,
        )
        return kwargs


def _ase_camera_chunk_projected_area_ratio(
    cam: Tuple[
        np.ndarray,
        np.ndarray,
        np.float32,
        np.float32,
        int,
        int,
        np.float32,
        np.float32,
        str,
    ],
    chunk_bounds_world: Tuple[List[float], List[float]],
) -> float:
    R, T, fovx, fovy, width, height, cx, cy, _ = cam
    chunk_min = np.asarray(chunk_bounds_world[0], dtype=np.float32)
    chunk_max = np.asarray(chunk_bounds_world[1], dtype=np.float32)

    corners = np.asarray(
        list(
            product(
                [chunk_min[0], chunk_max[0]],
                [chunk_min[1], chunk_max[1]],
                [chunk_min[2], chunk_max[2]],
            )
        ),
        dtype=np.float32,
    )
    center = 0.5 * (chunk_min + chunk_max)

    w2c_rotation = R.T.astype(np.float32)
    translation = T.astype(np.float32)
    min_depth = 1e-4

    points_world = np.concatenate([corners, center[None, :]], axis=0)
    cam_coords = points_world @ w2c_rotation.T + translation
    z = cam_coords[:, 2]
    in_front = z > min_depth
    if not np.any(in_front):
        return 0.0

    fx = 0.5 * float(width) / np.tan(0.5 * float(fovx))
    fy = 0.5 * float(height) / np.tan(0.5 * float(fovy))
    u = fx * (cam_coords[:, 0] / z) + float(cx)
    v = fy * (cam_coords[:, 1] / z) + float(cy)
    u = u[in_front]
    v = v[in_front]

    width_f = float(width)
    height_f = float(height)
    u_min = np.clip(float(np.min(u)), 0.0, width_f)
    u_max = np.clip(float(np.max(u)), 0.0, width_f)
    v_min = np.clip(float(np.min(v)), 0.0, height_f)
    v_max = np.clip(float(np.max(v)), 0.0, height_f)
    if u_max <= u_min or v_max <= v_min:
        return 0.0

    projected_area = (u_max - u_min) * (v_max - v_min)
    image_area = max(width_f * height_f, 1e-8)
    return float(np.clip(projected_area / image_area, 0.0, 1.0))
