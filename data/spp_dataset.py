from __future__ import annotations

import os
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from data.ase_dataset import _load_ase_ply_gaussians
from data.common import BaseSceneDataModule, BaseSceneDataset
from data.vfront import (
    focal2fov,
    safe_cv2_imread,
    safe_exists,
    safe_isdir,
    safe_isfile,
    safe_listdir,
    safe_open_json,
    safe_read_lines,
)


def get_default_data_split_spp(
    data_path: str,
    train_list_path: Optional[str] = None,
    val_list_path: Optional[str] = None,
    gaussian_subpath: str = "ckpts/point_cloud_30000.ply",
    require_files: bool = True,
    verbose: bool = False,
) -> Tuple[List[str], List[str]]:
    """
    Build train/val file lists for ScanNet++.
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
            "INFO: ScanNet++ split found "
            f"{len(train_scenes)} train scenes and {len(val_scenes)} val scenes."
        )

    def _gaussian_path(scene: str) -> str:
        path = root / scene / gaussian_subpath
        if require_files and not safe_exists(path):
            raise FileNotFoundError(
                f"Expected gaussian file {path} for scene '{scene}' but it was not found."
            )
        return str(path)

    train_files = [_gaussian_path(scene) for scene in train_scenes]
    val_files = [_gaussian_path(scene) for scene in val_scenes]
    return train_files, val_files


def _qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    q = np.asarray(qvec, dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-12)
    qw, qx, qy, qz = q
    return np.asarray(
        [
            [
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy - qz * qw),
                2.0 * (qx * qz + qy * qw),
            ],
            [
                2.0 * (qx * qy + qz * qw),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz - qx * qw),
            ],
            [
                2.0 * (qx * qz - qy * qw),
                2.0 * (qy * qz + qx * qw),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ],
        ],
        dtype=np.float32,
    )


def _read_colmap_extrinsics(
    colmap_images_path: str,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    lines = safe_read_lines(colmap_images_path, encoding="utf-8")
    extrinsics: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < 10:
            continue

        try:
            int(parts[0])
            qvec = np.asarray([float(v) for v in parts[1:5]], dtype=np.float64)
            tvec = np.asarray([float(v) for v in parts[5:8]], dtype=np.float32)
            int(parts[8])
        except ValueError:
            continue

        image_name = parts[9]
        rotation_w2c = _qvec_to_rotmat(qvec)
        extrinsics[image_name] = (rotation_w2c, tvec)
        extrinsics[Path(image_name).name] = (rotation_w2c, tvec)

    if len(extrinsics) == 0:
        raise RuntimeError(f"No COLMAP extrinsics parsed from {colmap_images_path}.")
    return extrinsics


def _resolve_spp_image_path(images_dir: str, file_path: str) -> str:
    raw = os.path.join(images_dir, file_path)
    if safe_isfile(raw):
        return raw

    stem_name = Path(file_path).name
    candidates = [
        os.path.join(images_dir, f"{file_path}.JPG"),
        os.path.join(images_dir, f"{file_path}.jpg"),
        os.path.join(images_dir, f"{file_path}.PNG"),
        os.path.join(images_dir, f"{file_path}.png"),
        os.path.join(images_dir, stem_name),
    ]
    for candidate in candidates:
        if safe_isfile(candidate):
            return candidate

    raise FileNotFoundError(
        f"Could not resolve image '{file_path}' in directory {images_dir}."
    )


def _read_spp_cameras(
    transforms_path: str,
    colmap_images_path: str,
    images_dir: str,
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
    frames = list(contents.get("frames", [])) + list(contents.get("test_frames", []))
    if len(frames) == 0:
        raise RuntimeError(f"No frames found in {transforms_path}.")

    width = int(round(float(contents.get("w", contents.get("width")))))
    height = int(round(float(contents.get("h", contents.get("height")))))

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

    colmap_extrinsics = _read_colmap_extrinsics(colmap_images_path)

    cams = []
    seen_paths: set[str] = set()
    for frame in frames:
        file_path = str(frame.get("file_path", ""))
        if file_path == "" or file_path in seen_paths:
            continue
        seen_paths.add(file_path)

        image_name = Path(file_path).name
        if image_name not in colmap_extrinsics:
            raise KeyError(
                f"COLMAP extrinsics missing for frame '{image_name}' "
                f"(from {colmap_images_path})."
            )
        rotation_w2c, translation_w2c = colmap_extrinsics[image_name]
        try:
            image_path = _resolve_spp_image_path(images_dir, file_path)
        except FileNotFoundError as exc:
            print(
                "WARNING[spp]: skipping camera frame with missing image "
                f"(frame={file_path}, scene_transforms={transforms_path}): {exc}"
            )
            continue

        # Keep R transposed to match the rest of this codebase's camera convention.
        R = rotation_w2c.T.astype(np.float32)
        T = translation_w2c.astype(np.float32)
        cams.append((R, T, fovx, fovy, width, height, cx, cy, image_path))

    if len(cams) == 0:
        raise RuntimeError(
            f"No valid camera entries produced from {transforms_path} and {colmap_images_path}."
        )
    return cams


class SPPGaussianDataset(BaseSceneDataset):
    def __init__(
        self,
        paths: Sequence[str],
        split: str,
        transforms_root: Optional[str],
        transforms_rel_path: str = "dslr/nerfstudio/transforms_undistorted.json",
        colmap_images_rel_path: str = "dslr/colmap/images.txt",
        images_rel_path: str = "dslr/resized_undistorted_images",
        **kwargs,
    ):
        super().__init__(
            paths,
            split=split,
            **kwargs,
        )

        self.transforms_root = transforms_root
        self.transforms_rel_path = transforms_rel_path
        self.colmap_images_rel_path = colmap_images_rel_path
        self.images_rel_path = images_rel_path

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
        if self.img_path is None:
            raise ValueError("ScanNet++ requires `img_path` when loading images.")

        scene_id = self.get_scene_id(path)
        transforms_path = os.path.join(
            self.transforms_root, scene_id, self.transforms_rel_path
        )
        colmap_images_path = os.path.join(
            self.transforms_root, scene_id, self.colmap_images_rel_path
        )
        images_dir = os.path.join(self.img_path, scene_id, self.images_rel_path)

        if not safe_exists(transforms_path):
            raise FileNotFoundError(
                f"Transforms file {transforms_path} not found for scene {scene_id}."
            )
        if not safe_exists(colmap_images_path):
            raise FileNotFoundError(
                f"COLMAP images file {colmap_images_path} not found for scene {scene_id}."
            )
        if not safe_isdir(images_dir):
            raise FileNotFoundError(
                f"Image directory {images_dir} not found for scene {scene_id}."
            )

        cam_infos = _read_spp_cameras(transforms_path, colmap_images_path, images_dir)
        candidate_indices = np.arange(len(cam_infos), dtype=np.int64)

        chunk_bounds_world = item_dict.get("_chunk_bounds_world")
        if chunk_bounds_world is not None:
            area_ratios = np.asarray(
                [
                    _spp_camera_chunk_projected_area_ratio(cam, chunk_bounds_world)
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

        images = []
        Rs, Ts = [], []
        fovxs, fovys = [], []
        widths, heights = [], []
        cxs, cys = [], []
        image_paths: List[str] = []
        selected_camera_idxs: List[int] = []
        bad_camera_indices: set[int] = set()

        def _try_append_camera(cam_idx: int) -> bool:
            R, T, fovx, fovy, width, height, cx, cy, image_path = cam_infos[cam_idx]
            try:
                frame = torch.from_numpy(
                    _read_image(image_path, self._background_rgb)
                ).float()
            except Exception as exc:
                print(
                    "WARNING[spp]: failed to load image, resampling camera "
                    f"(scene={scene_id}, cam_idx={cam_idx}, path={image_path}): {exc}"
                )
                bad_camera_indices.add(int(cam_idx))
                return False

            frame, width, height, cx, cy = self._downsample_image_and_camera(
                frame,
                width,
                height,
                cx,
                cy,
            )
            images.append(frame)
            Rs.append(R)
            Ts.append(T)
            fovxs.append(fovx)
            fovys.append(fovy)
            widths.append(width)
            heights.append(height)
            cxs.append(cx)
            cys.append(cy)
            image_paths.append(image_path)
            selected_camera_idxs.append(int(cam_idx))
            return True

        replace = len(candidate_indices) < self.n_images
        initial_camera_idxs = np.random.choice(
            candidate_indices,
            size=self.n_images,
            replace=replace,
        )
        for cam_idx in initial_camera_idxs:
            _try_append_camera(int(cam_idx))

        while len(images) < self.n_images:
            remaining = np.asarray(
                [
                    idx
                    for idx in candidate_indices
                    if int(idx) not in bad_camera_indices
                ],
                dtype=np.int64,
            )
            if remaining.size == 0:
                break
            unused = np.asarray(
                [idx for idx in remaining if int(idx) not in selected_camera_idxs],
                dtype=np.int64,
            )
            pool = unused if unused.size > 0 else remaining
            next_cam_idx = int(np.random.choice(pool, size=1, replace=False)[0])
            _try_append_camera(next_cam_idx)

        if len(images) == 0:
            raise RuntimeError(
                "Failed to load any SPP images for scene "
                f"{scene_id} after trying {len(candidate_indices)} cameras."
            )

        if len(images) < self.n_images:
            pad_count = self.n_images - len(images)
            pad_indices = np.random.choice(
                np.arange(len(images), dtype=np.int64),
                size=pad_count,
                replace=True,
            )
            for source_idx in pad_indices:
                images.append(images[int(source_idx)].clone())
                Rs.append(Rs[int(source_idx)])
                Ts.append(Ts[int(source_idx)])
                fovxs.append(fovxs[int(source_idx)])
                fovys.append(fovys[int(source_idx)])
                widths.append(widths[int(source_idx)])
                heights.append(heights[int(source_idx)])
                cxs.append(cxs[int(source_idx)])
                cys.append(cys[int(source_idx)])
                image_paths.append(image_paths[int(source_idx)])
                selected_camera_idxs.append(selected_camera_idxs[int(source_idx)])

        return {
            "images": torch.stack(images, dim=0),
            "cameras_R": torch.from_numpy(np.stack(Rs)).to(torch.float32),
            "cameras_T": torch.from_numpy(np.stack(Ts)).to(torch.float32),
            "cameras_FovX": torch.from_numpy(np.asarray(fovxs, dtype=np.float32)),
            "cameras_FovY": torch.from_numpy(np.asarray(fovys, dtype=np.float32)),
            "cameras_W": torch.tensor(widths),
            "cameras_H": torch.tensor(heights),
            "cameras_cx": torch.from_numpy(np.asarray(cxs, dtype=np.float32)),
            "cameras_cy": torch.from_numpy(np.asarray(cys, dtype=np.float32)),
            "camera_idxs": torch.tensor(selected_camera_idxs, dtype=torch.long),
            "cameras_image_path": image_paths,
        }

    def get_scene_id(self, path: str) -> str:
        p = Path(path)
        try:
            return p.parents[1].name
        except IndexError as exc:
            raise ValueError(f"Cannot infer scene id from path {path}") from exc


def _read_image(
    img_path: str, background_rgb: Tuple[float, float, float]
) -> np.ndarray:
    import cv2
    from skimage import color

    bgra = safe_cv2_imread(img_path, cv2.IMREAD_UNCHANGED)
    if bgra.shape[2] == 4:
        rgba = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA).astype(np.float32) / 255.0
        rgb = color.rgba2rgb(rgba, background=background_rgb)
    else:
        rgb = cv2.cvtColor(bgra, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.moveaxis(rgb, -1, 0)


class SPPDataModule(BaseSceneDataModule):
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
        if img_path is None:
            raise ValueError("ScanNet++ requires img_path to be set.")

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
        self.augmentations = list(augmentations or [])
        self.preload = preload
        self.gaussian_subpath = gaussian_subpath

    def build_file_lists(self) -> Tuple[List[str], List[str]]:
        return get_default_data_split_spp(
            self.data_path,
            train_list_path=self.train_list_path,
            val_list_path=self.val_list_path,
            gaussian_subpath=self.gaussian_subpath,
            verbose=self.verbose,
        )

    def get_dataset_class(self, split: str):
        return SPPGaussianDataset

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
            augmentations=augmentations,
            preload=self.preload,
        )
        return kwargs


def _spp_camera_chunk_projected_area_ratio(
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
