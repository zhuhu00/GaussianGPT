from __future__ import annotations

import os
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from data.common import BaseSceneDataModule, BaseSceneDataset, PreprocessedDataset
from data.vfront import (
    load_pth_sparse_gaussian,
    readCamerasFromTransforms,
    safe_cv2_imread,
    safe_exists,
    safe_isdir,
    safe_isfile,
    safe_listdir,
    safe_read_lines,
)

CHUNK_INSIDE_EPS = 1e-3


def get_default_data_split_vfront(
    data_path: str,
    train_list_path: Optional[str] = None,
    val_list_path: Optional[str] = None,
    gaussian_subpath: str = "v0.025_sigmoid_uniform_tanh/point_cloud/iteration_30000/ckpt.pth",
    require_files: bool = True,
    skip_missing_files: bool = True,
    allow_augmented_files: bool = False,
    verbose: bool = False,
) -> Tuple[List[str], List[str]]:
    """
    Build train/val file lists for the VFront dataset.
    """

    root = Path(data_path)
    if not safe_exists(root):
        raise FileNotFoundError(f"Gaussian root {data_path} does not exist.")

    all_entries = sorted(
        name for name in safe_listdir(str(root)) if not name.startswith(".")
    )
    scene_dirs = [name for name in all_entries if safe_isdir(root / name)]
    uses_scene_directories = len(scene_dirs) > 0
    if uses_scene_directories:
        all_scenes = scene_dirs
    else:
        pt_entries = [name for name in all_entries if name.endswith(".pt")]
        all_scenes = pt_entries if len(pt_entries) > 0 else all_entries
    scenes_are_pt_files = (
        not uses_scene_directories
        and len(all_scenes) > 0
        and all(scene.endswith(".pt") for scene in all_scenes)
    )
    if len(all_scenes) == 0:
        raise RuntimeError(f"No scene entries found under {data_path}.")

    def _load_list(file_path: Optional[str]) -> Optional[List[str]]:
        if not file_path:
            return None
        path = Path(file_path)
        if not safe_exists(path):
            raise FileNotFoundError(f"Scene list {file_path} does not exist.")
        scenes = [line.strip() for line in safe_read_lines(str(path)) if line.strip()]
        if scenes_are_pt_files:
            scenes = [
                f"{scene}.pt" if not scene.endswith(".pt") else scene
                for scene in scenes
            ]
        missing = [s for s in scenes if s not in all_scenes]
        if missing:
            # warn and filter
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
        entry_kind = "scene directories" if uses_scene_directories else "files"
        print(
            f"INFO: VFront split found {len(train_scenes)} train {entry_kind} and "
            f"{len(val_scenes)} val {entry_kind}."
        )

    missing_paths: List[str] = []

    def _has_augmented_variant(path: Path) -> bool:
        if not allow_augmented_files:
            return False
        parent = path.parent
        if not safe_exists(parent):
            return False
        prefix = f"{path.stem}_rot"
        suffix = path.suffix
        return any(
            name.startswith(prefix) and name.endswith(suffix)
            for name in safe_listdir(str(parent))
        )

    def _gaussian_path(scene: str) -> Optional[str]:
        path = root / scene
        if uses_scene_directories:
            path = path / gaussian_subpath
            exists = safe_exists(path)
            if not exists and allow_augmented_files:
                exists = _has_augmented_variant(path)
            should_check_existence = require_files or allow_augmented_files
            if should_check_existence and not skip_missing_files and not exists:
                raise FileNotFoundError(
                    f"Expected gaussian file {path} for scene '{scene}' but it was not found."
                )
            if should_check_existence and skip_missing_files and not exists:
                missing_paths.append(str(path))
                return None
        else:
            exists = safe_exists(path)
            if not exists and allow_augmented_files:
                exists = _has_augmented_variant(path)
            should_check_existence = require_files or allow_augmented_files
            if should_check_existence and not skip_missing_files and not exists:
                raise FileNotFoundError(f"Expected data file {path} was not found.")
            if should_check_existence and skip_missing_files and not exists:
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
            f"WARNING: Skipping {len(unique_missing_paths)} missing VFront files before dataset creation:"
        )
        for path in unique_missing_paths:
            print(f"  - {path}")

    return train_files, val_files


class VFrontGaussianDataset(BaseSceneDataset):
    def __init__(
        self,
        paths: Sequence[str],
        split: str,
        transforms_root: Optional[str],
        sh_degree: int = 0,
        voxel_size: float = 0.025,
        depth_enabled: bool = False,
        depth_subdir: str = "depth",
        depth_extension: str = ".exr",
        depth_chunk_mask_enabled: bool = False,
        depth_camera_sampling_enabled: bool = False,
        depth_camera_sampling_probe_multiplier: int = 3,
        depth_camera_sampling_stride: int = 8,
        **kwargs,
    ):
        super().__init__(
            paths,
            split=split,
            **kwargs,
        )

        self.transforms_root = transforms_root
        self.sh_degree = sh_degree
        self.voxel_size = voxel_size
        self.depth_enabled = bool(depth_enabled)
        self.depth_subdir = str(depth_subdir)
        self.depth_extension = str(depth_extension)
        if not self.depth_extension.startswith("."):
            self.depth_extension = f".{self.depth_extension}"
        self.depth_chunk_mask_enabled = bool(depth_chunk_mask_enabled)
        if self.depth_chunk_mask_enabled and not self.depth_enabled:
            raise ValueError(
                "`depth_chunk_mask_enabled` requires `depth_enabled=True`."
            )
        self.depth_camera_sampling_enabled = bool(depth_camera_sampling_enabled)
        self.depth_camera_sampling_probe_multiplier = max(
            1, int(depth_camera_sampling_probe_multiplier)
        )
        self.depth_camera_sampling_stride = max(1, int(depth_camera_sampling_stride))

    def load_gaussians(self, path: str) -> Dict[str, torch.Tensor]:
        data = load_pth_sparse_gaussian(path, self.sh_degree, self.voxel_size)

        anchor = torch.tensor(data["anchor"], dtype=torch.float32)
        offset = torch.tensor(data["offset"], dtype=torch.float32)
        scales = torch.tensor(data["scale"], dtype=torch.float32)

        # Keep dataset outputs in a canonical space expected by internal representations:
        # - coords in world space
        # - scales in log-space
        scales = torch.log(scales.clamp_min(1e-10))

        coords = anchor + offset

        sample: Dict[str, torch.Tensor] = {
            "coords": coords,
            # "anchor": anchor,
            # "coord_offset": offset,
            "sh0": torch.tensor(data["rgb"], dtype=torch.float32),
            "opacities": torch.tensor(data["opacity"], dtype=torch.float32).squeeze(-1),
            "scales": scales,
            "quats": torch.tensor(data["rotation"], dtype=torch.float32),
        }

        if "sh" in data:
            sample["sh"] = torch.tensor(data["sh"], dtype=torch.float32)

        return sample

    def load_images(
        self, path: str, item_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        if self.n_images <= 0 or self.img_path is None or self.transforms_root is None:
            return {}

        scene_id = self.get_scene_id(path)
        transforms_dir = os.path.join(self.transforms_root, scene_id)
        if not safe_isdir(transforms_dir):
            raise FileNotFoundError(
                f"Transforms directory {transforms_dir} not found for scene {scene_id}."
            )

        images_path = super()._resolve_image_path(path)
        json_candidates = [
            name
            for name in (
                "transforms.json",
                "transforms_train.json",
                "transforms_test.json",
            )
            if safe_isfile(os.path.join(transforms_dir, name))
        ]
        if not json_candidates:
            json_candidates = sorted(
                name
                for name in safe_listdir(transforms_dir)
                if name.startswith("transforms") and name.endswith(".json")
            )

        cam_infos = []
        for json_name in json_candidates:
            cam_infos.extend(
                readCamerasFromTransforms(
                    transforms_dir, json_name, images_path, extension=".png"
                )
            )

        if len(cam_infos) == 0:
            raise RuntimeError(f"No camera information found in {transforms_dir}.")

        chunk_bounds_world = item_dict.get("_chunk_bounds_world")
        depth_cache_by_cam_idx: Dict[int, torch.Tensor] = {}
        if self.depth_camera_sampling_enabled:
            all_indices = np.arange(len(cam_infos), dtype=np.int64)
            if (
                self.depth_enabled
                and self.depth_chunk_mask_enabled
                and chunk_bounds_world is not None
            ):
                camera_idxs, depth_cache_by_cam_idx = (
                    self._sample_camera_indices_depth_chunk_stats(
                        cam_infos=cam_infos,
                        images_path=images_path,
                        chunk_bounds_world=chunk_bounds_world,
                    )
                )
            else:
                replace = len(all_indices) < self.n_images
                camera_idxs = np.random.choice(
                    all_indices,
                    size=self.n_images,
                    replace=replace,
                )
        else:
            candidate_indices = np.arange(len(cam_infos), dtype=np.int64)
            if chunk_bounds_world is not None:
                area_ratios = np.asarray(
                    [
                        _camera_chunk_projected_area_ratio(cam, chunk_bounds_world)
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

        images = []
        Rs, Ts = [], []
        fovxs, fovys = [], []
        widths, heights = [], []
        cxs, cys = [], []
        image_paths: List[str] = []
        depths: List[torch.Tensor] = []
        loss_masks: List[torch.Tensor] = []

        for cam_idx in camera_idxs:
            cam = cam_infos[cam_idx]
            frame = torch.from_numpy(
                _read_image(cam.image_path, self._background_rgb)
            ).float()
            frame, width, height, cx, cy = self._downsample_image_and_camera(
                frame,
                cam.width,
                cam.height,
                cam.cx,
                cam.cy,
            )
            images.append(frame)
            Rs.append(cam.R.astype(np.float32))
            Ts.append(cam.T.astype(np.float32))
            fovxs.append(np.float32(cam.FovX))
            fovys.append(np.float32(cam.FovY))
            widths.append(width)
            heights.append(height)
            cxs.append(np.float32(cx))
            cys.append(np.float32(cy))
            image_paths.append(cam.image_path)

            if self.depth_enabled:
                depth = depth_cache_by_cam_idx.get(int(cam_idx))
                if depth is None:
                    depth_paths = _resolve_depth_paths(
                        images_path,
                        cam.image_path,
                        self.depth_subdir,
                        self.depth_extension,
                    )
                    depth_path = next((p for p in depth_paths if safe_isfile(p)), None)
                    if depth_path is None:
                        raise FileNotFoundError(
                            f"Depth file not found for scene {scene_id}, "
                            f"frame {Path(cam.image_path).name}. Tried: {depth_paths}."
                        )
                    depth = (
                        torch.from_numpy(_read_depth(depth_path)).float().unsqueeze(0)
                    )
                    depth, depth_width, depth_height, _, _ = (
                        self._downsample_depth_and_camera(
                            depth,
                            cam.width,
                            cam.height,
                            cam.cx,
                            cam.cy,
                        )
                    )
                else:
                    depth_width = int(depth.shape[-1])
                    depth_height = int(depth.shape[-2])
                if depth_width != width or depth_height != height:
                    raise RuntimeError(
                        "Depth and RGB downsampled shapes do not match "
                        f"for scene {scene_id}, frame {Path(cam.image_path).name}."
                    )
                depths.append(depth)

                if self.depth_chunk_mask_enabled and chunk_bounds_world is not None:
                    mask = _camera_chunk_membership_mask_from_depth(
                        cam,
                        depth,
                        chunk_bounds_world,
                        width=width,
                        height=height,
                        cx=cx,
                        cy=cy,
                        eps=CHUNK_INSIDE_EPS,
                    )
                    loss_masks.append(mask.to(torch.bool))

        payload = {
            "images": torch.stack(images, dim=0),
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
        if depths:
            payload["depths"] = torch.stack(depths, dim=0)
        if loss_masks:
            payload["loss_masks"] = torch.stack(loss_masks, dim=0)
        return payload

    def _sample_camera_indices_depth_chunk_stats(
        self,
        cam_infos,
        images_path: str,
        chunk_bounds_world: Tuple[List[float], List[float]],
    ) -> Tuple[np.ndarray, Dict[int, torch.Tensor]]:
        all_indices = np.arange(len(cam_infos), dtype=np.int64)
        target_probe_size = max(
            self.n_images,
            self.n_images * self.depth_camera_sampling_probe_multiplier,
        )
        probe_size = min(
            len(all_indices),
            max(self.n_images, target_probe_size),
        )
        if probe_size == len(all_indices):
            probe_indices = all_indices
        else:
            probe_indices = np.random.choice(
                all_indices,
                size=probe_size,
                replace=False,
            )

        scored_indices: List[int] = []
        scores: List[float] = []
        depth_cache_by_cam_idx: Dict[int, torch.Tensor] = {}

        for cam_idx in probe_indices:
            cam = cam_infos[int(cam_idx)]
            depth_paths = _resolve_depth_paths(
                images_path,
                cam.image_path,
                self.depth_subdir,
                self.depth_extension,
            )
            depth_path = next((p for p in depth_paths if safe_isfile(p)), None)
            if depth_path is None:
                continue

            try:
                depth = torch.from_numpy(_read_depth(depth_path)).float().unsqueeze(0)
            except Exception:
                continue

            depth, width, height, cx, cy = self._downsample_depth_and_camera(
                depth,
                cam.width,
                cam.height,
                cam.cx,
                cam.cy,
            )
            loss_keep = _camera_chunk_membership_mask_from_depth(
                cam,
                depth,
                chunk_bounds_world,
                width=width,
                height=height,
                cx=cx,
                cy=cy,
                eps=CHUNK_INSIDE_EPS,
            )[0]

            depth_map = depth[0]
            valid_scene = torch.isfinite(depth_map) & (depth_map > 0.0)
            background = torch.isposinf(depth_map)
            scene_in_chunk = valid_scene & loss_keep
            masked_scene = valid_scene & (~scene_in_chunk)

            stride = self.depth_camera_sampling_stride
            if stride > 1:
                valid_scene = valid_scene[::stride, ::stride]
                background = background[::stride, ::stride]
                scene_in_chunk = scene_in_chunk[::stride, ::stride]
                masked_scene = masked_scene[::stride, ::stride]

            denom = int(valid_scene.sum().item() + background.sum().item())
            if denom <= 0:
                continue

            scene_ratio = float(scene_in_chunk.sum().item()) / float(denom)
            masked_ratio = float(masked_scene.sum().item()) / float(denom)
            score = scene_ratio + (1e-3 * (1.0 - masked_ratio))
            if score <= 0.0:
                continue

            idx = int(cam_idx)
            scored_indices.append(idx)
            scores.append(score)
            depth_cache_by_cam_idx[idx] = depth

        if len(scored_indices) < self.n_images:
            replace = len(all_indices) < self.n_images
            camera_idxs = np.random.choice(
                all_indices,
                size=self.n_images,
                replace=replace,
            )
            return camera_idxs.astype(np.int64), {}

        score_arr = np.asarray(scores, dtype=np.float64)
        if not np.all(np.isfinite(score_arr)) or np.sum(score_arr) <= 0.0:
            replace = len(all_indices) < self.n_images
            camera_idxs = np.random.choice(
                all_indices,
                size=self.n_images,
                replace=replace,
            )
            return camera_idxs.astype(np.int64), {}

        probs = score_arr / np.sum(score_arr)
        candidate_arr = np.asarray(scored_indices, dtype=np.int64)
        replace = len(candidate_arr) < self.n_images
        camera_idxs = np.random.choice(
            candidate_arr,
            size=self.n_images,
            replace=replace,
            p=probs,
        )
        return camera_idxs.astype(np.int64), depth_cache_by_cam_idx

    def _downsample_depth_and_camera(
        self,
        depth: torch.Tensor,
        width: int,
        height: int,
        cx: float,
        cy: float,
    ) -> Tuple[torch.Tensor, int, int, float, float]:
        new_width, new_height, new_cx, new_cy = self._downsample_camera_params(
            width,
            height,
            cx,
            cy,
        )
        if new_width == int(width) and new_height == int(height):
            return depth, new_width, new_height, new_cx, new_cy

        downsampled = F.interpolate(
            depth.unsqueeze(0),
            size=(new_height, new_width),
            mode="nearest",
        ).squeeze(0)
        return downsampled, new_width, new_height, new_cx, new_cy

    def get_scene_id(self, path: str) -> str:
        p = Path(path)
        try:
            return p.parents[3].name
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


def _resolve_depth_paths(
    images_path: str,
    image_path: str,
    depth_subdir: str,
    depth_extension: str,
) -> List[str]:
    images_root = Path(images_path)
    image_file = Path(image_path)
    try:
        image_rel = image_file.relative_to(images_root)
    except ValueError:
        image_rel = Path(image_file.name)

    candidates = [images_root / depth_subdir / image_rel.with_suffix(depth_extension)]

    # Some datasets encode RGB root folders in transforms (e.g. "rgb_masked/..."),
    # while depth mirrors the remaining subpath under depth/.
    if len(image_rel.parts) > 1 and image_rel.parts[0] in {"rgb", "rgb_masked"}:
        candidates.append(
            images_root
            / depth_subdir
            / Path(*image_rel.parts[1:]).with_suffix(depth_extension)
        )

    unique_candidates: List[str] = []
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate_str not in unique_candidates:
            unique_candidates.append(candidate_str)
    return unique_candidates


def _read_depth(depth_path: str) -> np.ndarray:
    import imageio.v3 as iio

    depth = iio.imread(depth_path)
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth.astype(np.float32)


def _camera_chunk_projected_area_ratio(
    cam, chunk_bounds_world: Tuple[List[float], List[float]]
) -> float:
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

    w2c_rotation = cam.R.T.astype(np.float32)
    translation = cam.T.astype(np.float32)
    min_depth = 1e-4

    points_world = np.concatenate([corners, center[None, :]], axis=0)
    cam_coords = points_world @ w2c_rotation.T + translation
    z = cam_coords[:, 2]
    in_front = z > min_depth
    if not np.any(in_front):
        return 0.0

    fx = 0.5 * float(cam.width) / np.tan(0.5 * float(cam.FovX))
    fy = 0.5 * float(cam.height) / np.tan(0.5 * float(cam.FovY))
    u = fx * (cam_coords[:, 0] / z) + float(cam.cx)
    v = fy * (cam_coords[:, 1] / z) + float(cam.cy)
    u = u[in_front]
    v = v[in_front]

    width = float(cam.width)
    height = float(cam.height)
    u_min = np.clip(float(np.min(u)), 0.0, width)
    u_max = np.clip(float(np.max(u)), 0.0, width)
    v_min = np.clip(float(np.min(v)), 0.0, height)
    v_max = np.clip(float(np.max(v)), 0.0, height)
    if u_max <= u_min or v_max <= v_min:
        return 0.0

    projected_area = (u_max - u_min) * (v_max - v_min)
    image_area = max(width * height, 1e-8)
    return float(np.clip(projected_area / image_area, 0.0, 1.0))


def _camera_chunk_pixel_bounds(
    cam,
    chunk_bounds_world: Tuple[List[float], List[float]],
    *,
    width: int,
    height: int,
    cx: float,
    cy: float,
) -> Optional[Tuple[int, int, int, int]]:
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

    w2c_rotation = cam.R.T.astype(np.float32)
    translation = cam.T.astype(np.float32)
    min_depth = 1e-4

    points_world = np.concatenate([corners, center[None, :]], axis=0)
    cam_coords = points_world @ w2c_rotation.T + translation
    z = cam_coords[:, 2]
    in_front = z > min_depth
    if not np.any(in_front):
        return None

    fx = 0.5 * float(width) / np.tan(0.5 * float(cam.FovX))
    fy = 0.5 * float(height) / np.tan(0.5 * float(cam.FovY))
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
        return None

    x0 = max(0, int(np.floor(u_min)))
    x1 = min(width - 1, int(np.ceil(u_max)))
    y0 = max(0, int(np.floor(v_min)))
    y1 = min(height - 1, int(np.ceil(v_max)))
    if x1 < x0 or y1 < y0:
        return None

    return x0, x1, y0, y1


def _camera_chunk_membership_mask_from_depth(
    cam,
    depth: torch.Tensor,
    chunk_bounds_world: Tuple[List[float], List[float]],
    *,
    width: int,
    height: int,
    cx: float,
    cy: float,
    eps: float,
) -> torch.Tensor:
    if depth.ndim != 3 or depth.shape[0] != 1:
        raise ValueError(f"Expected depth shape (1, H, W), got {depth.shape}.")

    depth_map = depth[0]
    inf_keep_mask = torch.isposinf(depth_map)
    valid_depth = torch.isfinite(depth_map) & (depth_map > 0.0)
    if not torch.any(valid_depth):
        return inf_keep_mask.unsqueeze(0)

    pixel_bounds = _camera_chunk_pixel_bounds(
        cam,
        chunk_bounds_world,
        width=width,
        height=height,
        cx=cx,
        cy=cy,
    )
    if pixel_bounds is None:
        return inf_keep_mask.unsqueeze(0)

    x0, x1, y0, y1 = pixel_bounds
    bbox_mask = torch.zeros_like(valid_depth)
    bbox_mask[y0 : y1 + 1, x0 : x1 + 1] = True
    candidate_mask = valid_depth & bbox_mask
    if not torch.any(candidate_mask):
        return inf_keep_mask.unsqueeze(0)

    ys, xs = torch.where(candidate_mask)
    z = depth_map[ys, xs]

    fx = 0.5 * float(width) / np.tan(0.5 * float(cam.FovX))
    fy = 0.5 * float(height) / np.tan(0.5 * float(cam.FovY))
    x_cam = (xs.to(dtype=torch.float32) - float(cx)) * z / float(fx)
    y_cam = (ys.to(dtype=torch.float32) - float(cy)) * z / float(fy)
    cam_points = torch.stack([x_cam, y_cam, z], dim=1)

    w2c_rotation = torch.from_numpy(cam.R.T.astype(np.float32)).to(device=depth.device)
    translation = torch.from_numpy(cam.T.astype(np.float32)).to(device=depth.device)
    world_points = (cam_points - translation.unsqueeze(0)) @ w2c_rotation

    chunk_min = torch.tensor(
        chunk_bounds_world[0], device=depth.device, dtype=torch.float32
    )
    chunk_max = torch.tensor(
        chunk_bounds_world[1], device=depth.device, dtype=torch.float32
    )
    inside = torch.all(
        (world_points >= (chunk_min - float(eps)))
        & (world_points <= (chunk_max + float(eps))),
        dim=1,
    )

    mask = torch.zeros_like(depth_map, dtype=torch.bool)
    mask[ys, xs] = inside
    final_mask = mask | inf_keep_mask
    return final_mask.unsqueeze(0)


class VFrontDataModule(BaseSceneDataModule):
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
        sh_degree: int = 0,
        voxel_size: float = 0.025,
        gaussian_subpath: str = "v0.025_sigmoid_uniform_tanh/point_cloud/iteration_30000/ckpt.pth",
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
        depth_enabled: bool = False,
        depth_subdir: str = "depth",
        depth_extension: str = ".exr",
        depth_chunk_mask_enabled: bool = False,
        depth_camera_sampling_enabled: bool = False,
        depth_camera_sampling_probe_multiplier: int = 3,
        depth_camera_sampling_stride: int = 8,
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
        self.augmentations = list(augmentations or [])
        self.preload = preload
        self.sh_degree = sh_degree
        self.voxel_size = voxel_size
        self.gaussian_subpath = gaussian_subpath
        self.depth_enabled = bool(depth_enabled)
        self.depth_subdir = str(depth_subdir)
        self.depth_extension = str(depth_extension)
        self.depth_chunk_mask_enabled = bool(depth_chunk_mask_enabled)
        if self.depth_chunk_mask_enabled and not self.depth_enabled:
            raise ValueError(
                "`depth_chunk_mask_enabled` requires `depth_enabled=True`."
            )
        self.depth_camera_sampling_enabled = bool(depth_camera_sampling_enabled)
        self.depth_camera_sampling_probe_multiplier = int(
            depth_camera_sampling_probe_multiplier
        )
        self.depth_camera_sampling_stride = int(depth_camera_sampling_stride)

    def build_file_lists(self) -> Tuple[List[str], List[str]]:
        return get_default_data_split_vfront(
            self.data_path,
            train_list_path=self.train_list_path,
            val_list_path=self.val_list_path,
            gaussian_subpath=self.gaussian_subpath,
            skip_missing_files=True,
            verbose=self.verbose,
        )

    def get_dataset_class(self, split: str):
        return VFrontGaussianDataset

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
            sh_degree=self.sh_degree,
            voxel_size=self.voxel_size,
            depth_enabled=self.depth_enabled,
            depth_subdir=self.depth_subdir,
            depth_extension=self.depth_extension,
            depth_chunk_mask_enabled=self.depth_chunk_mask_enabled,
            depth_camera_sampling_enabled=self.depth_camera_sampling_enabled,
            depth_camera_sampling_probe_multiplier=self.depth_camera_sampling_probe_multiplier,
            depth_camera_sampling_stride=self.depth_camera_sampling_stride,
        )
        return kwargs


class VFrontPreprocessedDataModule(BaseSceneDataModule):
    def __init__(
        self,
        *,
        data_path: str,
        train_list_path: Optional[str],
        val_list_path: Optional[str],
        dataloader_kwargs: Optional[Dict] = None,
        overfit_scenes: int = 0,
        overfit_epoch_size: int = 1000,
        verbose: bool = False,
        augmentations: Optional[Iterable] = None,
        preprocessed_subpath: str = "v0.025_sigmoid_uniform_tanh/point_cloud/iteration_30000/ckpt.pt",
        background_color: str = "white",
        num_position_tokens: int = 2,
        position_vocab_size: Optional[int] = None,
        codebook_size: Optional[int] = None,
        chunk_shape: Optional[Sequence[int]] = None,
        dense_chunks: bool = False,
        chunk_order: str = "xyz",
        min_chunk_occupancy: float = 0.0,
        max_chunk_attempts: int = 1,
        chunk_origin: Optional[Sequence[Optional[int]]] = None,
        load_augmented_tokens: bool = False,
        shared: bool = False,
        ase_data_path: Optional[str] = None,
        ase_train_list_path: Optional[str] = None,
        ase_val_list_path: Optional[str] = None,
        ase_preprocessed_subpath: Optional[str] = None,
    ):
        super().__init__(
            data_path=data_path,
            train_list_path=train_list_path,
            val_list_path=val_list_path,
            img_path=None,
            overfit_scenes=overfit_scenes,
            overfit_epoch_size=overfit_epoch_size,
            max_points=None,
            n_images=0,
            load_normals=False,
            dataloader_kwargs=dataloader_kwargs,
            verbose=verbose,
            splitter=get_default_data_split_vfront,
            background_color=background_color,
        )

        self.augmentations = list(augmentations or [])
        self.preprocessed_subpath = preprocessed_subpath
        self.num_position_tokens = num_position_tokens
        self.position_vocab_size = position_vocab_size
        self.codebook_size = codebook_size
        self.chunk_shape = chunk_shape
        self.dense_chunks = dense_chunks
        self.chunk_order = chunk_order
        self.min_chunk_occupancy = min_chunk_occupancy
        self.max_chunk_attempts = max_chunk_attempts
        self.chunk_origin = chunk_origin
        self.load_augmented_tokens = load_augmented_tokens
        self.shared = shared
        self.ase_data_path = ase_data_path
        self.ase_train_list_path = ase_train_list_path
        self.ase_val_list_path = ase_val_list_path
        self.ase_preprocessed_subpath = ase_preprocessed_subpath

    def build_file_lists(self) -> Tuple[List[str], List[str]]:
        train_files, val_files = get_default_data_split_vfront(
            data_path=self.data_path,
            train_list_path=self.train_list_path,
            val_list_path=self.val_list_path,
            gaussian_subpath=self.preprocessed_subpath,
            require_files=not self.load_augmented_tokens,
            skip_missing_files=True,
            allow_augmented_files=self.load_augmented_tokens,
            verbose=self.verbose,
        )
        if self.ase_data_path:
            from data.ase_dataset import get_default_data_split_ase

            ase_subpath = self.ase_preprocessed_subpath or self.preprocessed_subpath
            ase_train_files, ase_val_files = get_default_data_split_ase(
                data_path=self.ase_data_path,
                train_list_path=self.ase_train_list_path,
                val_list_path=self.ase_val_list_path,
                gaussian_subpath=ase_subpath,
                require_files=not self.load_augmented_tokens,
                skip_missing_files=True,
                verbose=self.verbose,
            )
            train_files.extend(ase_train_files)
            val_files.extend(ase_val_files)
            if self.verbose:
                print(
                    "INFO: Combined tokenized training data with ASE "
                    f"({len(ase_train_files)} train, {len(ase_val_files)} val)."
                )
        return train_files, val_files

    def get_dataset_class(self, split: str):
        return PreprocessedDataset

    def get_dataset_kwargs(self, split: str, file_list: Sequence[str]) -> Dict:
        augmentations = self.augmentations if split == "train" else None
        return dict(
            file_list=file_list,
            augmentations=augmentations,
            num_position_tokens=self.num_position_tokens,
            position_vocab_size=self.position_vocab_size,
            codebook_size=self.codebook_size,
            chunk_shape=self.chunk_shape,
            dense_chunks=self.dense_chunks,
            chunk_order=self.chunk_order,
            min_chunk_occupancy=self.min_chunk_occupancy,
            max_chunk_attempts=self.max_chunk_attempts,
            chunk_origin=self.chunk_origin,
            load_augmented_tokens=self.load_augmented_tokens,
            shared=self.shared,
        )
