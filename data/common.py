from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from functools import partial
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import lightning
import torch
import torch.nn.functional as F

from conf.dataclasses import ImageKeys, PCKeys
from data.vfront import safe_exists, safe_torch_load
from serialization import encode
from utils.gaussian_vqvae_utils import int_cube_root
from utils.pos_tokens import coords_to_pos_tokens, dense_chunk_order_indices

_SUPPORTED_CHUNK_ORDERS = {"xyz", "xzy", "z", "z-trans", "hilbert", "hilbert-trans"}


def collate_fn(batch: list[Dict[str, torch.Tensor]], max_points: int = None):
    """
    Collate function for point clouds that combines multiple dicts into one.
    Batches are stored and point clouds are concatenated to form a single tensor.
    """
    if not batch:
        return {}

    max_points_limit = float("inf") if max_points is None else max_points

    original_batch_size = len(batch)
    total_points = 0
    truncated = False
    cutoff = original_batch_size

    for idx, item in enumerate(batch):
        num_points = item[PCKeys.COORDS].shape[0]

        if total_points and total_points + num_points > max_points_limit:
            truncated = True
            cutoff = idx
            break
        if total_points == 0 and num_points > max_points_limit:
            total_points = num_points
            cutoff = idx + 1
            truncated = idx < original_batch_size - 1
            break

        total_points += num_points

    if truncated:
        batch = batch[:cutoff]
        print(
            f"INFO: [collate_fn] truncated batch from {original_batch_size} "
            f"to {cutoff} items (using {total_points} points, limit {max_points})."
        )

    collated_batch = {}
    # add batch information for point cloud data
    # ie. N1 many 0, N2 many 1, ... -> (N1 + N2 + ...) integers
    collated_batch[PCKeys.BATCH] = torch.tensor(
        [i for i, item in enumerate(batch) for _ in range(item[PCKeys.COORDS].shape[0])]
    )

    for key in batch[0].keys():
        if key in iter(PCKeys):
            # Concatenate point cloud data into single tensors
            # ie. N1 x 3, N2 x 3, ... -> (N1 + N2 + ...) x 3
            collated_batch[key] = torch.cat([item[key] for item in batch], dim=0)

        elif key in iter(ImageKeys):
            # batch dimension is fine here, ie we want B x N x D
            if isinstance(batch[0][key], torch.Tensor):
                collated_batch[key] = torch.stack([item[key] for item in batch], dim=0)
            else:
                # image paths are strings
                collated_batch[key] = [item[key] for item in batch]

    return collated_batch


def collate_token_sequences(batch):
    """
    Collate token sequences for GPT training.
    Expected items: 1D LongTensor with shape (T,).
    Padding uses zeros; callers must provide lengths/attention masks to ignore pads.
    """
    if not batch:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

    tokens_list = batch

    lengths = torch.tensor([t.shape[0] for t in tokens_list], dtype=torch.long)
    max_len = int(lengths.max().item())
    padded = torch.zeros((len(tokens_list), max_len), dtype=torch.long)
    for i, t in enumerate(tokens_list):
        padded[i, : t.shape[0]] = t

    return padded, lengths


class BaseSceneDataset(torch.utils.data.Dataset, ABC):
    """
    Base PyTorch dataset for geometry-centric samples (e.g. surface points or Gaussians).

    Concrete datasets only need to implement :meth:`load_gaussians` (for the core
    geometry) and optionally :meth:`load_images` / :meth:`postprocess_item`.
    """

    def __init__(
        self,
        paths: Sequence[str],
        *,
        split: str,
        load_normals: bool = False,
        max_points: Optional[int] = None,
        deterministic_sampling: bool = False,
        n_images: int = 0,
        img_path: Optional[str] = None,
        augmentations: Optional[Iterable] = None,
        preload: bool = False,
        verbose: bool = False,
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
        super().__init__()

        self.paths = list(paths)
        self.split = split

        self.load_normals = load_normals
        self.max_points = max_points
        self.deterministic_sampling = deterministic_sampling

        self.n_images = n_images
        self.img_path = img_path

        if self.n_images > 0 and not self.img_path:
            raise ValueError(
                "Requested images but no image root (`img_path`) provided."
            )

        self.augmentations = list(augmentations or [])
        self.preload = preload
        self.verbose = verbose
        if background_color not in ("white", "black"):
            raise ValueError(
                f"Unsupported background color '{background_color}'. Expected 'white' or 'black'."
            )
        self.background_color = background_color
        self.center_sample = center_sample
        self.frustum_subsample = frustum_subsample
        self.frustum_subsample_margin = float(frustum_subsample_margin)
        self.chunk_subsample = bool(chunk_subsample)
        self.chunk_shape = (
            [int(chunk_shape)] * 3
            if isinstance(chunk_shape, int)
            else list(chunk_shape or [])
        )
        self.chunk_voxel_size = (
            None if chunk_voxel_size is None else float(chunk_voxel_size)
        )
        self.min_chunk_occupancy = float(min_chunk_occupancy)
        self.max_chunk_attempts = max(1, int(max_chunk_attempts))
        self.camera_chunk_min_area_ratio = float(camera_chunk_min_area_ratio)
        downsample_factor = float(image_downsample_factor)
        if downsample_factor < 1.0 or not downsample_factor.is_integer():
            raise ValueError(
                "`image_downsample_factor` must be an integer >= 1, "
                f"got {image_downsample_factor}."
            )
        self.image_downsample_factor = int(downsample_factor)
        if chunk_origin is None:
            self.chunk_origin = None
        else:
            if len(chunk_origin) != 3:
                raise ValueError("chunk_origin must have 3 elements.")
            self.chunk_origin = [
                None if v is None else int(v) for v in list(chunk_origin)
            ]
        if self.frustum_subsample_margin < 0.0:
            raise ValueError(
                "`frustum_subsample_margin` must be >= 0, "
                f"got {self.frustum_subsample_margin}."
            )
        if self.frustum_subsample and self.n_images <= 0:
            raise ValueError(
                "`frustum_subsample` requires `n_images > 0` so camera views are available."
            )
        if self.frustum_subsample and self.chunk_subsample:
            raise ValueError(
                "Frustum subsampling and chunk subsampling are mutually exclusive."
            )
        if self.min_chunk_occupancy < 0.0 or self.min_chunk_occupancy > 1.0:
            raise ValueError(
                "`min_chunk_occupancy` must be between 0 and 1, "
                f"got {self.min_chunk_occupancy}."
            )
        if (
            self.camera_chunk_min_area_ratio < 0.0
            or self.camera_chunk_min_area_ratio > 1.0
        ):
            raise ValueError(
                "`camera_chunk_min_area_ratio` must be between 0 and 1, "
                f"got {self.camera_chunk_min_area_ratio}."
            )
        if self.chunk_subsample:
            if len(self.chunk_shape) != 3:
                raise ValueError("chunk_shape must have 3 elements for chunk sampling.")
            if any(c <= 0 for c in self.chunk_shape):
                raise ValueError("chunk_shape must contain only positive values.")
            if self.chunk_voxel_size is None or self.chunk_voxel_size <= 0.0:
                raise ValueError(
                    "chunk_voxel_size must be > 0 when chunk_subsample is enabled."
                )
        self._background_rgb = (
            (1.0, 1.0, 1.0) if background_color == "white" else (0.0, 0.0, 0.0)
        )

        self._preloaded_data: Optional[List[Dict[str, torch.Tensor]]] = None
        if self.preload:
            self._preload_dataset()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        base_sample, path = self._get_base_sample(idx)
        item = self._finalise_sample(base_sample, path)
        for augmentation in self.augmentations:
            item = augmentation(item)
        return item

    # --------------------------------------------------------------------- #
    # Hooks for subclasses
    # --------------------------------------------------------------------- #
    @abstractmethod
    def load_gaussians(self, path: str) -> Dict[str, torch.Tensor]:
        """
        Read the core geometry for a single scene.

        The returned dictionary must at least contain the following tensors:

        ``coords``: ``torch.Tensor`` of shape [N, 3] describing Gaussian means.
        ``sh0``: ``torch.Tensor`` of shape [N, 3] holding spherical harmonic DCs.

        Optional keys can include ``opacities``, ``scales``, ``quats``, ``normal`` or
        any other per-Gaussian attributes that downstream models expect.
        """

    def load_images(
        self, path: str, item_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Load camera information and associated images for a scene.

        Subclasses can override this to append image tensors and camera intrinsics /
        extrinsics that match the requested number of images.
        """

        return {}

    def postprocess_item(
        self, item_dict: Dict[str, torch.Tensor], path: str
    ) -> Dict[str, torch.Tensor]:
        """
        Hook for subclasses to attach additional information after the base
        processing has run.  The default implementation simply returns the item.
        """

        return item_dict

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _preload_dataset(self) -> None:
        if self.verbose:
            print(f"INFO: Preloading {len(self)} geometry samples.")
        self._preloaded_data = []
        for path in self.paths:
            sample = self.load_gaussians(path)
            self._preloaded_data.append(sample)

    def _get_base_sample(self, idx: int) -> Tuple[Dict[str, torch.Tensor], str]:
        path = self.paths[idx]
        if self._preloaded_data is not None:
            base = copy.deepcopy(self._preloaded_data[idx])
        else:
            base = self.load_gaussians(path)
        return base, path

    def _finalise_sample(
        self, base_sample: Dict[str, torch.Tensor], path: str
    ) -> Dict[str, torch.Tensor]:
        item_dict = dict(base_sample)

        if "coords" not in item_dict or "sh0" not in item_dict:
            raise KeyError(
                "Base sample must provide 'coords' and 'sh0' tensors. "
                "Ensure your subclass returns the expected keys."
            )

        num_points = item_dict["coords"].shape[0]
        if (
            self.max_points is not None
            and num_points > self.max_points
            and not self.frustum_subsample
            and not self.chunk_subsample
        ):
            indices = self._select_point_indices(num_points, item_dict["coords"].device)
            self._apply_point_indices(item_dict, indices, num_points)

        if self.load_normals:
            normals = item_dict.get("normal")
            if normals is None:
                raise ValueError(
                    "Normals requested but the sample did not provide any."
                )
            item_dict["normal"] = normals
        else:
            item_dict.pop("normal", None)

        item_dict["scene_id"] = self.get_scene_id(path)

        chunk_indices: Optional[torch.Tensor] = None
        if self.chunk_subsample:
            chunk_indices, chunk_bounds_world = self._sample_chunk_indices_and_bounds(
                item_dict
            )
            item_dict["_chunk_bounds_world"] = chunk_bounds_world

        camera_payload = self.load_images(path, item_dict)
        if camera_payload:
            item_dict.update(camera_payload)
        item_dict.pop("_chunk_bounds_world", None)

        if self.frustum_subsample:
            indices = self._select_visible_point_indices(item_dict)
            self._apply_point_indices(
                item_dict,
                indices,
                item_dict["coords"].shape[0],
            )
        elif self.chunk_subsample:
            indices = (
                chunk_indices
                if chunk_indices is not None
                else self._select_chunk_point_indices(item_dict)
            )
            self._apply_point_indices(
                item_dict,
                indices,
                item_dict["coords"].shape[0],
            )

        if (
            self.max_points is not None
            and item_dict["coords"].shape[0] > self.max_points
        ):
            indices = self._select_point_indices(
                item_dict["coords"].shape[0], item_dict["coords"].device
            )
            self._apply_point_indices(
                item_dict,
                indices,
                item_dict["coords"].shape[0],
            )

        if self.center_sample:
            self._center_geometry_sample(item_dict)

        item_dict = self.postprocess_item(item_dict, path)
        return item_dict

    def _select_point_indices(
        self, num_points: int, device: torch.device
    ) -> torch.Tensor:
        if self.max_points is None:
            raise ValueError("`max_points` must be set when sampling indices.")
        if self.deterministic_sampling:
            step = max(1, num_points // self.max_points)
            return torch.arange(0, step * self.max_points, step, device=device)[
                : self.max_points
            ]
        return torch.randperm(num_points, device=device)[: self.max_points]

    def _apply_point_indices(
        self,
        item_dict: Dict[str, torch.Tensor],
        indices: torch.Tensor,
        num_points: int,
    ) -> None:
        for key, value in list(item_dict.items()):
            if (
                isinstance(value, torch.Tensor)
                and value.ndim > 0
                and value.shape[0] == num_points
            ):
                item_dict[key] = value[indices]

    def _select_chunk_point_indices(
        self, item_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        indices, _ = self._sample_chunk_indices_and_bounds(item_dict)
        return indices

    def _sample_chunk_indices_and_bounds(
        self, item_dict: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[List[float], List[float]]]:
        if not self.chunk_shape:
            raise ValueError("chunk_shape must be set for chunk sampling.")
        if self.chunk_voxel_size is None or self.chunk_voxel_size <= 0:
            raise ValueError("chunk_voxel_size must be set for chunk sampling.")

        coords = item_dict[PCKeys.COORDS]
        num_points = coords.shape[0]
        device = coords.device

        if num_points == 0:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return empty, ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

        voxel_coords = torch.round(
            coords.to(dtype=torch.float32) / self.chunk_voxel_size
        ).to(dtype=torch.long)
        min_coord = voxel_coords.min(dim=0).values
        max_coord = voxel_coords.max(dim=0).values
        chunk = torch.tensor(self.chunk_shape, dtype=torch.long, device=device)
        max_start = max_coord - chunk + 1

        num_voxels = int(chunk[0] * chunk[1] * chunk[2])
        min_required = int(self.min_chunk_occupancy * num_voxels)

        best_mask = None
        best_start = None
        best_count = -1
        for _ in range(self.max_chunk_attempts):
            start = []
            for dim in range(3):
                if self.chunk_origin is not None and self.chunk_origin[dim] is not None:
                    start.append(self.chunk_origin[dim])
                    continue

                min_val = int(min_coord[dim].item())
                max_val = int(max_start[dim].item())
                if max_val < min_val:
                    start.append(min_val)
                    continue

                start.append(
                    int(torch.randint(min_val, max_val + 1, (1,), device=device).item())
                )

            start = torch.tensor(start, dtype=torch.long, device=device)
            end = start + chunk
            mask = torch.all((voxel_coords >= start) & (voxel_coords < end), dim=1)
            count = int(mask.sum().item())

            if count > best_count:
                best_count = count
                best_mask = mask
                best_start = start

            if count >= min_required and count > 0:
                break

        if best_mask is None or best_count <= 0:
            if self.verbose:
                print(
                    "WARNING: Chunk sampling produced an empty chunk; falling back "
                    "to full-scene geometry (may increase memory usage)."
                )
            min_world = coords.min(dim=0).values.to(dtype=torch.float32)
            max_world = coords.max(dim=0).values.to(dtype=torch.float32)
            return (
                torch.arange(num_points, device=device),
                (min_world.tolist(), max_world.tolist()),
            )

        if best_start is None:
            raise RuntimeError("Chunk sampling failed to keep the best chunk origin.")

        start_world = best_start.to(dtype=torch.float32) * float(self.chunk_voxel_size)
        end_world = (best_start + chunk).to(dtype=torch.float32) * float(
            self.chunk_voxel_size
        )
        return (
            torch.nonzero(best_mask, as_tuple=False).flatten(),
            (start_world.tolist(), end_world.tolist()),
        )

    def _select_visible_point_indices(
        self, item_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        coords = item_dict[PCKeys.COORDS]
        num_points = coords.shape[0]
        device = coords.device

        required_keys = (
            ImageKeys.CAMERAS_R,
            ImageKeys.CAMERAS_T,
            ImageKeys.CAMERAS_FOVX,
            ImageKeys.CAMERAS_FOVY,
            ImageKeys.CAMERAS_W,
            ImageKeys.CAMERAS_H,
        )
        if any(key not in item_dict for key in required_keys):
            raise KeyError(
                "Frustum subsampling requires camera intrinsics/extrinsics in the sample."
            )

        rotations = item_dict[ImageKeys.CAMERAS_R].to(
            dtype=torch.float32, device=device
        )
        translations = item_dict[ImageKeys.CAMERAS_T].to(
            dtype=torch.float32, device=device
        )
        fovx = item_dict[ImageKeys.CAMERAS_FOVX].to(dtype=torch.float32, device=device)
        fovy = item_dict[ImageKeys.CAMERAS_FOVY].to(dtype=torch.float32, device=device)
        widths = item_dict[ImageKeys.CAMERAS_W].to(dtype=torch.float32, device=device)
        heights = item_dict[ImageKeys.CAMERAS_H].to(dtype=torch.float32, device=device)

        if ImageKeys.CAMERAS_CX in item_dict:
            cxs = item_dict[ImageKeys.CAMERAS_CX].to(dtype=torch.float32, device=device)
        else:
            cxs = 0.5 * widths
        if ImageKeys.CAMERAS_CY in item_dict:
            cys = item_dict[ImageKeys.CAMERAS_CY].to(dtype=torch.float32, device=device)
        else:
            cys = 0.5 * heights

        visible = torch.zeros(num_points, dtype=torch.bool, device=device)
        min_depth = 1e-4
        margin_ratio = float(self.frustum_subsample_margin)
        coords_float = coords.to(dtype=torch.float32)
        for cam_idx in range(rotations.shape[0]):
            w2c_rotation = rotations[cam_idx].transpose(0, 1)
            cam_coords = coords_float @ w2c_rotation.T + translations[cam_idx]
            z = cam_coords[:, 2]
            in_front = z > min_depth
            if not in_front.any():
                continue

            fx = 0.5 * widths[cam_idx] / torch.tan(0.5 * fovx[cam_idx])
            fy = 0.5 * heights[cam_idx] / torch.tan(0.5 * fovy[cam_idx])
            x = cam_coords[:, 0]
            y = cam_coords[:, 1]
            u = fx * (x / z) + cxs[cam_idx]
            v = fy * (y / z) + cys[cam_idx]

            margin_x = widths[cam_idx] * margin_ratio
            margin_y = heights[cam_idx] * margin_ratio
            inside = (
                in_front
                & (u >= -margin_x)
                & (u < widths[cam_idx] + margin_x)
                & (v >= -margin_y)
                & (v < heights[cam_idx] + margin_y)
            )
            visible |= inside

        return torch.nonzero(visible, as_tuple=False).flatten()

    def _resolve_image_path(self, geometry_path: str) -> str:
        """
        Default helper for datasets that collocate images with their geometry file.
        Subclasses can override if they use bespoke layouts.
        """

        if self.img_path is None:
            raise ValueError("Cannot resolve image path without `img_path`.")
        scene_id = self.get_scene_id(geometry_path)
        return f"{self.img_path.rstrip('/')}/{scene_id}"

    def _downsample_camera_params(
        self,
        width: int,
        height: int,
        cx: float,
        cy: float,
    ) -> Tuple[int, int, float, float]:
        factor = self.image_downsample_factor
        width_i = int(width)
        height_i = int(height)
        if factor == 1:
            return width_i, height_i, float(cx), float(cy)

        new_width = max(1, width_i // factor)
        new_height = max(1, height_i // factor)
        scale_x = float(new_width) / float(width_i)
        scale_y = float(new_height) / float(height_i)
        return (
            new_width,
            new_height,
            float(cx) * scale_x,
            float(cy) * scale_y,
        )

    def _downsample_image_and_camera(
        self,
        image: Optional[torch.Tensor],
        width: int,
        height: int,
        cx: float,
        cy: float,
    ) -> Tuple[Optional[torch.Tensor], int, int, float, float]:
        new_width, new_height, new_cx, new_cy = self._downsample_camera_params(
            width,
            height,
            cx,
            cy,
        )
        if image is None:
            return None, new_width, new_height, new_cx, new_cy

        if new_width == int(width) and new_height == int(height):
            return image, new_width, new_height, new_cx, new_cy

        downsampled = F.interpolate(
            image.unsqueeze(0),
            size=(new_height, new_width),
            mode="area",
        ).squeeze(0)
        return downsampled, new_width, new_height, new_cx, new_cy

    def get_scene_id(self, path: str) -> str:
        """
        Provide a stable identifier for the scene the current sample belongs to.
        """

        filename = path.rstrip("/").split("/")[-1]
        return filename.split(".")[0]

    def _center_geometry_sample(self, item_dict: Dict[str, torch.Tensor]) -> None:
        coords = item_dict["coords"]
        if coords.numel() == 0:
            return

        min_x, max_x = torch.aminmax(coords[:, 0])
        min_y, max_y = torch.aminmax(coords[:, 1])
        min_z, max_z = torch.aminmax(coords[:, 2])
        centroid = torch.tensor(
            [[0.5 * (min_x + max_x), 0.5 * (min_y + max_y), 0.5 * (min_z + max_z)]],
            device=coords.device,
        )
        item_dict["coords"] = coords - centroid

        if ImageKeys.CAMERAS_T not in item_dict:
            return

        translations = item_dict[ImageKeys.CAMERAS_T]
        rotations = item_dict[ImageKeys.CAMERAS_R]

        centroid_vec = centroid.squeeze(0).to(translations)
        rotation_matrices = rotations.transpose(-1, -2)
        offset = torch.matmul(rotation_matrices, centroid_vec)
        item_dict[ImageKeys.CAMERAS_T] = translations + offset


class PreprocessedDataset(torch.utils.data.Dataset):
    """
    Small convenience dataset for loading pre-serialised ``torch.save`` payloads.
    """

    def __init__(
        self,
        file_list: Sequence[str],
        augmentations: Optional[Iterable] = None,
        num_position_tokens: int = 2,
        position_vocab_size: Optional[int] = None,
        codebook_size: int = None,
        chunk_shape: Optional[Sequence[int]] = None,
        dense_chunks: bool = False,
        chunk_order: str = "xyz",
        min_chunk_occupancy: float = 0.0,
        max_chunk_attempts: int = 1,
        chunk_origin: Optional[Sequence[Optional[int]]] = None,
        load_augmented_tokens: bool = False,
        shared: bool = False,
    ):
        super().__init__()
        self.file_list = self._resolve_file_list(file_list, load_augmented_tokens)
        self.augmentations = list(augmentations or [])
        self.num_position_tokens = num_position_tokens
        if codebook_size is None:
            raise ValueError("codebook_size must be set for tokenized data loading.")
        self.codebook_size = int(codebook_size)
        if position_vocab_size is None:
            # Legacy path: unshifted feature ids and pad at codebook_size + 1.
            self.position_vocab_size = self.codebook_size
            self.position_side_length = int_cube_root(self.position_vocab_size)
            self.feature_token_offset = 0
            self.pad_token_id = self.codebook_size + 1
        else:
            self.position_vocab_size = int(position_vocab_size)
            if self.position_vocab_size <= 0:
                raise ValueError("position_vocab_size must be > 0.")
            self.position_side_length = int_cube_root(self.position_vocab_size)
            if self.position_side_length**3 != self.position_vocab_size:
                raise ValueError(
                    "position_vocab_size must be a perfect cube (k^3) for 3D position tokenization."
                )
            if shared:
                self.feature_token_offset = 0
                self.pad_token_id = (
                    max(self.position_vocab_size, self.codebook_size) + 1
                )
            else:
                self.feature_token_offset = self.position_vocab_size
                self.pad_token_id = self.feature_token_offset + self.codebook_size + 1
        self.chunk_shape = (
            [int(chunk_shape)] * 3
            if isinstance(chunk_shape, int)
            else list(chunk_shape or [])
        )
        self.dense_chunks = dense_chunks
        if chunk_order not in _SUPPORTED_CHUNK_ORDERS:
            raise ValueError(
                "Unsupported chunk_order. Expected one of "
                f"{sorted(_SUPPORTED_CHUNK_ORDERS)}, got '{chunk_order}'."
            )
        self.chunk_order = chunk_order
        self.min_chunk_occupancy = min_chunk_occupancy
        self.max_chunk_attempts = max(1, max_chunk_attempts)
        self._dense_order = None
        if self.dense_chunks and self.chunk_shape:
            self._dense_order = dense_chunk_order_indices(
                self.chunk_shape, self.chunk_order
            )
        if chunk_origin is None:
            self.chunk_origin = None
        else:
            if len(chunk_origin) != 3:
                raise ValueError("chunk_origin must have 3 elements.")
            self.chunk_origin = [
                None if v is None else int(v) for v in list(chunk_origin)
            ]

    @staticmethod
    def _resolve_file_list(
        file_list: Sequence[str], load_augmented_tokens: bool
    ) -> List[str]:
        resolved: List[str] = []

        for file_path in file_list:
            path = Path(file_path)
            candidates: List[Path] = []

            if load_augmented_tokens:
                augmented = sorted(path.parent.glob(f"{path.stem}_rot*{path.suffix}"))
                if augmented:
                    candidates.extend(augmented)
                elif safe_exists(path):
                    candidates.append(path)
            elif safe_exists(path):
                candidates.append(path)

            if not candidates:
                raise FileNotFoundError(
                    f"No tokenized file found for split entry {file_path}."
                )

            resolved.extend(str(candidate) for candidate in candidates)

        return resolved

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int):
        path = self.file_list[idx]
        data = safe_torch_load(path)

        for augmentation in self.augmentations:
            data = augmentation(data)

        if (
            not isinstance(data, dict)
            or "coords" not in data
            or "feature_ids" not in data
        ):
            raise ValueError(
                "Expected tokenized data dict with 'coords' and 'feature_ids'."
            )

        coords = data["coords"]
        feature_ids = data["feature_ids"]

        coords = coords.to(torch.long)
        feature_ids = feature_ids.to(torch.long)
        if feature_ids.dim() == 1:
            feature_ids = feature_ids.unsqueeze(1)
        if coords.dim() != 2 or coords.shape[1] != 3:
            raise ValueError(f"Expected coords with shape (N, 3), got {coords.shape}.")
        if feature_ids.dim() != 2:
            raise ValueError(
                f"Expected feature_ids with shape (N, T), got {feature_ids.shape}."
            )
        if feature_ids.shape[0] != coords.shape[0]:
            raise ValueError(
                "coords/feature_ids length mismatch: "
                f"{coords.shape[0]} vs {feature_ids.shape[0]}."
            )
        pad_token_id = self.pad_token_id

        if coords.numel() == 0 and self.dense_chunks:
            if not self.chunk_shape:
                return torch.empty(0, dtype=torch.long)
            num_voxels = int(
                self.chunk_shape[0] * self.chunk_shape[1] * self.chunk_shape[2]
            )
            num_features = int(feature_ids.shape[1])
            tokens = torch.full(
                (num_voxels * num_features,),
                pad_token_id,
                dtype=torch.long,
            )
            return tokens
        if coords.numel() == 0:
            return torch.empty(0, dtype=torch.long)

        origin = None
        if self.chunk_shape:
            if len(self.chunk_shape) != 3:
                raise ValueError("chunk_shape must have 3 elements.")
            min_coord = coords.min(dim=0).values
            max_coord = coords.max(dim=0).values
            chunk = torch.tensor(self.chunk_shape, device=coords.device)
            max_start = max_coord - chunk + 1
            num_features = int(feature_ids.shape[1])
            num_voxels = int(chunk[0] * chunk[1] * chunk[2])
            min_required = int(self.min_chunk_occupancy * num_voxels)

            best_mask = None
            best_start = None
            best_count = -1
            for _ in range(self.max_chunk_attempts):
                start = []
                for dim in range(3):
                    if (
                        self.chunk_origin is not None
                        and self.chunk_origin[dim] is not None
                    ):
                        start.append(self.chunk_origin[dim])
                        continue
                    if max_start[dim] < min_coord[dim]:
                        start.append(min_coord[dim].item())
                    else:
                        start.append(
                            int(
                                torch.randint(
                                    min_coord[dim],
                                    max_start[dim] + 1,
                                    (1,),
                                    device=coords.device,
                                ).item()
                            )
                        )
                start = torch.tensor(start, device=coords.device)
                end = start + chunk
                mask = torch.all((coords >= start) & (coords < end), dim=1)
                count = int(mask.sum().item())
                if count > best_count:
                    best_count = count
                    best_mask = mask
                    best_start = start
                if count >= min_required:
                    break

            coords = coords[best_mask]
            feature_ids = feature_ids[best_mask]
            if coords.numel() == 0:
                if self.dense_chunks:
                    tokens = torch.full(
                        (num_voxels * num_features,),
                        pad_token_id,
                        dtype=torch.long,
                    )
                    return tokens
                return torch.empty(0, dtype=torch.long)
            origin = best_start
        else:
            origin = coords.min(dim=0).values

        coords = coords - origin

        if self.dense_chunks:
            if not self.chunk_shape:
                raise ValueError("chunk_shape must be set for dense chunks.")
            chunk = torch.tensor(self.chunk_shape, device=coords.device)
            if coords.min().item() < 0 or torch.any(coords >= chunk):
                raise ValueError("Coords out of bounds for dense chunk.")
            num_features = int(feature_ids.shape[1])
            num_voxels = int(chunk[0] * chunk[1] * chunk[2])
            feature_ids_full = torch.full(
                (num_voxels, num_features),
                pad_token_id,
                dtype=torch.long,
                device=coords.device,
            )
            linear_idx = (
                coords[:, 0] * (chunk[1] * chunk[2])
                + coords[:, 1] * chunk[2]
                + coords[:, 2]
            )
            feature_ids_full[linear_idx] = feature_ids + self.feature_token_offset
            if self._dense_order is not None:
                order = self._dense_order.to(feature_ids_full.device)
                feature_ids_full = feature_ids_full[order]
            return feature_ids_full.reshape(-1).cpu()

        if coords.numel() > 0:
            if self.chunk_order == "xyz":
                side_y = int(coords[:, 1].max().item()) + 1
                side_z = int(coords[:, 2].max().item()) + 1
                keys = (
                    coords[:, 0] * (side_y * side_z)
                    + coords[:, 1] * side_z
                    + coords[:, 2]
                )
            elif self.chunk_order == "xzy":
                side_y = int(coords[:, 1].max().item()) + 1
                side_z = int(coords[:, 2].max().item()) + 1
                keys = (
                    coords[:, 0] * (side_y * side_z)
                    + coords[:, 2] * side_y
                    + coords[:, 1]
                )
            else:
                depth = int(coords.max().item() + 1).bit_length()
                keys = encode(coords, depth=depth, order=self.chunk_order)
            order = torch.argsort(keys.reshape(-1))
            coords = coords.index_select(0, order)
            feature_ids = feature_ids.index_select(0, order)

        base_side_length = self.position_side_length
        side_length = base_side_length**self.num_position_tokens
        if self.chunk_shape and any(c > side_length for c in self.chunk_shape):
            raise ValueError(
                "chunk_shape exceeds available position token side length."
            )
        if coords.min().item() < 0 or coords.max().item() >= side_length:
            raise ValueError("Coords out of bounds for position tokenization.")

        pos_tokens = coords_to_pos_tokens(
            coords, self.num_position_tokens, base_side_length
        )
        tokens = torch.cat(
            [pos_tokens, feature_ids + self.feature_token_offset], dim=1
        ).reshape(-1)

        return tokens


# TODO for all children of this: don't pass all parameters one by one, just use **kwargs for maintanability
class BaseSceneDataModule(lightning.LightningDataModule, ABC):
    """
    Lightning datamodule that factors out recurring plumbing for our datasets.

    Shared responsibilities:
      * Build train/val file lists.
      * Expand overfit subsets to reach a target epoch size.
      * Instantiate a dataset per split with shared parameters.
    """

    def __init__(
        self,
        *,
        data_path: str,
        train_list_path: Optional[str],
        val_list_path: Optional[str],
        img_path: Optional[str] = None,
        overfit_scenes: int = 0,
        overfit_epoch_size: int = 1000,
        overfit_min_val_scenes: int = 10,
        max_points: Optional[int] = None,
        max_batch_points: Optional[int] = None,
        n_images: int = 0,
        load_normals: bool = False,
        dataloader_kwargs: Optional[Dict] = None,
        deterministic_sampling: bool = False,
        verbose: bool = False,
        splitter: Optional[Callable[..., Tuple[List[str], List[str]]]] = None,
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
        super().__init__()

        self.data_path = data_path
        self.train_list_path = train_list_path
        self.val_list_path = val_list_path
        self.img_path = img_path

        self.overfit_scenes = overfit_scenes
        self.overfit_epoch_size = overfit_epoch_size
        self.overfit_min_val_scenes = overfit_min_val_scenes
        self.max_points = max_points
        self.max_batch_points = max_batch_points
        self.n_images = n_images
        self.load_normals = load_normals

        self.deterministic_sampling = deterministic_sampling

        self.dataloader_kwargs_train = dict(
            {
                "num_workers": 4,
                "pin_memory": True,
                "persistent_workers": False,
                "prefetch_factor": 1,
                "shuffle": True,
                "drop_last": True,
            }
        )
        self.dataloader_kwargs_val = dict(
            {
                "num_workers": 2,
                "pin_memory": True,
                "persistent_workers": False,
                "prefetch_factor": 1,
                "shuffle": False,
                "drop_last": False,
            }
        )

        if dataloader_kwargs:
            self.dataloader_kwargs_train.update(dataloader_kwargs)
            self.dataloader_kwargs_val.update(dataloader_kwargs)
        self.verbose = verbose

        self.splitter = splitter

        self.train_dataset: Optional[torch.utils.data.Dataset] = None
        self.val_dataset: Optional[torch.utils.data.Dataset] = None
        if background_color not in ("white", "black"):
            raise ValueError(
                f"Unsupported background color '{background_color}'. Expected 'white' or 'black'."
            )
        self.background_color = background_color
        self.center_sample = center_sample
        self.frustum_subsample = bool(frustum_subsample)
        self.frustum_subsample_margin = float(frustum_subsample_margin)
        self.chunk_subsample = bool(chunk_subsample)
        self.chunk_shape = chunk_shape
        self.chunk_voxel_size = chunk_voxel_size
        self.min_chunk_occupancy = float(min_chunk_occupancy)
        self.max_chunk_attempts = int(max_chunk_attempts)
        self.chunk_origin = chunk_origin
        self.camera_chunk_min_area_ratio = float(camera_chunk_min_area_ratio)
        downsample_factor = float(image_downsample_factor)
        if downsample_factor < 1.0 or not downsample_factor.is_integer():
            raise ValueError(
                "`image_downsample_factor` must be an integer >= 1, "
                f"got {image_downsample_factor}."
            )
        self.image_downsample_factor = int(downsample_factor)

    def setup(self, stage: Optional[str] = None) -> None:
        if self.train_dataset is not None and self.val_dataset is not None:
            return

        train_files, val_files = self.build_file_lists()

        if self.overfit_scenes > 0:
            if len(train_files) < self.overfit_scenes:
                raise ValueError(
                    f"Requested to overfit on {self.overfit_scenes} scenes, "
                    f"but only {len(train_files)} training scenes are available."
                )
            overfit_subset = train_files[: self.overfit_scenes]
            if self.verbose and self.overfit_scenes <= 10:
                lengths = []
                for path in overfit_subset:
                    data = safe_torch_load(path)
                    if (
                        isinstance(data, dict)
                        and "coords" in data
                        and "feature_ids" in data
                    ):
                        feature_ids = data["feature_ids"]
                        if feature_ids.dim() == 1:
                            num_features = 1
                        else:
                            num_features = int(feature_ids.shape[1])
                        if getattr(self, "dense_chunks", False):
                            chunk_shape = getattr(self, "chunk_shape", None)
                            if isinstance(chunk_shape, int):
                                chunk_shape = [chunk_shape] * 3
                            if chunk_shape:
                                num_voxels = int(
                                    chunk_shape[0] * chunk_shape[1] * chunk_shape[2]
                                )
                                lengths.append(num_voxels * num_features)
                            else:
                                lengths.append(int(data["coords"].shape[0]))
                        else:
                            num_points = int(data["coords"].shape[0])
                            num_position_tokens = getattr(
                                self, "num_position_tokens", 0
                            )
                            if num_position_tokens:
                                lengths.append(
                                    num_points * (num_position_tokens + num_features)
                                )
                            else:
                                lengths.append(num_points)
                    elif torch.is_tensor(data):
                        lengths.append(int(data.shape[0]))
                if lengths:
                    avg_len = sum(lengths) / len(lengths)
                    print(
                        "INFO: Overfit token lengths "
                        f"(min/mean/max): {min(lengths)}/{avg_len:.1f}/{max(lengths)}"
                    )
            train_files = self._expand_overfit_subset(overfit_subset)

            val_files = list(overfit_subset)
            min_val_files = self.overfit_min_val_scenes
            if len(val_files) < min_val_files and len(val_files) > 0:
                # TODO this is pretty ugly, should use same helper as above
                repeat = (min_val_files + len(val_files) - 1) // len(val_files)
                print(
                    f"INFO: Expanding validation set to at least {min_val_files} scenes"
                    f" - repeating {repeat} times."
                )
                val_files *= repeat

        self.train_dataset = self.create_dataset("train", train_files)
        self.val_dataset = self.create_dataset("val", val_files)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Call `setup` before requesting a dataloader.")
        kwargs = dict(self.dataloader_kwargs_train)
        if isinstance(self.train_dataset, PreprocessedDataset):
            coll_fn = collate_token_sequences
        else:
            coll_fn = partial(collate_fn, max_points=self.max_batch_points)
        return torch.utils.data.DataLoader(
            self.train_dataset, collate_fn=coll_fn, **kwargs
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("Call `setup` before requesting a dataloader.")
        kwargs = dict(self.dataloader_kwargs_val)
        if isinstance(self.val_dataset, PreprocessedDataset):
            coll_fn = collate_token_sequences
        else:
            coll_fn = partial(collate_fn, max_points=self.max_batch_points)
        return torch.utils.data.DataLoader(
            self.val_dataset, collate_fn=coll_fn, **kwargs
        )

    # ------------------------------------------------------------------ #
    # Hooks for subclasses
    # ------------------------------------------------------------------ #
    def build_file_lists(self) -> Tuple[List[str], List[str]]:
        splitter = self.get_splitter()
        if splitter is not None:
            return splitter(**self.get_splitter_kwargs())
        return self.custom_data_split()

    def get_splitter(self) -> Optional[Callable[..., Tuple[List[str], List[str]]]]:
        """
        Provide a callable that splits the dataset into train/val file lists.
        Override together with :meth:`get_splitter_kwargs`. Return ``None`` to
        fall back to :meth:`custom_data_split`.
        """

        return self.splitter

    def get_splitter_kwargs(self) -> Dict:
        """
        Keyword arguments forwarded to the splitter callable.
        Subclasses should override to match the expected signature.
        """

        raise NotImplementedError(
            "Override `get_splitter_kwargs` or provide a custom implementation of "
            "`build_file_lists`."
        )

    def custom_data_split(self) -> Tuple[List[str], List[str]]:
        """
        Subclasses can override when they require bespoke train/val splits.
        """

        raise NotImplementedError(
            "Override `custom_data_split` when no splitter callable is provided."
        )

    def create_dataset(
        self, split: str, file_list: Sequence[str]
    ) -> torch.utils.data.Dataset:
        dataset_cls = self.get_dataset_class(split)
        kwargs = self.get_dataset_kwargs(split, file_list)
        return dataset_cls(**kwargs)

    def get_dataset_class(self, split: str):
        """
        Return the dataset class to instantiate for a given split.
        """

        raise NotImplementedError(
            "Datamodules must define the dataset class to use via `get_dataset_class`."
        )

    def get_dataset_kwargs(self, split: str, file_list: Sequence[str]) -> Dict:
        """
        Common kwargs shared by geometry datasets. Subclasses can extend or modify
        this payload before instantiation.
        """

        return dict(
            paths=file_list,
            split=split,
            load_normals=self.load_normals,
            max_points=self.max_points,
            deterministic_sampling=self.deterministic_sampling,
            n_images=self.n_images,
            img_path=self.img_path,
            verbose=self.verbose,
            background_color=self.background_color,
            center_sample=self.center_sample,
            frustum_subsample=self.frustum_subsample,
            frustum_subsample_margin=self.frustum_subsample_margin,
            chunk_subsample=self.chunk_subsample,
            chunk_shape=self.chunk_shape,
            chunk_voxel_size=self.chunk_voxel_size,
            min_chunk_occupancy=self.min_chunk_occupancy,
            max_chunk_attempts=self.max_chunk_attempts,
            chunk_origin=self.chunk_origin,
            camera_chunk_min_area_ratio=self.camera_chunk_min_area_ratio,
            image_downsample_factor=self.image_downsample_factor,
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _expand_overfit_subset(self, train_files: List[str]) -> List[str]:
        if self.overfit_scenes <= 0:
            return train_files
        repeat_target = max(1, self.overfit_epoch_size // max(1, self.overfit_scenes))
        if self.verbose:
            print(
                "INFO: Repeating the first "
                f"{self.overfit_scenes} scenes {repeat_target} times "
                "to honour `overfit_epoch_size`."
            )

            # if less than 10 scenes are used for overfitting, print their names
            if self.overfit_scenes <= 10:
                print("INFO: Overfit scenes:")
                for scene in train_files:
                    print(f"  - {scene}")
        return train_files[: self.overfit_scenes] * repeat_target
