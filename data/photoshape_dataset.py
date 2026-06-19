from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

from data.common import BaseSceneDataModule, BaseSceneDataset, PreprocessedDataset
from data.photoshape import get_default_data_split_photoshape, load_inria_data
from data.vfront import (
    readCamerasFromTransforms,
    safe_cv2_imread,
    safe_isdir,
    safe_isfile,
)
from utils.transforms import normalize_and_standardize_quaternion


class PhotoshapeGaussianDataset(BaseSceneDataset):
    def __init__(
        self,
        paths: Sequence[str],
        split: str,
        transforms_root: Optional[str],
        transforms_filename: Optional[str] = None,
        gaussian_subpath: str = "point_cloud/iteration_30000/point_cloud.ply",
        sh_degree: int = 0,
        **kwargs,
    ):
        # Must be set before BaseSceneDataset.__init__ because preload may call load_gaussians.
        self.transforms_root = transforms_root
        self.transforms_filename = transforms_filename
        self.gaussian_subpath = gaussian_subpath
        self.sh_degree = sh_degree
        super().__init__(
            paths,
            split=split,
            **kwargs,
        )

    def load_gaussians(self, path: str) -> Dict[str, torch.Tensor]:
        ply_path = os.path.join(path, self.gaussian_subpath)
        loaded = load_inria_data(ply_path, sh_degree=self.sh_degree)
        means, sh0, opacities, scales, quats = loaded[:5]
        quats = normalize_and_standardize_quaternion(quats)
        sample = {
            "coords": means,
            "sh0": sh0,
            "opacities": opacities,
            "scales": scales,
            "quats": quats,
        }
        if len(loaded) > 5:
            sample["sh"] = loaded[5]
        return sample

    def load_images(
        self, path: str, item_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        if self.n_images <= 0:
            return {}
        if self.transforms_root is None:
            raise ValueError("Photoshape requires `transforms_root` when n_images > 0.")
        if self.img_path is None:
            raise ValueError("Photoshape requires `img_path` when n_images > 0.")

        import cv2
        from skimage import color

        scene_id = self.get_scene_id(path)
        images_path = super()._resolve_image_path(path)
        transforms_path = os.path.join(self.transforms_root, scene_id)
        if not safe_isdir(transforms_path):
            raise FileNotFoundError(
                f"Transforms directory {transforms_path} not found for scene {scene_id}."
            )

        transforms_name = self._resolve_transforms_filename(transforms_path)

        cam_params = readCamerasFromTransforms(
            transforms_path,
            transforms_name,
            images_path,
            extension=".png",
            rotate_y_up=True,
        )

        if len(cam_params) == 0:
            raise RuntimeError(f"No camera parameters found for scene {scene_id}.")

        candidate_indices = np.arange(len(cam_params), dtype=np.int64)
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
        image_paths = []
        for cam_idx in camera_idxs:
            cam = cam_params[cam_idx]
            img = safe_cv2_imread(cam.image_path, cv2.IMREAD_UNCHANGED)
            if img.ndim != 3:
                raise RuntimeError(
                    f"Unsupported image shape {img.shape} for {cam.image_path}."
                )
            if img.shape[2] == 4:
                rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA).astype(np.float32) / 255.0
                rgb = color.rgba2rgb(rgba, background=self._background_rgb)
            else:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            frame = torch.from_numpy(np.moveaxis(rgb, -1, 0)).to(torch.float32)
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
            "camera_idxs": torch.tensor(camera_idxs, dtype=torch.long),
            "cameras_image_path": image_paths,
        }

    def _resolve_transforms_filename(self, transforms_path: str) -> str:
        candidate_names: List[str] = []
        if self.transforms_filename:
            candidate_names.append(self.transforms_filename)
        candidate_names.extend(
            [
                f"transforms_{self.split}.json",
                "transforms_train.json",
                "transforms_val.json",
                "transforms_test.json",
                "transforms.json",
            ]
        )

        unique_names: List[str] = []
        for name in candidate_names:
            if name not in unique_names:
                unique_names.append(name)

        for name in unique_names:
            if safe_isfile(os.path.join(transforms_path, name)):
                return name

        raise FileNotFoundError(
            f"No transforms file found in {transforms_path}. " f"Tried: {unique_names}."
        )


class PhotoshapeDataModule(BaseSceneDataModule):
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
        gaussian_subpath: str = "point_cloud/iteration_30000/point_cloud.ply",
        transforms_filename: Optional[str] = None,
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
            splitter=get_default_data_split_photoshape,
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
        self.sh_degree = int(sh_degree)
        self.gaussian_subpath = gaussian_subpath
        self.transforms_filename = transforms_filename
        self._logged_sh_augmentation_disable = False

    def get_splitter_kwargs(self) -> Dict:
        return dict(
            data_path=self.data_path,
            train_list_path=self.train_list_path,
            val_list_path=self.val_list_path,
            gaussian_subpath=self.gaussian_subpath,
            verbose=self.verbose,
        )

    def get_dataset_class(self, split: str):
        return PhotoshapeGaussianDataset

    def get_dataset_kwargs(self, split: str, file_list: Sequence[str]) -> Dict:
        kwargs = super().get_dataset_kwargs(split, file_list)
        augmentations = None
        if split == "train":
            if self.sh_degree > 0:
                augmentations = []
                if (
                    self.verbose
                    and len(self.augmentations) > 0
                    and not self._logged_sh_augmentation_disable
                ):
                    print(
                        "INFO: Disabling Photoshape train augmentations because sh_degree > 0."
                    )
                    self._logged_sh_augmentation_disable = True
            elif self.n_images > 0:
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
            gaussian_subpath=self.gaussian_subpath,
            augmentations=augmentations,
            preload=self.preload,
            sh_degree=self.sh_degree,
        )
        return kwargs


class PhotoshapePreprocessedDataModule(BaseSceneDataModule):
    def __init__(
        self,
        *,
        data_path: str,
        train_list_path: Optional[str] = None,
        val_list_path: Optional[str] = None,
        dataloader_kwargs: Optional[Dict] = None,
        overfit_scenes: int = 0,
        overfit_epoch_size: int = 1000,
        verbose: bool = False,
        augmentations: Optional[Iterable] = None,
        preprocessed_subpath: str = "sample.pt",
        background_color: str = "white",
    ):
        super().__init__(
            data_path=data_path,
            train_list_path=train_list_path,
            val_list_path=val_list_path,
            img_path=None,
            overfit_scenes=overfit_scenes,
            overfit_epoch_size=overfit_epoch_size,
            max_points=None,
            max_batch_points=None,
            n_images=0,
            load_normals=False,
            dataloader_kwargs=dataloader_kwargs,
            verbose=verbose,
            splitter=get_default_data_split_photoshape,
            background_color=background_color,
        )

        self.augmentations = list(augmentations or [])
        self.preprocessed_subpath = preprocessed_subpath

    def get_splitter_kwargs(self) -> Dict:
        return dict(
            data_path=self.data_path,
            train_list_path=self.train_list_path,
            val_list_path=self.val_list_path,
            gaussian_subpath=self.preprocessed_subpath,
            verbose=self.verbose,
        )

    def get_dataset_class(self, split: str):
        return PreprocessedDataset

    def get_dataset_kwargs(self, split: str, file_list: Sequence[str]) -> Dict:
        augmentations = self.augmentations if split == "train" else None
        return dict(file_list=file_list, augmentations=augmentations)
