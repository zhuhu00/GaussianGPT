import os
from statistics import mean
from typing import Callable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)

from conf.dataclasses import GaussianFeatures, ImageKeys, PCKeys
from utils.render import GaussianScene, render

from .loss_utils import lpips, ssim
from .vgg_loss import VggLoss


def _as_mask(
    mask: Optional[torch.Tensor], like: torch.Tensor
) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    mask_t = mask.to(device=like.device)
    if mask_t.ndim == 3:
        mask_t = mask_t.unsqueeze(1)
    if mask_t.ndim != 4:
        raise ValueError(f"Mask must have shape (N, 1, H, W), got {mask_t.shape}.")
    if mask_t.shape[0] != like.shape[0] or mask_t.shape[-2:] != like.shape[-2:]:
        raise ValueError(
            "Mask shape must match rendered image shape on (N, H, W), got "
            f"{mask_t.shape} vs {like.shape}."
        )
    if mask_t.dtype != torch.bool:
        mask_t = mask_t > 0.5
    return mask_t


def _expand_mask(mask: torch.Tensor, channels: int) -> torch.Tensor:
    if mask.shape[1] == channels:
        return mask
    if mask.shape[1] == 1:
        return mask.expand(-1, channels, -1, -1)
    raise ValueError(
        f"Mask channels ({mask.shape[1]}) do not match tensor channels ({channels})."
    )


def _masked_mean(values: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return values.mean()
    expanded = _expand_mask(mask, values.shape[1]).to(dtype=values.dtype)
    denom = expanded.sum().clamp_min(1.0)
    return (values * expanded).sum() / denom


def _masked_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    return _masked_mean(torch.abs(prediction - target), mask)


def _masked_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor],
    beta: float = 1.0,
) -> torch.Tensor:
    loss = F.smooth_l1_loss(prediction, target, reduction="none", beta=beta)
    return _masked_mean(loss, mask)


class RerenderingLoss(nn.Module):
    def __init__(
        self,
        device,
        scales_activation_fn: Callable = torch.exp,
        opacities_activation_fn: Callable = torch.sigmoid,
        background_color: str = "white",
    ):
        super(RerenderingLoss, self).__init__()

        self.device = device  # required for rendering anyways

        self.scales_activation_fn = scales_activation_fn
        self.opacities_activation_fn = opacities_activation_fn
        self.set_background_color(background_color)

        self.vgg_loss_f: Optional[VggLoss] = None

        self.data_range = (0.0, 1.0)
        self.psnr_f = PeakSignalNoiseRatio(self.data_range, reduction="none").to(
            self.device
        )
        self.ssim_f = StructuralSimilarityIndexMeasure(data_range=self.data_range).to(
            self.device
        )
        # self.lpips_f = LearnedPerceptualImagePatchSimilarity(normalize=True).to(
        #     self.device
        # )
        self.lpips_f = None

    def lazy_lpips_f(self):
        if self.lpips_f is None:
            print(f"INFO: Loading LPIPS loss on device {self.device}.")
            self.lpips_f = LearnedPerceptualImagePatchSimilarity(normalize=True).to(
                self.device
            )
        return self.lpips_f

    def set_background_color(self, background_color: str) -> None:
        if background_color not in ("white", "black"):
            raise ValueError(
                f"Unsupported background color '{background_color}'. Expected 'white' or 'black'."
            )
        self.background_color = background_color

    def render_images(
        self,
        to_render: List[GaussianScene] | dict,
        image_dict: dict,
        render_mode: str = "RGB",
    ) -> List[torch.Tensor]:
        if ImageKeys.CAMERAS_R not in image_dict:
            raise KeyError(
                f"Missing {ImageKeys.CAMERAS_R} in image_dict; cannot render targets."
            )

        def fov2focal(fov, size):
            return 0.5 * size / torch.tan(0.5 * fov)

        with torch.amp.autocast_mode.autocast(self.device.type, enabled=False):
            B = len(image_dict[ImageKeys.CAMERAS_R])
            rendered_batches = []

            for i in range(B):
                n_images = int(image_dict[ImageKeys.CAMERAS_R][i].shape[0])
                H = int(image_dict[ImageKeys.CAMERAS_H][i][0].item())
                W = int(image_dict[ImageKeys.CAMERAS_W][i][0].item())

                focal_x = fov2focal(
                    image_dict[ImageKeys.CAMERAS_FOVX][i].to(
                        device=self.device, dtype=torch.float32
                    ),
                    float(W),
                )
                focal_y = fov2focal(
                    image_dict[ImageKeys.CAMERAS_FOVY][i].to(
                        device=self.device, dtype=torch.float32
                    ),
                    float(H),
                )
                cx = image_dict[ImageKeys.CAMERAS_CX][i].to(
                    device=self.device, dtype=torch.float32
                )
                cy = image_dict[ImageKeys.CAMERAS_CY][i].to(
                    device=self.device, dtype=torch.float32
                )

                intrinsics = torch.zeros(
                    (n_images, 3, 3), device=self.device, dtype=torch.float32
                )
                intrinsics[:, 0, 0] = focal_x
                intrinsics[:, 1, 1] = focal_y
                intrinsics[:, 0, 2] = cx
                intrinsics[:, 1, 2] = cy
                intrinsics[:, 2, 2] = 1.0

                view_matrices = (
                    torch.eye(4, device=self.device, dtype=torch.float32)
                    .unsqueeze(0)
                    .repeat(n_images, 1, 1)
                )
                view_matrices[:, :3, :3] = (
                    image_dict[ImageKeys.CAMERAS_R][i]
                    .to(device=self.device, dtype=torch.float32)
                    .transpose(1, 2)
                )
                view_matrices[:, :3, 3] = image_dict[ImageKeys.CAMERAS_T][i].to(
                    device=self.device, dtype=torch.float32
                )

                if isinstance(to_render, list):
                    to_render_scene = to_render[i].to(self.device, dtype=torch.float32)
                else:
                    batch_i = to_render[PCKeys.BATCH] == i
                    to_render_scene = GaussianScene(
                        means=to_render[GaussianFeatures.COORDS][batch_i],
                        opacities=to_render[GaussianFeatures.OPACITIES][batch_i],
                        sh0=to_render[GaussianFeatures.SH0][batch_i],
                        sh=(
                            to_render[PCKeys.SH][batch_i]
                            if PCKeys.SH in to_render
                            else None
                        ),
                        scales=to_render[GaussianFeatures.SCALES][batch_i],
                        quats=to_render[GaussianFeatures.QUATS][batch_i],
                    ).to(self.device, dtype=torch.float32)

                rendered_preds, _ = render(
                    to_render_scene,
                    view_matrices,
                    intrinsics,
                    (W, H),
                    scales_activation_fn=self.scales_activation_fn,
                    opacities_activation_fn=self.opacities_activation_fn,
                    background_color=self.background_color,
                    render_mode=render_mode,
                )
                rendered_batches.append(rendered_preds)

            return rendered_batches

    def forward(
        self,
        to_render: List[GaussianScene] | dict,
        image_dict: dict,
        calc_metrics: bool = False,
        use_lpips_loss: bool = False,
        use_ssim_loss: bool = False,
        use_vgg_loss: bool = False,
        output_dir: Optional[str] = None,
        offset: int = 0,
        depth_weight: float = 0.0,
    ):
        with torch.amp.autocast_mode.autocast(self.device.type, enabled=False):
            if ImageKeys.IMAGES not in image_dict:
                raise KeyError(
                    f"Missing {ImageKeys.IMAGES} in image_dict; expected supervision images."
                )
            need_depth_loss = depth_weight > 0.0
            if need_depth_loss and ImageKeys.DEPTHS not in image_dict:
                raise KeyError(
                    f"Missing {ImageKeys.DEPTHS} in image_dict while depth_weight>0."
                )
            if output_dir is not None:
                img_dir = os.path.join(output_dir, "img")
                tar_dir = os.path.join(output_dir, "tar")
                os.makedirs(img_dir, exist_ok=True)
                os.makedirs(tar_dir, exist_ok=True)
                tar_masked_dir = None

            # infer batch size from images, most stable option
            B = len(image_dict[ImageKeys.IMAGES])
            rendered_preds_batch = self.render_images(
                to_render,
                image_dict,
                render_mode="RGB+ED" if need_depth_loss else "RGB",
            )

            l1_loss_coll = torch.zeros((), device=self.device, dtype=torch.float32)
            lpips_loss_coll = torch.zeros((), device=self.device, dtype=torch.float32)
            ssim_loss_coll = torch.zeros((), device=self.device, dtype=torch.float32)
            vgg_loss_coll = torch.zeros((), device=self.device, dtype=torch.float32)
            depth_loss_coll = torch.zeros((), device=self.device, dtype=torch.float32)
            mask_coverage_coll = torch.zeros(
                (), device=self.device, dtype=torch.float32
            )
            num_masked_batches = 0

            if calc_metrics:
                all_psnrs = []
                total_ssim = 0.0
                total_lpips = 0.0

            for i in range(B):
                rendered_preds_full = rendered_preds_batch[i]
                if rendered_preds_full.shape[1] < 3:
                    raise ValueError(
                        f"Expected at least 3 rendered channels, got {rendered_preds_full.shape[1]}."
                    )
                rendered_preds = rendered_preds_full[:, :3]
                targets = image_dict["images"][i].to(
                    device=self.device, dtype=torch.float32
                )
                loss_mask = None
                if ImageKeys.LOSS_MASKS in image_dict:
                    loss_mask = _as_mask(
                        image_dict[ImageKeys.LOSS_MASKS][i], rendered_preds
                    )
                    mask_coverage_coll += loss_mask.to(torch.float32).mean()
                    num_masked_batches += 1

                masked_preds = rendered_preds
                masked_targets = targets
                if loss_mask is not None:
                    rgb_mask = _expand_mask(loss_mask, rendered_preds.shape[1]).to(
                        dtype=rendered_preds.dtype
                    )
                    masked_preds = rendered_preds * rgb_mask
                    masked_targets = targets * rgb_mask

                # 2b. save the rendered images if output_dir is specified
                if output_dir is not None:
                    if loss_mask is not None and tar_masked_dir is None:
                        tar_masked_dir = os.path.join(output_dir, "tar_masked")
                        os.makedirs(tar_masked_dir, exist_ok=True)

                    for k, (rendering, target, masked_target) in enumerate(
                        zip(rendered_preds, targets, masked_targets)
                    ):
                        cam_idx = image_dict[ImageKeys.CAMERAS_IDXS][i][k]
                        img_path = os.path.join(
                            img_dir, f"{i + offset}_cam_{cam_idx}.png"
                        )
                        target_path = os.path.join(
                            tar_dir, f"{i + offset}_cam_{cam_idx}.png"
                        )

                        torchvision.utils.save_image(rendering, img_path)
                        torchvision.utils.save_image(target, target_path)
                        if loss_mask is not None and tar_masked_dir is not None:
                            masked_target_path = os.path.join(
                                tar_masked_dir, f"{i + offset}_cam_{cam_idx}.png"
                            )
                            keep_mask = _expand_mask(
                                loss_mask[k : k + 1], masked_target.shape[0]
                            )[0]
                            masked_highlight = torch.zeros_like(masked_target)
                            masked_highlight[min(1, masked_highlight.shape[0] - 1)] = (
                                1.0
                            )
                            masked_target_vis = torch.where(
                                keep_mask,
                                masked_target,
                                masked_highlight,
                            )
                            torchvision.utils.save_image(
                                masked_target_vis, masked_target_path
                            )

                # 3. compute the losses
                l1_loss_coll += _masked_l1(rendered_preds, targets, loss_mask)
                if use_lpips_loss:
                    lpips_loss_coll += lpips(masked_preds, masked_targets)
                if use_ssim_loss:
                    ssim_loss_coll += 1.0 - ssim(masked_preds, masked_targets)
                if use_vgg_loss:
                    if self.vgg_loss_f is None:
                        print(f"INFO: Loading VGG loss on device {self.device}.")
                        self.vgg_loss_f = VggLoss(
                            device=self.device, resize=True
                        ).eval()
                    vgg_loss_coll += self.vgg_loss_f(masked_preds, masked_targets)

                if need_depth_loss:
                    if rendered_preds_full.shape[1] < 4:
                        raise ValueError(
                            "Depth guidance expects render_mode='RGB+ED' to return "
                            "an expected-depth channel in the last output channel."
                        )
                    pred_depth = rendered_preds_full[:, -1:, :, :]
                    target_depth = image_dict[ImageKeys.DEPTHS][i].to(
                        device=self.device, dtype=torch.float32
                    )
                    if pred_depth.shape != target_depth.shape:
                        raise ValueError(
                            "Predicted and target depth shapes must match, got "
                            f"{pred_depth.shape} and {target_depth.shape}."
                        )
                    depth_mask = (
                        torch.isfinite(pred_depth)
                        & torch.isfinite(target_depth)
                        & (target_depth > 0.0)
                    )
                    if loss_mask is not None:
                        depth_mask = depth_mask & _expand_mask(loss_mask, 1)
                    depth_loss_coll += _masked_smooth_l1(
                        pred_depth,
                        target_depth,
                        depth_mask,
                    )

                # 4. (optional) compute metrics if calc_metrics is True
                if calc_metrics:
                    pred_det = masked_preds.detach().clamp(0.0, 1.0).to(torch.float32)
                    target_det = masked_targets.clamp(0.0, 1.0).to(torch.float32)

                    psnrs = self.psnr_f(pred_det, target_det).cpu().tolist()
                    if isinstance(psnrs, float):
                        psnrs = [psnrs]
                    all_psnrs.extend(psnrs)
                    total_ssim += self.ssim_f(pred_det, target_det).cpu().item()
                    total_lpips += (
                        self.lazy_lpips_f()(pred_det, target_det).cpu().item()
                    )

            # normalize by batch size, scale by weight and return
            return_dict = {
                "l1_loss": l1_loss_coll / B,
                "lpips_loss": lpips_loss_coll / B,
                "ssim_loss": ssim_loss_coll / B,
                "vgg_loss": vgg_loss_coll / B,
                "depth_loss": depth_loss_coll / B,
            }
            if num_masked_batches > 0:
                return_dict["mask_coverage"] = mask_coverage_coll / num_masked_batches

            if calc_metrics:
                return_dict["psnr"] = mean(all_psnrs) if len(all_psnrs) > 0 else 0.0
                return_dict["ssim"] = total_ssim / B
                return_dict["lpips"] = total_lpips / B

            return return_dict
