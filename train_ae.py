import os

import hydra
import lightning
import torch
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    RichModelSummary,
    TQDMProgressBar,
)
from lightning.pytorch.profilers import AdvancedProfiler, SimpleProfiler

from conf.dataclasses import GradientClippingMode, ImageKeys
from data.common import collate_fn
from data.vfront_dataset import VFrontDataModule
from model.gaussian_vqvae import GaussianVQVAE
from utils.augmentations import RandomZAxisDiscreteAugmentation
from utils.gaussian_vqvae_utils import split_batch_dict
from utils.inference import temporary_inference_mode
from utils.optim import (
    FreeCacheCallback,
    GradNormLoggingCallback,
    ModelCheckpointWithoutEquals,
    VRAMMonitorCallback,
)

# To enable tensor core usage
torch.set_float32_matmul_precision("high")


class RenderCallback(lightning.Callback):
    def __init__(
        self, every_n_epochs, num_samples_train, num_samples_val, output_dir=None
    ):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.num_samples_train = num_samples_train
        self.num_samples_val = num_samples_val

        self.output_dir = output_dir  # can be None at this point but needs to be set before the first call

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if (epoch + 1) % self.every_n_epochs != 0:
            # Render only every n epochs
            return

        if self.output_dir is None:
            # we do not sample
            return

        # self._render_samples(trainer, pl_module, epoch)
        world_size = trainer.world_size
        rank = trainer.global_rank

        available_val_samples = len(trainer.val_dataloaders.dataset)
        num_val_samples = min(self.num_samples_val, available_val_samples)
        total_samples = self.num_samples_train + num_val_samples

        # Temporarily enter evaluation mode with no grads and restore afterwards
        with temporary_inference_mode(pl_module, use_no_grad=True):
            # per rank indices, combining train and validation samples
            indices = range(rank, total_samples, world_size)
            for i in indices:
                # get the corresponding sample
                if i < self.num_samples_train:
                    train_dataset = trainer.train_dataloader.dataset

                    # choose train idx such that they are evenly distributed
                    train_idx = (i * len(train_dataset)) // self.num_samples_train

                    feature_dict = train_dataset[train_idx]
                    name = "train"
                else:
                    val_dataset = trainer.val_dataloaders.dataset
                    val_idx = (
                        (i - self.num_samples_train) * len(val_dataset)
                    ) // num_val_samples

                    feature_dict = val_dataset[val_idx]
                    name = "val"

                batched_dict = collate_fn([feature_dict])
                point_dict, image_dict = split_batch_dict(
                    batched_dict, device=pl_module.device
                )

                input_points = dict(point_dict)
                points, _ = pl_module.autoencoder(dict(input_points))

                if ImageKeys.IMAGES not in image_dict:
                    image_dict = dict(image_dict)
                    image_dict[ImageKeys.IMAGES] = [
                        image.detach()
                        for image in pl_module.rerendering_loss.render_images(
                            input_points, image_dict
                        )
                    ]

                output_dir = os.path.join(self.output_dir, f"epoch_{epoch}_{name}")
                pl_module.rerendering_loss(
                    dict(points),
                    image_dict,
                    calc_metrics=False,
                    use_lpips_loss=False,
                    use_ssim_loss=False,
                    output_dir=output_dir,
                    offset=i,
                )


@hydra.main(config_path="conf", config_name="vqvae", version_base=None)
def train(cfg):
    continue_mode = getattr(cfg.experiment, "continue_mode", "resume")
    if continue_mode not in {"resume", "weights_only"}:
        raise ValueError(
            f"Unknown experiment.continue_mode '{continue_mode}'. "
            "Expected 'resume' or 'weights_only'."
        )

    checkpoint_path = cfg.experiment.checkpoint_path
    if continue_mode == "weights_only" and checkpoint_path is None:
        raise ValueError(
            "experiment.checkpoint_path must be set when "
            "experiment.continue_mode=weights_only."
        )

    if checkpoint_path is not None and continue_mode == "weights_only":
        print(
            f"INFO: Loading VQ-VAE weights from {checkpoint_path} and resetting "
            "optimizer/scheduler/trainer state."
        )
        # pylint: disable-next=no-value-for-parameter
        ae = GaussianVQVAE.load_from_checkpoint(
            checkpoint_path,
            model_config=cfg.model,
            training_config=cfg.training,
        )
        fit_ckpt_path = None
    else:
        ae = GaussianVQVAE(cfg.model, cfg.training)
        fit_ckpt_path = checkpoint_path

    # Callbacks
    callbacks = []
    callbacks.append(
        RenderCallback(
            every_n_epochs=cfg.training.output.render_frequency,
            num_samples_train=cfg.training.output.num_samples_train,
            num_samples_val=cfg.training.output.num_samples_val,
        )
    )
    callbacks.append(
        ModelCheckpointWithoutEquals(
            monitor=(
                ae.val_loss_str if cfg.data.overfit.scenes == 0 else ae.train_loss_str
            ),
            mode="min",
            save_top_k=1,
            filename="loss_monitor_{epoch}_{step}",
            save_last=True,
        )
    )
    callbacks.append(TQDMProgressBar(refresh_rate=10))
    callbacks.append(LearningRateMonitor(logging_interval="step"))
    callbacks.append(
        RichModelSummary(max_depth=3)
    )  # so that we can see the individual autoencoders
    if (
        cfg.training.grad_norm_log.frequency is not None
        and cfg.training.grad_norm_log.frequency > 0
    ):
        callbacks.append(
            GradNormLoggingCallback(
                freq=cfg.training.grad_norm_log.frequency,
                l2_only=cfg.training.grad_norm_log.l2_only,
                log_module_l2=cfg.training.grad_norm_log.log_modules,
                module_max_depth=cfg.training.grad_norm_log.module_depth,
                log_class_l2=cfg.training.grad_norm_log.log_classes,
            )
        )

    # Trainer
    num_gpus = torch.cuda.device_count()
    strategy = "ddp" if num_gpus > 1 else "auto"  # force ddp or error

    # handle gradient clipping
    if cfg.training.gradient_clip.algorithm in [
        GradientClippingMode.VALUE,
        GradientClippingMode.NORM,
    ]:
        print(
            f"INFO: Using gradient clipping of type "
            f" {cfg.training.gradient_clip.algorithm}"
            f" with value {cfg.training.gradient_clip.val}"
        )
        gradient_clipping_kwargs = {
            "gradient_clip_algorithm": cfg.training.gradient_clip.algorithm,
            "gradient_clip_val": cfg.training.gradient_clip.val,
        }
    else:
        print("INFO: No gradient clipping applied")
        gradient_clipping_kwargs = {"gradient_clip_algorithm": None}

    # profiling
    if cfg.training.profiler.enabled:
        filename = f"profiler_{cfg.training.profiler.type}.txt"
        prof_dir = os.path.join(cfg.experiment.log_dir, cfg.experiment.name, "profile")
        if cfg.training.profiler.type == "advanced":
            print("INFO: Using advanced profiler")
            profiler = AdvancedProfiler(prof_dir, filename)
        elif cfg.training.profiler.type == "simple":
            print("INFO: Using simple profiler")
            profiler = SimpleProfiler(prof_dir, filename)
        else:
            raise ValueError(f"Unknown profiler type {cfg.training.profiler.type}")
        os.makedirs(prof_dir, exist_ok=True)
    else:
        # no profiler
        profiler = None

    # log device stats
    if cfg.training.device_stats_log.enabled:
        callbacks.append(
            VRAMMonitorCallback(
                cfg.training.device_stats_log.frequency,
                cfg.training.device_stats_log.per_device,
            )
        )

    # free cache callback
    if cfg.training.free_cache.enabled:
        callbacks.append(FreeCacheCallback(freq=cfg.training.free_cache.frequency))

    # fast dev run
    if cfg.training.fast_dev_run:
        # either bool or int
        if isinstance(cfg.training.fast_dev_run, bool):
            print("INFO: Using fast dev run with 1 batch")
            # set overfit to 1 batch
            cfg.data.overfit.scenes = 1
            cfg.data.overfit.epoch_size = 1
        else:
            print(f"INFO: Using fast dev run with {cfg.training.fast_dev_run} batches")
            # set overfit to n batches
            cfg.data.overfit.scenes = cfg.training.fast_dev_run
            cfg.data.overfit.epoch_size = cfg.training.fast_dev_run

    log_every_n_steps = getattr(cfg.training, "log_every_n_steps", 50)

    trainer = lightning.Trainer(
        default_root_dir=os.path.join(cfg.experiment.log_dir, cfg.experiment.name),
        max_epochs=cfg.training.max_epochs,
        accelerator="gpu",
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        devices=num_gpus,
        precision=cfg.training.precision,
        log_every_n_steps=min(
            log_every_n_steps,
            cfg.training.grad_norm_log.frequency,
            cfg.training.device_stats_log.frequency,
        ),
        strategy=strategy,
        callbacks=callbacks,
        profiler=profiler,
        fast_dev_run=cfg.training.fast_dev_run,
        num_sanity_val_steps=0,  # not compatible with ME pruning
        **gradient_clipping_kwargs,
    )

    # create dir for visual output
    if (
        cfg.training.output.num_samples_train == 0
        and cfg.training.output.num_samples_val == 0
        or cfg.training.fast_dev_run
    ):
        # no samples to render or fast dev run
        if trainer.is_global_zero:
            print("INFO: No samples to render, skipping visualization output.")
        callbacks[0].output_dir = None
    elif not cfg.training.fast_dev_run:
        output_dir = os.path.join(trainer.log_dir, "vis")
        if trainer.is_global_zero:
            print(f"INFO: Creating output directory for visualizations: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
        callbacks[0].output_dir = output_dir

    # Data Module
    # TODO just pass+store the data cfg to the module and dataset, they pick the values they need - much less boilerplate, minimal overhead - also assume all values are in the config, no need for getattr
    center_sample = bool(getattr(cfg.data, "center_sample", False))
    augmentations = None
    train_aug_cfg = getattr(cfg.data, "train_augmentation", None)
    if train_aug_cfg is not None and bool(getattr(train_aug_cfg, "enabled", False)):
        augmentations = [
            RandomZAxisDiscreteAugmentation(
                probability=float(getattr(train_aug_cfg, "probability", 1.0)),
                allow_flip=bool(getattr(train_aug_cfg, "allow_flip", False)),
            )
        ]
        if trainer.is_global_zero:
            print(
                "INFO: Enabled train-time z-axis discrete scene augmentation "
                f"(probability={getattr(train_aug_cfg, 'probability', 1.0)}, "
                f"allow_flip={getattr(train_aug_cfg, 'allow_flip', False)})."
            )

    dataset_name = str(cfg.data.dataset_name).lower()
    depth_cfg = getattr(cfg.data, "depth", None)
    depth_chunk_mask_cfg = (
        getattr(depth_cfg, "chunk_mask", None) if depth_cfg is not None else None
    )
    depth_camera_sampling_cfg = (
        getattr(depth_cfg, "camera_sampling", None) if depth_cfg is not None else None
    )
    vfront_depth_kwargs = dict(
        depth_enabled=(
            bool(getattr(depth_cfg, "enabled", False))
            if depth_cfg is not None
            else False
        ),
        depth_subdir=(
            str(getattr(depth_cfg, "subdir", "depth"))
            if depth_cfg is not None
            else "depth"
        ),
        depth_extension=(
            str(getattr(depth_cfg, "extension", ".exr"))
            if depth_cfg is not None
            else ".exr"
        ),
        depth_chunk_mask_enabled=(
            bool(getattr(depth_chunk_mask_cfg, "enabled", False))
            if depth_chunk_mask_cfg is not None
            else False
        ),
        depth_camera_sampling_enabled=(
            bool(getattr(depth_camera_sampling_cfg, "enabled", False))
            if depth_camera_sampling_cfg is not None
            else False
        ),
        depth_camera_sampling_probe_multiplier=(
            int(getattr(depth_camera_sampling_cfg, "probe_multiplier", 3))
            if depth_camera_sampling_cfg is not None
            else 3
        ),
        depth_camera_sampling_stride=(
            int(getattr(depth_camera_sampling_cfg, "depth_stat_stride", 8))
            if depth_camera_sampling_cfg is not None
            else 8
        ),
    )
    data_module_kwargs = dict(
        data_path=cfg.data.data_path,
        train_list_path=getattr(cfg.data, "train_split", None),
        val_list_path=getattr(cfg.data, "val_split", None),
        img_path=cfg.data.img_path,
        dataloader_kwargs={"batch_size": cfg.training.batch_scenes},
        deterministic_sampling=cfg.data.deterministic_sampling,
        overfit_scenes=cfg.data.overfit.scenes,
        overfit_epoch_size=cfg.data.overfit.epoch_size,
        overfit_min_val_scenes=cfg.data.overfit.min_val_scenes,
        max_points=getattr(cfg.data, "max_points", None),
        max_batch_points=cfg.training.max_batch_points,
        n_images=cfg.data.n_images,
        load_normals=getattr(cfg.data, "load_normals", False),
        verbose=trainer.is_global_zero,
        preload=cfg.data.preload,
        background_color=cfg.data.background_color,
        center_sample=center_sample,
        frustum_subsample=bool(getattr(cfg.data, "frustum_subsample", False)),
        frustum_subsample_margin=float(
            getattr(cfg.data, "frustum_subsample_margin", 0.0)
        ),
        chunk_subsample=bool(getattr(cfg.data, "chunk_subsample", False)),
        chunk_shape=getattr(cfg.data, "chunk_shape", None),
        chunk_voxel_size=getattr(
            cfg.data, "chunk_voxel_size", getattr(cfg.data, "voxel_size", None)
        ),
        min_chunk_occupancy=float(getattr(cfg.data, "min_chunk_occupancy", 0.0)),
        max_chunk_attempts=int(getattr(cfg.data, "max_chunk_attempts", 1)),
        chunk_origin=getattr(cfg.data, "chunk_origin", None),
        camera_chunk_min_area_ratio=float(
            getattr(cfg.data, "camera_chunk_min_area_ratio", 0.0)
        ),
        image_downsample_factor=int(getattr(cfg.data, "image_downsample_factor", 1)),
        augmentations=augmentations,
    )
    if dataset_name == "vfront":
        data_module = VFrontDataModule(
            transforms_path=cfg.data.transforms_path,
            gaussian_subpath=getattr(
                cfg.data,
                "gaussian_subpath",
                "v0.025_sigmoid_uniform_tanh/point_cloud/iteration_30000/ckpt.pth",
            ),
            sh_degree=getattr(cfg.data, "sh_degree", 0),
            voxel_size=getattr(cfg.data, "voxel_size", 0.025),
            **vfront_depth_kwargs,
            **data_module_kwargs,
        )
    elif dataset_name == "ase":
        # Import lazily to avoid requiring ASE dependencies for VFront runs.
        from data.ase_dataset import ASEDataModule

        data_module = ASEDataModule(
            transforms_path=cfg.data.transforms_path,
            gaussian_subpath=getattr(
                cfg.data, "gaussian_subpath", "ckpts/point_cloud_30000.ply"
            ),
            transforms_filename=getattr(
                cfg.data, "transforms_filename", "transforms_train.json"
            ),
            **data_module_kwargs,
        )
    elif dataset_name in {"spp", "spp_v2"}:
        from data.spp_dataset import SPPDataModule

        data_module = SPPDataModule(
            transforms_path=cfg.data.transforms_path,
            gaussian_subpath=getattr(
                cfg.data, "gaussian_subpath", "ckpts/point_cloud_30000.ply"
            ),
            **data_module_kwargs,
        )
    elif dataset_name == "photoshape":
        from data.photoshape_dataset import PhotoshapeDataModule

        data_module = PhotoshapeDataModule(
            transforms_path=cfg.data.transforms_path,
            gaussian_subpath=getattr(
                cfg.data,
                "gaussian_subpath",
                "point_cloud/iteration_30000/point_cloud.ply",
            ),
            sh_degree=getattr(cfg.data, "sh_degree", 0),
            transforms_filename=getattr(cfg.data, "transforms_filename", None),
            **data_module_kwargs,
        )
    else:
        raise ValueError(
            f"Unsupported dataset_name '{cfg.data.dataset_name}'. "
            "Expected 'vfront', 'ase', 'spp_v2', or 'photoshape'."
        )

    ae.set_background_color(cfg.data.background_color)
    trainer.fit(ae, datamodule=data_module, ckpt_path=fit_ckpt_path)


if __name__ == "__main__":
    # Seed
    lightning.seed_everything(0)

    train()  # pylint: disable=no-value-for-parameter
