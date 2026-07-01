"""
Main training script for training a GaussianGPT model.
"""

import os
import re
from pathlib import Path
from typing import Optional

import hydra
import lightning
import torch
from hydra import compose
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    RichModelSummary,
    TQDMProgressBar,
)
from lightning.pytorch.profilers import AdvancedProfiler, SimpleProfiler
from omegaconf import OmegaConf

from conf.dataclasses import GradientClippingMode
from data.vfront_dataset import VFrontPreprocessedDataModule
from generate_chunks import sample_and_render
from model.gaussian_gpt import GaussianGPT
from utils.optim import (
    FreeCacheCallback,
    GradNormLoggingCallback,
    ModelCheckpointWithoutEquals,
    VRAMMonitorCallback,
)

# Enable tensor core usage
torch.set_float32_matmul_precision("high")

_VERSION_PATTERN = re.compile(r"^version_(\d+)$")
_EPOCH_PATTERN = re.compile(r"(?:^|_)epoch_(\d+)(?:_|$|\.)")
_STEP_PATTERN = re.compile(r"(?:^|_)step_(\d+)(?:_|$|\.)")


def _version_key(path: Path) -> Optional[int]:
    match = _VERSION_PATTERN.match(path.name)
    if match is None:
        return None
    return int(match.group(1))


def _extract_epoch_step(ckpt_name: str) -> tuple[int, int]:
    epoch_match = _EPOCH_PATTERN.search(ckpt_name)
    step_match = _STEP_PATTERN.search(ckpt_name)
    epoch = int(epoch_match.group(1)) if epoch_match is not None else -1
    step = int(step_match.group(1)) if step_match is not None else -1
    return epoch, step


_MAX_CHECKPOINT_VALIDATION_ATTEMPTS = 5


def _pick_checkpoint_candidates(checkpoint_dir: Path) -> list[Path]:
    """All checkpoints in priority order (most-recent first)."""
    if not checkpoint_dir.is_dir():
        return []

    candidates: list[Path] = []
    last_checkpoint = checkpoint_dir / "last.ckpt"
    if last_checkpoint.is_file():
        candidates.append(last_checkpoint)

    others = [
        path
        for path in checkpoint_dir.glob("*.ckpt")
        if path.is_file() and path.name != "last.ckpt"
    ]

    def _sort_key(path: Path):
        epoch, step = _extract_epoch_step(path.name)
        return (epoch, step, path.stat().st_mtime)

    others.sort(key=_sort_key, reverse=True)
    candidates.extend(others)

    if last_checkpoint.is_symlink():
        try:
            last_target = last_checkpoint.resolve()
            candidates = [
                p
                for i, p in enumerate(candidates)
                if i == 0 or p.resolve() != last_target
            ]
        except OSError:
            pass

    return candidates


def _checkpoint_loads_ok(path: Path) -> bool:
    # NFS-truncated commits leave a zip with a missing central directory; torch.load
    # detects that immediately. weights_only=False is required for Lightning ckpts.
    try:
        torch.load(str(path), map_location="cpu", weights_only=False)
        return True
    except Exception as exc:  # noqa: BLE001
        print(
            f"WARNING: checkpoint integrity check failed for {path} "
            f"({type(exc).__name__}: {exc}); skipping."
        )
        return False


def _resolve_previous_run_checkpoint(cfg) -> Path:
    lightning_logs_dir = (
        Path(cfg.experiment.log_dir) / cfg.experiment.name / "lightning_logs"
    )
    if not lightning_logs_dir.is_dir():
        raise FileNotFoundError(
            "experiment.continue_run=true but no previous run directory was found at "
            f"{lightning_logs_dir}."
        )

    version_dirs = []
    for path in lightning_logs_dir.iterdir():
        if not path.is_dir():
            continue
        version = _version_key(path)
        if version is None:
            continue
        version_dirs.append((version, path))
    version_dirs.sort(key=lambda item: item[0], reverse=True)

    if not version_dirs:
        raise FileNotFoundError(
            "experiment.continue_run=true but no lightning_logs/version_* directories "
            f"were found in {lightning_logs_dir}."
        )

    attempts = 0
    skipped: list[Path] = []
    for _, version_dir in version_dirs:
        for candidate in _pick_checkpoint_candidates(version_dir / "checkpoints"):
            if attempts >= _MAX_CHECKPOINT_VALIDATION_ATTEMPTS:
                break
            attempts += 1
            if _checkpoint_loads_ok(candidate):
                if skipped:
                    print(
                        f"INFO: continue_run skipped {len(skipped)} unreadable "
                        f"checkpoint(s); resuming from {candidate}."
                    )
                return candidate
            skipped.append(candidate)
        if attempts >= _MAX_CHECKPOINT_VALIDATION_ATTEMPTS:
            break

    if skipped:
        raise FileNotFoundError(
            f"experiment.continue_run=true: tried {attempts} checkpoint(s) and all "
            f"failed to load. Last attempted: {skipped[-1]}."
        )
    raise FileNotFoundError(
        "experiment.continue_run=true but no .ckpt file was found in any "
        f"{lightning_logs_dir}/version_*/checkpoints directory."
    )


class EvaluateCallback(lightning.Callback):
    """
    Callback that runs inline evaluation at the end of each epoch.
    Expects a specific format for checkpoint names.
    """

    def __init__(
        self,
        every_n_epochs: int,
        num_samples: int = 16,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        dataset: str = "vfront",
        background_color: str = "white",
        data_config: Optional[str] = None,
        vqvae_checkpoint: Optional[str] = None,
        max_length: Optional[int] = None,
        batch_size: int = 4,
        store_samples: bool = False,
        data_cfg: Optional[dict] = None,
    ):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.num_samples = num_samples
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.dataset = dataset
        self.background_color = background_color
        self.max_length = max_length
        self.batch_size = batch_size
        self.store_samples = store_samples
        self.data_cfg = data_cfg

        self.data_config = data_config
        if self.data_config is None:
            raise ValueError(
                "Evaluation data config must be provided. Set"
                " `training.output.eval_data_config`."
            )

        if vqvae_checkpoint is None:
            raise ValueError("VQ-VAE checkpoint path is required for evaluation.")
        self.vqvae_checkpoint = vqvae_checkpoint

    def _run_inline(self, trainer, pl_module, epoch):
        if self.data_cfg is None:
            raise ValueError("Inline evaluation requires data_cfg.")
        model_was_training = pl_module.training
        try:
            world_size = max(1, int(getattr(trainer, "world_size", 1)))
            rank = int(trainer.global_rank)
            base_samples = self.num_samples // world_size
            extra_samples = self.num_samples % world_size
            rank_num_samples = base_samples + int(rank < extra_samples)

            if rank_num_samples <= 0:
                print(
                    f"INFO: Skipping inline evaluation on rank {rank} "
                    f"(assigned {rank_num_samples} samples)."
                )
                return

            rank_seed = (int.from_bytes(os.urandom(8), "big") + rank) % (2**31 - 1)
            torch.manual_seed(rank_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(rank_seed)

            output_directory = os.path.join(
                trainer.log_dir,
                "rendered_images",
                f"epoch_{epoch:04d}",
                f"rank_{rank:04d}",
            )
            print(
                f"INFO: Running inline evaluation on rank {rank}/{world_size} "
                f"with {rank_num_samples} samples (seed={rank_seed})."
            )
            pl_module.eval()
            sample_and_render(
                dataset=self.dataset,
                output_dir=Path(output_directory),
                num_samples=rank_num_samples,
                batch_size=self.batch_size,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                max_length=self.max_length,
                background_color=self.background_color,
                gif_frames=120,
                gif_fps=24,
                render_gifs=True,
                vqvae_checkpoint=self.vqvae_checkpoint,
                store_samples=self.store_samples,
                data_cfg=self.data_cfg,
                gpt=pl_module,
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"ERROR: Failed to run inline evaluation: {e}")
        finally:
            if model_was_training:
                pl_module.train()

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch

        if (epoch + 1) % self.every_n_epochs != 0:
            return

        # Keep all DDP ranks aligned around rank-0-only eval side effects.
        trainer.strategy.barrier()

        self._run_inline(trainer, pl_module, epoch)

        trainer.strategy.barrier()

        return super().on_train_epoch_end(trainer, pl_module)


@hydra.main(version_base=None, config_path="conf", config_name="gpt")
def train(cfg):
    """
    Setup for training a GaussianGPT model, training loop is handled by lightning.
    Takes parameters from the hydra config.
    """

    # Initialize the model using the centralized factory with ModelConfig
    if cfg.data.vqvae_path is not None:
        if cfg.model.vqvae.checkpoint_path is not None:
            if cfg.data.vqvae_path != cfg.model.vqvae.checkpoint_path:
                print(
                    "WARNING: VQVAE path in data config does not match model config."
                    " Using the one from the model config."
                )
        # use the VQVAE path from the data config if its not set in the model config
        cfg.model.vqvae.checkpoint_path = cfg.data.vqvae_path

    assert (
        cfg.model.vqvae.checkpoint_path is not None
    ), "VQVAE checkpoint must be provided"
    if getattr(cfg.data, "sort_latents", None) is not None:
        raise ValueError(
            "`data.sort_latents` is deprecated outside tokenization; use `model.chunk_order`."
        )

    chunk_shape = cfg.model.chunk_shape
    if isinstance(chunk_shape, int):
        chunk_shape = [chunk_shape] * 3
    if (
        not chunk_shape
        or len(chunk_shape) != 3
        or any(int(v) <= 0 for v in chunk_shape)
    ):
        raise ValueError(
            "GPT training requires chunked data. Set `model.chunk_shape` to 3 positive integers."
        )
    cfg.model.chunk_shape = [int(v) for v in chunk_shape]

    continue_mode = getattr(cfg.experiment, "continue_mode", "resume")
    if continue_mode not in {"resume", "weights_only"}:
        raise ValueError(
            f"Unknown experiment.continue_mode '{continue_mode}'. "
            "Expected 'resume' or 'weights_only'."
        )

    checkpoint_path = cfg.experiment.checkpoint_path
    continue_run = bool(getattr(cfg.experiment, "continue_run", False))
    if continue_run and checkpoint_path is not None:
        raise ValueError(
            "experiment.continue_run=true cannot be used together with "
            "experiment.checkpoint_path. Set only one of them."
        )
    if continue_run:
        resolved_checkpoint = _resolve_previous_run_checkpoint(cfg)
        version_name = resolved_checkpoint.parent.parent.name
        print(
            "INFO: Resolved checkpoint from previous run "
            f"({version_name}): {resolved_checkpoint}"
        )
        checkpoint_path = str(resolved_checkpoint)

    if continue_mode == "weights_only" and checkpoint_path is None:
        raise ValueError(
            "experiment.checkpoint_path must be set when "
            "experiment.continue_mode=weights_only."
        )

    if checkpoint_path is not None and continue_mode == "weights_only":
        print(
            f"INFO: Loading GPT weights from {checkpoint_path} and resetting "
            "optimizer/scheduler/trainer state."
        )
        # pylint: disable-next=no-value-for-parameter
        gpt = GaussianGPT.load_from_checkpoint(
            checkpoint_path,
            model_config=cfg.model,
            training_config=cfg.training,
        )
        fit_ckpt_path = None
    else:
        gpt = GaussianGPT(cfg.model, cfg.training)
        fit_ckpt_path = checkpoint_path

    gpt.vqvae.set_background_color(cfg.data.background_color)

    if cfg.model.dense_chunks:
        num_features = gpt.vqvae.autoencoder.vq.num_tokens
        dense_len = int(chunk_shape[0] * chunk_shape[1] * chunk_shape[2]) * num_features
        if cfg.data.max_tokens != dense_len:
            print(f"INFO: Overriding max_tokens to {dense_len} for dense chunks.")
        cfg.data.max_tokens = dense_len

    eval_data_config = cfg.training.output.eval_data_config
    if eval_data_config is None:
        raise ValueError("training.output.eval_data_config must be set for evaluation.")
    eval_cfg = compose(
        config_name="generate_chunks",
        overrides=[f"data={eval_data_config}"],
    )
    data_cfg = OmegaConf.to_container(eval_cfg.data, resolve=True)
    if not isinstance(data_cfg, dict):
        raise TypeError("Expected eval_cfg.data to convert to a dictionary.")
    if data_cfg.get("sort_latents") is not None:
        raise ValueError(
            "`data.sort_latents` is deprecated outside tokenization; use `model.chunk_order`."
        )

    num_gpus = torch.cuda.device_count()
    # Callbacks for Rendering and Evaluation
    callbacks = []
    checkpoint_frequency = (
        getattr(cfg.training.output, "checkpoint_frequency", None)
        or cfg.training.output.render_frequency
    )
    callbacks.append(
        ModelCheckpointWithoutEquals(
            every_n_epochs=checkpoint_frequency,
            monitor="epoch",
            mode="max",
            save_top_k=-1,
            filename="{epoch}",
        )
    )
    callbacks.append(
        EvaluateCallback(
            every_n_epochs=cfg.training.output.render_frequency,
            num_samples=cfg.training.output.num_samples,
            temperature=cfg.training.output.temperature,
            top_k=getattr(cfg.training.output, "top_k", None),
            top_p=getattr(cfg.training.output, "top_p", None),
            max_length=cfg.data.max_tokens,
            dataset=cfg.data.dataset_name,
            background_color=cfg.data.background_color,
            data_config=cfg.training.output.eval_data_config,
            vqvae_checkpoint=cfg.model.vqvae.checkpoint_path,
            batch_size=cfg.training.output.batch_size,
            store_samples=cfg.training.output.store_samples,
            data_cfg=data_cfg,
        )
    )

    # Callbacks for saving best checkpoints
    if cfg.data.overfit.scenes == 0:
        # not overfitting so we have a validations set
        callbacks.append(
            ModelCheckpointWithoutEquals(
                monitor="loss/val",
                mode="min",
                save_top_k=1,
                filename="val_loss_monitor_{epoch}_{step}",
            )
        )

    callbacks.append(
        ModelCheckpointWithoutEquals(
            monitor="loss/train",
            mode="min",
            save_top_k=1,
            filename="train_loss_monitor_{epoch}_{step}",
        )
    )

    # Callbacks for logging
    callbacks.append(TQDMProgressBar(refresh_rate=10))  # avoid cluttering the log
    callbacks.append(LearningRateMonitor(logging_interval="step"))
    callbacks.append(RichModelSummary(max_depth=2))
    callbacks.append(
        GradNormLoggingCallback(
            freq=cfg.training.grad_norm_log.frequency,
            l2_only=cfg.training.grad_norm_log.l2_only,
            log_module_l2=cfg.training.grad_norm_log.log_modules,
            module_max_depth=cfg.training.grad_norm_log.module_depth,
            log_class_l2=cfg.training.grad_norm_log.log_classes,
        )
    )

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

    log_every_n_steps = getattr(cfg.training, "log_every_n_steps", 50)
    if cfg.data.overfit.scenes > 0:
        log_every_n_steps = min(10, log_every_n_steps)

    # Trainer
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
        strategy="ddp" if num_gpus > 1 else "auto",
        callbacks=callbacks,
        profiler=profiler,
        fast_dev_run=cfg.training.fast_dev_run,
        **gradient_clipping_kwargs,
    )

    # Load tokenized dataset.
    dataset_name = str(cfg.data.dataset_name).lower()
    if dataset_name not in {"vfront", "ase", "photoshape", "spp", "spp_v2"}:
        raise ValueError(
            f"Unsupported dataset_name '{cfg.data.dataset_name}'. "
            "Expected 'vfront', 'ase', 'photoshape', 'spp', or 'spp_v2'."
        )

    preprocessed_kwargs = dict(
        data_path=cfg.data.data_path,
        train_list_path=getattr(cfg.data, "train_split", None),
        val_list_path=cfg.data.val_split,
        dataloader_kwargs={"batch_size": cfg.training.batch_scenes},
        overfit_scenes=cfg.data.overfit.scenes,
        overfit_epoch_size=cfg.data.overfit.epoch_size,
        verbose=trainer.global_rank == 0,
        background_color=cfg.data.background_color,
        num_position_tokens=cfg.model.num_position_tokens,
        position_vocab_size=cfg.model.position_vocab_size,
        codebook_size=gpt.vqvae.autoencoder.vq.codebook_size,
        shared=getattr(cfg.model, "shared_vocab", False),
        chunk_shape=getattr(cfg.model, "chunk_shape", None),
        dense_chunks=getattr(cfg.model, "dense_chunks", False),
        chunk_order=getattr(cfg.model, "chunk_order", "xyz"),
        min_chunk_occupancy=getattr(cfg.data, "min_chunk_occupancy", 0.0),
        max_chunk_attempts=getattr(cfg.data, "max_chunk_attempts", 1),
        chunk_origin=getattr(cfg.data, "chunk_origin", None),
        load_augmented_tokens=getattr(cfg.data, "load_augmented_tokens", False),
    )
    preprocessed_subpath = getattr(cfg.data, "preprocessed_subpath", None)
    if preprocessed_subpath is not None:
        preprocessed_kwargs["preprocessed_subpath"] = preprocessed_subpath
    ase_data_path = getattr(cfg.data, "ase_data_path", None)
    if ase_data_path is not None:
        preprocessed_kwargs["ase_data_path"] = ase_data_path
        preprocessed_kwargs["ase_train_list_path"] = getattr(
            cfg.data, "ase_train_split", None
        )
        preprocessed_kwargs["ase_val_list_path"] = getattr(
            cfg.data, "ase_val_split", None
        )
        ase_preprocessed_subpath = getattr(cfg.data, "ase_preprocessed_subpath", None)
        if ase_preprocessed_subpath is not None:
            preprocessed_kwargs["ase_preprocessed_subpath"] = ase_preprocessed_subpath
    data_module = VFrontPreprocessedDataModule(**preprocessed_kwargs)

    # Train the model
    trainer.fit(
        gpt,
        datamodule=data_module,
        ckpt_path=fit_ckpt_path,
    )


if __name__ == "__main__":
    # Seed
    lightning.seed_everything(0)

    train()  # pylint: disable=no-value-for-parameter
