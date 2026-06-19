"""
Useful helpers for optimizing models with pytorch lightning.
"""

import math
from collections import defaultdict
from typing import Dict

import lightning
import torch
import torch.distributed as dist
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.optim.lr_scheduler import (
    LambdaLR,
    LinearLR,
    SequentialLR,
)


def get_decay_groups(named_parameters, weight_decay, verbose=False):
    """
    Splits named parameters by whether they require weight decay.
    Decay everything with at least 2D weights, don't decay otherwise.

    Implementation following:
    https://github.com/audi/MeshGPT/blob/main/model/nanogpt.py
    """

    param_dict = {pn: p for pn, p in named_parameters if p.requires_grad}

    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    no_decay_params = [p for n, p in param_dict.items() if p.dim() < 2]

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    if verbose:
        num_decay_params = sum(p.numel() for p in decay_params)
        num_no_decay_params = sum(p.numel() for p in no_decay_params)

        # print small info with #tensors, #params, wd for each group in table style
        print("INFO: Parameter groups for optimization:")
        print(f"{'Group':<20} {'#Tensors':<10} {'#Params':<15} {'Weight Decay':<15}")
        print(
            f"{'Decay':<20} {len(decay_params):<10} {num_decay_params:<15} {weight_decay:<15}"
        )
        print(
            f"{'No Decay':<20} {len(no_decay_params):<10} {num_no_decay_params:<15} {0.0:<15}"
        )

    return optim_groups


def get_cosine_annealing_with_warmup_scheduler(
    optimizer, total_steps, warmup_ratio=0.0, final_lr_frac=0.1
):
    """
    Returns a cosine annealing learning rate scheduler with warmup.
    Args:
        optimizer: The optimizer to schedule.
        total_steps: The total number of training steps (including warmup).
        warmup_ratio: Fraction of total steps used for warmup.
        final_lr_frac: Minimum learning rate as a fraction of the base LR.
    """
    num_iterations = int(total_steps)
    warmup_steps = round(warmup_ratio * num_iterations)
    cosine_iters = max(num_iterations - warmup_steps, 1)

    def cosine_lr_lambda(it):
        t = min(it, cosine_iters)
        cosine = 0.5 * (1.0 + math.cos(math.pi * t / cosine_iters))
        return final_lr_frac + (1.0 - final_lr_frac) * cosine

    cosine_schedule = LambdaLR(optimizer, lr_lambda=cosine_lr_lambda)

    if warmup_steps > 0:
        sched = {
            "scheduler": SequentialLR(
                optimizer,
                [
                    LinearLR(optimizer, start_factor=1e-7, total_iters=warmup_steps),
                    cosine_schedule,
                ],
                milestones=[warmup_steps],
            ),
            "interval": "step",
        }
    else:
        sched = {
            "scheduler": cosine_schedule,
            "interval": "step",
        }

    return sched


def get_constant_lr_scheduler(optimizer):
    """
    Returns a constant learning rate scheduler.
    """
    return {
        "scheduler": LambdaLR(optimizer, lr_lambda=lambda _: 1.0),
        "interval": "step",
    }


def get_linear_warmup_warmdown_scheduler(
    optimizer, total_steps, warmup_ratio, warmdown_ratio, final_lr_frac
):
    """
    Returns a linear warmup -> constant -> linear warmdown scheduler.
    """
    num_iterations = int(total_steps)
    warmup_iters = round(warmup_ratio * num_iterations)
    warmdown_iters = round(warmdown_ratio * num_iterations)

    def lr_lambda(it):
        if warmup_iters > 0 and it < warmup_iters:
            return (it + 1) / warmup_iters
        if warmdown_iters > 0 and it > num_iterations - warmdown_iters:
            progress = (num_iterations - it) / warmdown_iters
            return progress * 1.0 + (1 - progress) * final_lr_frac
        return 1.0

    return {
        "scheduler": LambdaLR(optimizer, lr_lambda=lr_lambda),
        "interval": "step",
    }


# gradient norm helpers
@torch.no_grad()
def grad_total_l2(model) -> float:
    """Global L2 norm over all available (non-None) gradients.
    - Dense grads: include all elements.
    - Sparse grads: include only stored (non-zero) entries.
    """
    device = next(model.parameters()).device
    sumsq = torch.zeros((), device=device)

    for p in model.parameters():
        g = p.grad
        if g is None:
            continue
        if g.is_sparse:
            vals = g.coalesce().values().float()
        else:
            vals = g.detach().float()

        sumsq += (vals * vals).sum()

    return sumsq.sqrt().item()


@torch.no_grad()
def grad_rms(model) -> float:
    """Root-mean-square of all available (non-None) gradients.
    - Dense grads: count all elements.
    - Sparse grads: count only stored (non-zero) entries.
    """
    device = next(model.parameters()).device
    sumsq = torch.zeros((), device=device)
    count = torch.zeros((), device=device)

    for p in model.parameters():
        g = p.grad
        if g is None:
            continue
        if g.is_sparse:
            vals = g.coalesce().values().float()
        else:
            vals = g.detach().float()

        sumsq += (vals * vals).sum()
        count += vals.numel()

    return (sumsq / count.clamp_min(1)).sqrt().item()


@torch.no_grad()
def grad_mean_abs(model) -> float:
    """Mean absolute gradient over all available (non-None) gradients.
    - Dense grads: count all elements.
    - Sparse grads: count only stored (non-zero) entries.
    """
    device = next(model.parameters()).device
    sumabs = torch.zeros((), device=device)
    count = torch.zeros((), device=device)

    for p in model.parameters():
        g = p.grad
        if g is None:
            continue
        if g.is_sparse:
            vals = g.coalesce().values().float()
        else:
            vals = g.detach().float()

        sumabs += vals.abs().sum()
        count += vals.numel()

    return (sumabs / count.clamp_min(1)).item()


@torch.no_grad()
def grad_module_l2(model, max_depth=None) -> Dict[str, float]:
    """L2 norm of gradients for each module in the model up to a certain depth."""
    module_grads = {}
    for name, module in model.named_modules():
        # if module has no parameters, skip it
        if not list(module.parameters()):
            continue
        # if its lower than max depth, skip it
        depth = name.count(".")
        if max_depth is not None and depth >= max_depth:
            continue
        # if module has no gradients, skip it
        if not any(p.grad is not None for p in module.parameters()):
            continue

        # add depth to name so its nicely sorted
        class_str = type(module).__name__
        log_name = f"{depth:02}_{class_str}_{name}"
        module_grads[log_name] = grad_total_l2(module)
    return module_grads


@torch.no_grad()
def grad_class_l2(model) -> Dict[str, float]:
    """L2 norm of gradients averaged over the class of the module."""
    class_grads = defaultdict(list)
    for _, module in model.named_modules():
        # if module has no parameters, skip it
        if not list(module.parameters()):
            continue
        # if module has no gradients, skip it
        if not any(p.grad is not None for p in module.parameters()):
            continue

        class_str = type(module).__name__
        class_grads[class_str].append(grad_total_l2(module))

    # average over all modules in the class
    return {k: sum(v) / len(v) for k, v in class_grads.items()}


class GradNormLoggingCallback(lightning.Callback):
    """
    Logs gradient norms during training.
    """

    def __init__(
        self,
        freq=1,
        l2_only=True,
        log_module_l2=False,
        module_max_depth=None,
        log_class_l2=False,
    ):
        super().__init__()
        self.freq = freq
        self.l2_only = l2_only

        self.log_module_l2 = log_module_l2
        self.module_max_depth = module_max_depth

        self.log_class_l2 = log_class_l2

        self.logging_kwargs = dict(
            rank_zero_only=True, on_step=True, on_epoch=False, sync_dist=False
        )

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        if (trainer.global_step + 1) % self.freq == 0:
            total = grad_total_l2(pl_module)
            rms = grad_rms(pl_module)
            mean1 = grad_mean_abs(pl_module)

            pl_module.log("grad_norm/total_l2", total, **self.logging_kwargs)

            if not self.l2_only:
                pl_module.log("grad_norm/rms", rms, **self.logging_kwargs)
                pl_module.log("grad_norm/mean_abs", mean1, **self.logging_kwargs)

            # more expensive in terms of compute and log size -> opt-in
            if self.log_module_l2:
                module_l2 = grad_module_l2(pl_module, self.module_max_depth)
                for name, l2 in module_l2.items():
                    pl_module.log(
                        f"grad_norm_module_l2/{name}", l2, **self.logging_kwargs
                    )

            if self.log_class_l2:
                class_l2 = grad_class_l2(pl_module)
                for name, l2 in class_l2.items():
                    pl_module.log(
                        f"grad_norm_class_l2/{name}", l2, **self.logging_kwargs
                    )


class ModelCheckpointWithoutEquals(ModelCheckpoint):
    """
    A ModelCheckpoint callback that does not use '=' in the checkpoint filenames.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.CHECKPOINT_EQUALS_CHAR = "_"

    @staticmethod
    def get_equals_char() -> str:
        return "_"


class VRAMMonitorCallback(lightning.Callback):
    def __init__(self, log_every_n_steps: int = 50, per_device: bool = False):
        """
        Small Callback that logs VRAM usage every n steps (in GB).
        """
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.per_device = per_device

    def on_before_backward(self, trainer, pl_module, loss):
        if self.log_every_n_steps <= 0:
            return
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        # only meaningful on CUDA
        device = pl_module.device
        if device.type != "cuda":
            return

        allocated = torch.cuda.memory_allocated(device) / (1024**3)
        reserved = torch.cuda.memory_reserved(device) / (1024**3)

        if not self.per_device or not dist.is_available() or not dist.is_initialized():
            # log total allocated and reserved VRAM

            pl_module.log(
                f"vram_allocated/{device}_gb",
                allocated,
                rank_zero_only=True,
                on_step=True,
                on_epoch=False,
                sync_dist=False,
            )
            pl_module.log(
                f"vram_reserved/{device}_gb",
                reserved,
                rank_zero_only=True,
                on_step=True,
                on_epoch=False,
                sync_dist=False,
            )
            return

        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            allocated_tensor = torch.tensor(
                [allocated], dtype=torch.float32, device=device
            )
            reserved_tensor = torch.tensor(
                [reserved], dtype=torch.float32, device=device
            )

            world_size = dist.get_world_size()
            gathered_allocated = [
                torch.zeros_like(allocated_tensor) for _ in range(world_size)
            ]
            gathered_reserved = [
                torch.zeros_like(reserved_tensor) for _ in range(world_size)
            ]
            dist.all_gather(gathered_allocated, allocated_tensor)
            dist.all_gather(gathered_reserved, reserved_tensor)

            if not trainer.is_global_zero:
                return

            logging_params = dict(
                rank_zero_only=True, on_step=True, on_epoch=False, sync_dist=False
            )

            for rank in range(world_size):
                device_str = f"cuda:{rank}"
                pl_module.log(
                    f"vram_allocated/{device_str}_gb",
                    gathered_allocated[rank].item(),
                    **logging_params,
                )
                pl_module.log(
                    f"vram_reserved/{device_str}_gb",
                    gathered_reserved[rank].item(),
                    **logging_params,
                )
            return


class FreeCacheCallback(lightning.Callback):
    def __init__(self, freq: int = 1):
        """
        Callback that frees the CUDA cache after every n steps.
        """
        super().__init__()
        self.freq = freq

    def on_after_backward(self, trainer, pl_module):
        if self.freq <= 0:
            return
        if trainer.global_step % self.freq != 0:
            return

        torch.cuda.empty_cache()
