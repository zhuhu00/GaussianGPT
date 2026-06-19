from contextlib import contextmanager, nullcontext
from functools import wraps
from typing import Callable, Iterator

import torch
import torch.nn as nn


@contextmanager
def temporary_inference_mode(
    module: nn.Module, use_no_grad: bool = True
) -> Iterator[nn.Module]:
    """
    Temporarily set a module (and all its submodules) to eval mode, optionally
    disabling gradients, and restore the exact prior train/eval state.
    """

    modules = list(module.modules())
    prior_states = [m.training for m in modules]

    try:
        module.eval()
        cm = torch.no_grad() if use_no_grad else nullcontext()
        with cm:
            yield module
    finally:
        for m, was_training in zip(modules, prior_states):
            m.train(was_training)


def inference_call(fn: Callable) -> Callable:
    """
    Decorator for nn.Module methods: runs in eval + no_grad and restores state.
    """

    @wraps(fn)
    def wrapper(self: nn.Module, *args, **kwargs):
        with temporary_inference_mode(self, use_no_grad=True):
            return fn(self, *args, **kwargs)

    return wrapper
