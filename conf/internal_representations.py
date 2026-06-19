from typing import Callable, Optional

import torch
import torch.nn as nn

from utils.transforms import (
    inverse_sigmoid,
    normalize_and_standardize_quaternion,
    quaternion_to_rot6d,
    rgb2sh,
    rot6d_loss,
    rot6d_to_quaternion,
    sh2rgb,
)

"""
In this file we define transformations between representations used for Gaussian Splat features.
They allow to specify how features are represented to the model internally, and define the corresponding transformations.
Additionally, they allow to define custom loss functions.

Naming: <Method><Feature>Representation. For each feature we keep the active representation
plus one alternative; the active ones are wired in conf/model/vqvae_cnn.yaml.
"""


class InternalRepresentation:
    """
    Base class for internal representations.
    This class is intended to be subclassed by specific internal representation implementations.
    It provides a common interface for all internal representations.
    """

    def to_internal_representation(self, external_repr: torch.Tensor) -> torch.Tensor:
        """
        If not set, we assume the external representation is the internal representation.
        """
        return external_repr

    def to_external_representation(self, internal_repr: torch.Tensor) -> torch.Tensor:
        """
        If not set, we assume the internal representation is the external representation.
        """
        return internal_repr

    def activation(self) -> nn.Module:
        """
        Activation applied to the internal representation, possibly before converting to the external representation.
        Needs to return a nn.Module so it can be integrated into a PyTorch model.
        """
        return nn.Identity()

    def loss_fn(self) -> Optional[Callable]:
        """
        Loss function applied to the internal representation.
        If not set, we use a default loss function (either MSE or CrossEntropy depending on the discreteness of the feature).
        This should not be used for discrete features.
        """
        return None

    def const_init_bias(self) -> Optional[float]:
        """
        If not None, initializes the bias of the feature to a constant value and the weights to zero.
        This is useful for features that are expected to have a certain value at initialization.
        """
        return None


class ExponentialActivation(nn.Module):
    def forward(self, x):
        return torch.exp(x)


# --- scales (stored as log of the actual value) ---
class SoftplusScaleRepresentation(InternalRepresentation):
    def __init__(self):
        super().__init__()
        # scale up so we have similar magnitudes to the other features
        self.scale_factor = 1e2
        self._const_init_bias = 0.1 * self.scale_factor

    def to_internal_representation(self, external_repr):
        scale_logit = external_repr.clamp(-10.0, -2.0)
        return torch.exp(scale_logit) * self.scale_factor

    def to_external_representation(self, internal_repr):
        scale = internal_repr / self.scale_factor
        return torch.log(scale.clamp(min=1e-10))

    def activation(self):
        return nn.Softplus()

    def const_init_bias(self):
        # before softplus(x) / scale_factor to get world space;
        # softplus is close to identity there so we just use this
        return self._const_init_bias


class ExpScaleRepresentation(InternalRepresentation):
    def __init__(self):
        super().__init__()
        self.scale_factor = 1e2
        self._const_init_bias = 0.1 * self.scale_factor

    def to_internal_representation(self, external_repr):
        scale_logit = external_repr.clamp(-10.0, -2.0)
        return torch.exp(scale_logit) * self.scale_factor

    def to_external_representation(self, internal_repr):
        scale = internal_repr / self.scale_factor
        return torch.log(scale.clamp(min=1e-10))

    def activation(self):
        return ExponentialActivation()

    def const_init_bias(self) -> Optional[float]:
        # in scaled world space
        return self._const_init_bias


# --- opacities ---
# Adapted from L3DG (https://barbararoessle.github.io/l3dg/)
class ScaledLogitOpacityRepresentation(InternalRepresentation):
    """Internal value = opacity logit clipped to [-10, 10] and scaled to [-1, 1]."""

    def to_internal_representation(self, external_repr):
        return external_repr.clip(-10.0, 10.0) / 10.0

    def to_external_representation(self, internal_repr):
        return internal_repr * 10.0

    def const_init_bias(self):
        # in logits/10 before sigmoid so ~0.12
        return -0.2


class SigmoidOpacityRepresentation(InternalRepresentation):
    """Internal value = sigmoid(external value)."""

    def to_internal_representation(self, external_repr):
        return torch.sigmoid(external_repr)

    def to_external_representation(self, internal_repr):
        return inverse_sigmoid(internal_repr)

    def activation(self):
        return nn.Sigmoid()


# --- quaternions ---
class StandardizedQuatRepresentation(InternalRepresentation):
    def to_internal_representation(self, external_repr):
        return normalize_and_standardize_quaternion(external_repr)


class Rot6DQuatRepresentation(InternalRepresentation):
    """Converts between quaternion and 6D rotation representation."""

    def to_internal_representation(self, external_repr):
        return quaternion_to_rot6d(external_repr)

    def to_external_representation(self, internal_repr):
        return rot6d_to_quaternion(internal_repr)

    def loss_fn(self):
        return rot6d_loss


# --- colors (sh0) ---
# Adapted from L3DG (https://barbararoessle.github.io/l3dg/)
class ClampedRGBColorRepresentation(InternalRepresentation):
    """Internal value = sh0 converted to RGB and clamped to [0, 1]."""

    def to_internal_representation(self, external_repr):
        return sh2rgb(external_repr).clip(0.0, 1.0)

    def to_external_representation(self, internal_repr):
        return rgb2sh(internal_repr.clip(0.0, 1.0))

    def const_init_bias(self):
        # this is in RGB space, so grey
        return 0.5


class SoftplusColorRepresentation(InternalRepresentation):
    def to_internal_representation(self, external_repr):
        # values of primitive colors can be > 1 and only clamp on pixels,
        # so technically this is too restrictive, but rarely matters in practice
        return sh2rgb(external_repr).clip(min=0.0)

    def to_external_representation(self, internal_repr):
        return rgb2sh(internal_repr.clip(min=0.0))

    def activation(self):
        return nn.Softplus()

    def const_init_bias(self):
        # this is in Softplus RGB space, so ~0.47
        return -0.5


# --- 3D offsets (no transformation) ---
class OffsetRepresentation(InternalRepresentation):
    def const_init_bias(self):
        # initialize to zero offset
        return 0.0


# Back-compat aliases: checkpoints store the instantiation target as a string
# (conf.internal_representations.<OldName>) and re-instantiate it on load, so the
# old names must keep resolving. Keep these as long as such checkpoints exist.
BoundedSoftplusScaleWorldRepresentation = SoftplusScaleRepresentation
L3DGOpacities = ScaledLogitOpacityRepresentation
StandardizedQuats = StandardizedQuatRepresentation
L3DGRGB = ClampedRGBColorRepresentation
OffsetInternalRepresentation = OffsetRepresentation
