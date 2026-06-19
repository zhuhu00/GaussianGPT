from enum import Enum
from typing import Any, Dict


# If another layer type is added, it also needs to be added in
# - model/vqvae/cnn/configurable_vqvae.py:construct_layers
class LayerType(str, Enum):
    FC = "fc"
    RESIDUAL_BLOCK = "residual_block"
    IDENTITY = "identity"


class GaussianFeatures(str, Enum):
    COORDS = "coords"
    SCALES = "scales"
    OPACITIES = "opacities"
    QUATS = "quats"
    SH0 = "sh0"


class GradientClippingMode(str, Enum):
    NONE = "none"
    VALUE = "value"
    NORM = "norm"


# similar, to GaussianFeatures, but not all required
class PCKeys(str, Enum):
    COORDS = GaussianFeatures.COORDS.value
    SH0 = GaussianFeatures.SH0.value
    SH = "sh"
    OPACITIES = GaussianFeatures.OPACITIES.value
    SCALES = GaussianFeatures.SCALES.value
    QUATS = GaussianFeatures.QUATS.value
    COORD_OFFSET = "coord_offset"
    NORMALS = "normals"
    BATCH = "batch"
    OFFSET = "offset"
    EMBEDDINGS = "embeddings"
    COORD_IDX = "coord_idx"
    EMBEDDING_IDX = "embedding_idx"


class ImageKeys(str, Enum):
    IMAGES = "images"
    DEPTHS = "depths"
    LOSS_MASKS = "loss_masks"
    CAMERAS_R = "cameras_R"
    CAMERAS_T = "cameras_T"
    CAMERAS_FOVX = "cameras_FovX"
    CAMERAS_FOVY = "cameras_FovY"
    CAMERAS_W = "cameras_W"
    CAMERAS_H = "cameras_H"
    CAMERAS_CX = "cameras_cx"
    CAMERAS_CY = "cameras_cy"
    CAMERAS_IDXS = "camera_idxs"
    CAMERAS_IMAGE_PATH = "cameras_image_path"


# This maps layer types to their default kwargs.
DEFAULT_LAYER_CONFIGS: Dict[LayerType, Dict[str, Any]] = {
    LayerType.FC: {"out_channels": 256, "bias": True, "dropout": 0.0},
    LayerType.RESIDUAL_BLOCK: {"out_channels": 256, "dropout": 0.0, "repeat": 1},
    LayerType.IDENTITY: {},
}

# this is missing the dimension and optional internal representation, which need to be set
DEFAULT_FEATURE_CONFIG: Dict[str, Any] = {
    "in_input": True,
    "in_output": True,
    "dim_embedding": 16,
    "project": True,  # if set, do not use feature embeddigs, just project the input features
    "feature_head": [
        {
            "kind": LayerType.RESIDUAL_BLOCK,
            "out_channels": 64,
            "dropout": 0.0,
            "repeat": 2,
        }
    ],
}
