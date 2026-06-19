import os

# just to disable warning - set OMP_NUM_THREADS to 16, is already the default for ME
os.environ["OMP_NUM_THREADS"] = "16"

from typing import Any, Dict, List, Tuple

import MinkowskiEngine as ME
import torch
import torch.nn as nn

from conf.dataclasses import LayerType, PCKeys

from .minkowski_decoder import SparseConvDecoder
from .minkowski_encoder import SparseConvEncoder
from .minkowski_vq import SparseLatentVectorQuantizer

# NOTE: torch_scatter is an optional dependency, imported lazily in embed() only
# when make_unique=True. See README install section "Optional: per-voxel dedup".


# pylint: disable=redefined-outer-name


class GaussianMultiplier(nn.Module):
    def __init__(self, dim, num_gaussians):
        super(GaussianMultiplier, self).__init__()
        self.num_gaussians = num_gaussians

        # Offsets are in grid coordinates (same frame as decoder coords).
        self.offsets = nn.Parameter(torch.empty(num_gaussians, 3), requires_grad=True)
        nn.init.normal_(self.offsets, mean=0.0, std=0.2)

        self.multiplier = nn.Sequential(
            nn.Linear(dim, dim * num_gaussians),
            nn.SiLU(),
            nn.Linear(dim * num_gaussians, dim * num_gaussians),
        )

    def forward(self, points):
        new_batch = points[PCKeys.BATCH].unsqueeze(-1).repeat(1, self.num_gaussians)
        new_coords = points[PCKeys.COORDS].unsqueeze(1) + self.offsets.unsqueeze(0)
        new_coords = new_coords.view(-1, 3)

        new_features = self.multiplier(
            points[PCKeys.EMBEDDINGS]
        )  # (N, dim * num_gaussians)
        new_features = new_features.view(
            -1, self.num_gaussians, new_features.shape[-1] // self.num_gaussians
        )  # (N, num_gaussians, dim)
        new_features = new_features.view(
            -1, new_features.shape[-1]
        )  # (N * num_gaussians, dim)

        return {
            PCKeys.BATCH: new_batch.view(-1),
            PCKeys.COORDS: new_coords,
            PCKeys.EMBEDDINGS: new_features,
        }


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(Block, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.norm(x)
        x = self.dropout(x)
        x = self.act(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(ResidualBlock, self).__init__()
        self.block1 = Block(in_channels, out_channels, dropout)
        self.block2 = Block(out_channels, out_channels, dropout)
        self.skip = (
            nn.Linear(in_channels, out_channels)
            if in_channels != out_channels
            else None
        )

    def forward(self, x):
        skip = x if self.skip is None else self.skip(x)
        x = self.block1(x)
        x = self.block2(x)
        return x + skip


def construct_layers(
    layer_configs: List[Dict[str, Any]],
    curr_dim: int,
    verbose: bool = False,
) -> Tuple[nn.ModuleList, int]:
    layers = nn.ModuleList()

    def maybe_log(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    maybe_log(
        f"INFO: Constructing {len(layer_configs)} layers with input dim {curr_dim} latents"
    )

    for layer in layer_configs:
        if layer["kind"] == LayerType.FC:
            layers.append(
                nn.Linear(curr_dim, layer["out_channels"], bias=layer["bias"])
            )
            curr_dim = layer["out_channels"]

            if layer["dropout"] > 0:
                layers.append(nn.Dropout(layer["dropout"]))

        elif layer["kind"] == LayerType.RESIDUAL_BLOCK:
            for _ in range(layer["repeat"]):
                layers.append(
                    ResidualBlock(curr_dim, layer["out_channels"], layer["dropout"])
                )
                curr_dim = layer["out_channels"]

        elif layer["kind"] == LayerType.IDENTITY:
            layers.append(nn.Identity())
        else:
            raise ValueError(f"Unknown layer type: {layer}")

        num_params = sum(p.numel() for p in layers[-1].parameters())
        maybe_log(
            f"\t Added layer {layer['kind']} with {num_params} parameters and output dim {curr_dim}"
        )

    return layers, curr_dim


class ConfigurableVQVAE(nn.Module):
    """
    Minimal configurable VQ-VAE wrapper for the sparse CNN (Minkowski) backbone.
    """

    def __init__(
        self,
        features: Dict[str, Dict[str, Any]],
        encoder_config: Dict[str, Any],
        decoder_config: Dict[str, Any],
        vq_config: Dict[str, Any],
        default_grid_size: float,
        make_unique: bool = False,  # keep one (highest-opacity) Gaussian per voxel; needs torch_scatter
        gaussians_per_voxel: int = 1,
    ):
        super().__init__()

        self.features = features
        self.default_grid_size = default_grid_size

        self.make_unique = make_unique

        encoder_cfg = dict(encoder_config)
        decoder_cfg = dict(decoder_config)
        vq_cfg = dict(vq_config)

        # Embeddings
        self.embeddings = nn.ModuleDict()
        input_dim = 0
        for feature, params in features.items():
            if not params["in_input"]:
                continue

            output_dimension = params["dim_embedding"] * params["dimension"]
            if not params.get("project", False):
                raise NotImplementedError(
                    "Non-projecting feature embeddings are not implemented yet."
                )
            else:
                self.embeddings[feature] = nn.Sequential(
                    nn.Linear(params["dimension"], output_dimension),
                    ResidualBlock(output_dimension, output_dimension),
                )

            input_dim += output_dimension

        # Encoder, Decoder, VQ
        encoder_cfg["in_c"] = input_dim
        self.encoder = SparseConvEncoder(**encoder_cfg)
        self.stride = 2 ** len(self.encoder.stages)
        decoder_cfg.pop("prune_thresh_schedule", None)
        self.decoder = SparseConvDecoder(**decoder_cfg)
        self.vq = SparseLatentVectorQuantizer(dim=encoder_cfg["embed_dim"], **vq_cfg)

        if (
            isinstance(gaussians_per_voxel, bool)
            or not isinstance(gaussians_per_voxel, int)
            or gaussians_per_voxel < 1
        ):
            raise ValueError(
                f"gaussians_per_voxel must be a positive integer, got {gaussians_per_voxel}."
            )
        self.gaussians_per_voxel = gaussians_per_voxel
        if gaussians_per_voxel > 1:
            print(
                f"INFO: Using GaussianMultiplier with {gaussians_per_voxel} gaussians per voxel"
            )
            self.gaussian_multiplier = GaussianMultiplier(
                dim=decoder_cfg["out_c"],
                num_gaussians=gaussians_per_voxel,
            )

        # Feature heads
        decoder_output_dim = decoder_cfg["out_c"]
        self.feature_heads = nn.ModuleDict()
        for feature, params in features.items():
            if not params["in_output"]:
                if feature != PCKeys.COORDS:
                    print(f"INFO: Not outputting feature {feature}, skipping.")
                continue

            curr_dim = decoder_output_dim  # reset this

            feature_head_layers, curr_dim = construct_layers(
                features[feature]["feature_head"], curr_dim
            )

            final_proj = nn.Linear(curr_dim, params["dimension"])
            feature_head_layers.append(final_proj)

            internal_repr = params.get("internal_representation", None)
            if internal_repr is not None:
                const_init_bias = internal_repr.const_init_bias()
                if const_init_bias is not None:
                    print(
                        f"INFO: Initializing {feature} head bias to {const_init_bias}"
                    )
                    nn.init.constant_(final_proj.bias, const_init_bias)
                    nn.init.constant_(final_proj.weight, 0.0)

            self.feature_heads[feature] = nn.Sequential(*feature_head_layers)

        self.encoder.weight_initialization()
        self.decoder.weight_initialization()

        self.codebook_size = self.vq.codebook_size

    def embed(self, points) -> ME.SparseTensor:
        # first do voxelization to get the offset feature
        xyz_voxel = (points[PCKeys.COORDS] / self.default_grid_size).round().int()

        voxel_coords = torch.cat(
            [points[PCKeys.BATCH].unsqueeze(-1), xyz_voxel], dim=-1
        ).to(torch.int32)

        if self.make_unique:
            # Optional path: collapse each voxel to its highest-opacity Gaussian.
            # Requires torch_scatter, which is not installed by default - see the
            # "Optional: per-voxel dedup" install note in the README.
            try:
                from torch_scatter import scatter_max
            except ImportError as e:
                raise ImportError(
                    "make_unique=True requires the optional torch_scatter dependency. "
                    "Install it (see README) or set make_unique=False."
                ) from e

            # get unique mask based on highest opacity
            _, inverse = torch.unique(voxel_coords, return_inverse=True, dim=0)

            # use torch scatter to get the max opacity point per voxel
            opa = points[PCKeys.OPACITIES]
            _, max_indices = scatter_max(opa, inverse, dim=0)
            rep_idx = max_indices[inverse]
            for feature in self.features.keys():
                points[feature] = points[feature][rep_idx]

        points[PCKeys.COORD_OFFSET] = points[
            PCKeys.COORDS
        ] / self.default_grid_size - xyz_voxel.to(points[PCKeys.COORDS].dtype)

        embedded = []
        for feature, params in sorted(self.features.items()):
            if not params["in_input"]:
                continue

            if feature not in points:
                feature_name = getattr(feature, "value", str(feature))
                available_features = sorted(
                    getattr(key, "value", str(key)) for key in points.keys()
                )
                hint = ""
                if feature_name == PCKeys.SH:
                    hint = (
                        " For Photoshape tokenization with SH-enabled checkpoints, "
                        "set data.sh_degree=1."
                    )
                raise KeyError(
                    f"Missing required input feature '{feature_name}'. "
                    f"Available features: {available_features}.{hint}"
                )

            internal_repr = params["internal_representation"]

            internal_feature = internal_repr.to_internal_representation(points[feature])
            if not torch.is_tensor(internal_feature):
                feature_name = getattr(feature, "value", str(feature))
                raise TypeError(
                    "Internal representation for feature "
                    f"'{feature_name}' returned {type(internal_feature).__name__}, "
                    "expected torch.Tensor."
                )
            if internal_feature.dim() == 1:
                internal_feature = internal_feature.unsqueeze(-1)

            embedded.append(self.embeddings[feature](internal_feature))
            del points[feature]

        assert len(embedded) > 0, "No input features to embed!"
        embedded_feats = torch.cat(embedded, dim=-1)

        return (
            ME.SparseTensor(
                features=embedded_feats,
                coordinates=voxel_coords,
                device=embedded_feats.device,
                quantization_mode=ME.SparseTensorQuantizationMode.RANDOM_SUBSAMPLE,
            ),
            voxel_coords,
        )

    def _predict_from_features(self, points: dict):
        """
        Convert embeddings of the given points into their Gaussian features.
        Overwrites the existing features (if they exist) and returns the updated points.
        """

        for feature, params in self.features.items():
            if feature in self.feature_heads:
                points[feature] = self.feature_heads[feature](points[PCKeys.EMBEDDINGS])

            internal_repr = params.get("internal_representation")
            if internal_repr is not None and feature in points:
                points[feature] = internal_repr.activation()(points[feature])
                points[feature] = internal_repr.to_external_representation(
                    points[feature]
                )

        return points

    def encode(self, x: ME.SparseTensor) -> Tuple[ME.SparseTensor, Dict]:
        latents, enc_dict = self.encoder(x)
        return latents, enc_dict

    def quantize(self, latents: ME.SparseTensor) -> Tuple[ME.SparseTensor, Dict]:
        latents, vq_dict = self.vq(latents)
        return latents, vq_dict

    def decode(self, latents, target_key=None):
        latents, decode_dict = self.decoder(latents, target_key=target_key)

        points = {
            PCKeys.EMBEDDINGS: latents.F,
            PCKeys.BATCH: latents.C[:, 0],
            PCKeys.COORDS: latents.C[:, 1:].to(torch.float32),
        }
        if self.gaussians_per_voxel > 1:
            points = self.gaussian_multiplier(points)

        points = self._predict_from_features(points)
        return points, decode_dict

    def autoencode(
        self, points: dict, add_target_key=False
    ) -> Tuple[Any, torch.Tensor]:
        x, voxel_coords = self.embed(points)

        # occupancy guidance during training
        if add_target_key:
            target_key = voxel_coords
            cm = x.coordinate_manager
            target_key, _ = cm.insert_and_map(target_key, string_id="target")
        else:
            target_key = None

        latents, enc_dict = self.encode(x)
        latents, vq_dict = self.quantize(latents)
        prediction, decode_dict = self.decode(latents, target_key=target_key)

        pred_grid_coords = prediction[PCKeys.COORDS] + prediction[PCKeys.COORD_OFFSET]
        prediction[PCKeys.COORDS] = pred_grid_coords * self.default_grid_size

        # merge dicts and return
        info_dict = {}
        info_dict.update(enc_dict)
        info_dict.update(vq_dict)
        info_dict.update(decode_dict)

        return prediction, info_dict

    def forward(self, *args, **kwargs) -> Tuple[Any, torch.Tensor]:
        return self.autoencode(*args, **kwargs)

    def get_idxs(self, points: dict) -> torch.Tensor:
        x, _ = self.embed(points)
        latents, _ = self.encode(x)
        x, vq_dict = self.quantize(latents)

        actual_stride = int(x.tensor_stride[0])
        assert actual_stride == self.stride, f"{actual_stride} != {self.stride}"

        # divide coordinates by stride to get latent indices
        x.C[:, 1:] = x.C[:, 1:] // self.stride

        return x.C, vq_dict["idxs"]

    def decode_idxs(self, coords: torch.Tensor, idxs: torch.Tensor) -> dict:
        """
        Gets input in the same format get_idxs outputs it.
        """
        device = idxs.device
        latents_F = self.vq.get_output_from_idxs(idxs).to(device)

        # undo coords scaling
        coords *= self.stride

        if coords.shape[-1] == 3:
            batch_dim = torch.zeros(
                coords.shape[0], 1, device=device, dtype=torch.int32
            )
            coords = torch.cat([batch_dim, coords.to(torch.int32)], dim=-1)
        else:
            coords = coords.to(torch.int32)

        latents = ME.SparseTensor(
            features=latents_F,
            coordinates=coords,
            device=device,
            tensor_stride=torch.tensor(
                [self.stride, self.stride, self.stride], device=device
            ),
        )

        points, _ = self.decode(latents)

        pred_grid_coords = points[PCKeys.COORDS] + points[PCKeys.COORD_OFFSET]
        points[PCKeys.COORDS] = pred_grid_coords * self.default_grid_size

        return points
