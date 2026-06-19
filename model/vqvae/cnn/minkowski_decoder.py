from typing import Dict, Tuple

import MinkowskiEngine as ME
import torch
from torch import nn

from .minkowski_blocks import (
    BasicConvolutionBlock,
    BasicDeconvolutionBlock,
    ResidualBlock,
)

# Adapted from L3DG (https://barbararoessle.github.io/l3dg/)


class SparseConvDecoder(nn.Module):
    def __init__(
        self,
        out_c,
        embed_dim=64,
        max_channel=512,
        leaky=False,
        dropout=0.0,
        stages=2,
        prune_thresh=[0.0, 0.0],
    ):
        super().__init__()

        assert len(prune_thresh) == stages
        self.prune_thresh = list(prune_thresh)  # copy to avoid modifying input list

        cs = [min(128 * (2**i), max_channel) for i in range(stages + 2)][::-1]

        self.stem = BasicConvolutionBlock(embed_dim, cs[0], ks=3, stride=1, leaky=leaky)

        self.ups = nn.ModuleList()
        self.up_cls = nn.ModuleList()
        for i in range(stages):
            up = nn.Sequential(
                ResidualBlock(
                    cs[i], cs[i + 1], ks=3, stride=1, dilation=1, leaky=leaky
                ),
                ResidualBlock(
                    cs[i + 1], cs[i + 1], ks=3, stride=1, dilation=1, leaky=leaky
                ),
                ME.MinkowskiDropout(p=dropout),
                BasicDeconvolutionBlock(
                    cs[i + 1],
                    cs[i + 1],
                    ks=3,
                    stride=2,
                    leaky=leaky,
                ),  # Upsampling (*2)
            )
            self.ups.append(up)

            up_cls = nn.Sequential(
                ME.MinkowskiConvolution(
                    cs[i + 1], 1, kernel_size=1, bias=True, dimension=3
                ),
            )
            self.up_cls.append(up_cls)

        self.up_final = nn.Sequential(
            ResidualBlock(cs[-2], cs[-1], ks=3, stride=1, dilation=1, leaky=leaky),
            ResidualBlock(cs[-1], cs[-1], ks=3, stride=1, dilation=1, leaky=leaky),
            ME.MinkowskiDropout(p=dropout),
        )

        self.out = ME.MinkowskiConvolution(
            cs[-1], out_c, kernel_size=3, stride=1, dimension=3
        )

        self.pruning = ME.MinkowskiPruning()

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @torch.no_grad()
    def get_target(self, out, target_key, kernel_size=1):
        """
        This function creates a boolean mask (target) over the output coordinates (out), where each entry is True if that coordinate lies within a local neighborhood (defined by the kernel) around any of the target coordinates.
        """
        target = torch.zeros(len(out), dtype=torch.bool, device=out.device)
        cm = out.coordinate_manager
        strided_target_key = cm.stride(
            target_key,
            out.tensor_stride[0],
        )  # downsamples target coordinates to the same stride/resolution as out tensor
        kernel_map = cm.kernel_map(
            out.coordinate_map_key,
            strided_target_key,
            kernel_size=kernel_size,
            region_type=1,
        )  # find mappings between coordinates in out and target_key using a kernel neighborhood
        for k, curr_in in kernel_map.items():
            target[curr_in[0].long()] = (
                1  # all out coordinates that lie within a kernel neighborhood of the target positions are marked as True
            )
        return target

    # set target_key to compute occupancy loss, not needed at test time
    def forward(
        self, x: ME.SparseTensor, target_key=None
    ) -> Tuple[ME.SparseTensor, Dict]:

        out_cls, targets = [], []

        x0 = self.stem(x)

        prune_median, prune_75p = [], []
        for i in range(len(self.ups)):
            x0 = self.ups[i](x0)
            out_i_cls = self.up_cls[i](x0)

            if target_key is not None:
                target = self.get_target(x0, target_key)
                targets.append(target)
                out_cls.append(out_i_cls)

            keep = (out_i_cls.F > self.prune_thresh[i]).squeeze()

            prune_median.append(out_i_cls.F.median())
            prune_75p.append(
                out_i_cls.F.to(torch.float32).quantile(q=0.75)
            )  # quantile does NOT support bfloat16

            if (
                self.training
            ):  # during training, force target shape generation, use net.eval() to disable
                keep += target

            x0 = self.pruning(x0, keep)

        x0 = self.up_final(x0)
        x0 = self.out(x0)

        out_dict = {
            "out_cls": out_cls,
            "targets": targets,
            "prune_median": prune_median,
            "prune_75p": prune_75p,
        }

        return x0, out_dict
