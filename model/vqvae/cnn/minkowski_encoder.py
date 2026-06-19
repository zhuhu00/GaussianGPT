from typing import Dict, Tuple

import MinkowskiEngine as ME
from torch import nn

from .minkowski_blocks import BasicConvolutionBlock, ResidualBlock


class SparseConvEncoder(nn.Module):
    def __init__(
        self,
        in_c,
        embed_dim=64,
        max_channel=512,
        leaky=False,
        dropout=0.0,
        stages=2,
    ):
        super().__init__()

        self.lat_dim = embed_dim

        cs = [min(128 * (2**i), max_channel) for i in range(stages + 2)]

        self.stem = BasicConvolutionBlock(
            in_c, cs[0], ks=3, stride=1, dilation=1, leaky=leaky
        )

        self.stages = nn.ModuleList()
        for i in range(stages):
            stage = nn.Sequential(
                ResidualBlock(
                    cs[i], cs[i + 1], ks=3, stride=1, dilation=1, leaky=leaky
                ),
                ResidualBlock(
                    cs[i + 1], cs[i + 1], ks=3, stride=1, dilation=1, leaky=leaky
                ),
                ME.MinkowskiDropout(p=dropout),
                BasicConvolutionBlock(
                    cs[i + 1],
                    cs[i + 1],
                    ks=3,
                    stride=2,
                    dilation=1,
                    leaky=leaky,
                ),  # Downsampling (/2)
            )
            self.stages.append(stage)

        # last block without downsample
        self.stage_final = nn.Sequential(
            ResidualBlock(
                cs[stages], cs[stages + 1], ks=3, stride=1, dilation=1, leaky=leaky
            ),
            ME.MinkowskiDropout(p=dropout),
        )

        self.out = ME.MinkowskiConvolution(
            cs[stages + 1], embed_dim, kernel_size=3, stride=1, dimension=3
        )

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: ME.SparseTensor) -> Tuple[ME.SparseTensor, Dict]:
        x0 = self.stem(x)
        for stage in self.stages:
            x0 = stage(x0)
        x0 = self.stage_final(x0)
        x0 = self.out(x0)  # sparse tensor

        # num latents per batch
        num_latents = x0.F.shape[0] / (x0.C[:, 0].max().item() + 1)
        return x0, {"num_latents": num_latents}
