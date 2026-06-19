import MinkowskiEngine as ME
from torch import nn

# Adapted from L3DG (https://barbararoessle.github.io/l3dg/)


class BasicConvolutionBlock(nn.Module):

    def __init__(self, inc, outc, ks=3, stride=1, dilation=1, leaky=False):
        super().__init__()
        self.net = nn.Sequential(
            ME.MinkowskiConvolution(
                inc, outc, kernel_size=ks, dilation=dilation, stride=stride, dimension=3
            ),
            ME.MinkowskiBatchNorm(outc),
            (
                ME.MinkowskiLeakyReLU(inplace=True)
                if leaky
                else ME.MinkowskiReLU(inplace=True)
            ),
        )

    def forward(self, x):
        out = self.net(x)
        return out


class BasicDeconvolutionBlock(nn.Module):

    def __init__(self, inc, outc, ks=3, stride=1, leaky=False):
        super().__init__()
        self.net = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                inc, outc, kernel_size=ks, stride=stride, dimension=3
            ),
            ME.MinkowskiBatchNorm(outc),
            (
                ME.MinkowskiLeakyReLU(inplace=True)
                if leaky
                else ME.MinkowskiReLU(inplace=True)
            ),
        )

    def forward(self, x):
        return self.net(x)


class BasicNonGenerativeDeconvolutionBlock(nn.Module):

    def __init__(self, inc, outc, ks=3, stride=1, leaky=False):
        super().__init__()
        self.net = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                inc, outc, kernel_size=ks, stride=stride, dimension=3
            ),
            ME.MinkowskiBatchNorm(outc),
            (
                ME.MinkowskiLeakyReLU(inplace=True)
                if leaky
                else ME.MinkowskiReLU(inplace=True)
            ),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):

    def __init__(self, inc, outc, ks=3, stride=1, dilation=1, leaky=False):
        super().__init__()
        self.net = nn.Sequential(
            ME.MinkowskiConvolution(
                inc, outc, kernel_size=ks, dilation=dilation, stride=stride, dimension=3
            ),
            ME.MinkowskiBatchNorm(outc),
            (
                ME.MinkowskiLeakyReLU(inplace=True)
                if leaky
                else ME.MinkowskiReLU(inplace=True)
            ),
            ME.MinkowskiConvolution(
                outc, outc, kernel_size=ks, dilation=dilation, stride=1, dimension=3
            ),
            ME.MinkowskiBatchNorm(outc),
        )

        if inc == outc and stride == 1:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Sequential(
                ME.MinkowskiConvolution(
                    inc, outc, kernel_size=1, dilation=1, stride=stride, dimension=3
                ),
                ME.MinkowskiBatchNorm(outc),
            )

        self.relu = (
            ME.MinkowskiLeakyReLU(inplace=True)
            if leaky
            else ME.MinkowskiReLU(inplace=True)
        )

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out


if __name__ == "__main__":
    import torch

    x = ME.SparseTensor(
        features=torch.randn(100, 3),
        coordinates=torch.randint(0, 10, (100, 4)),
    )
    block = ResidualBlock(3, 16)
    out = block(x)
    print(out)
