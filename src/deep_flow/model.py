"""U-Net generator used for airfoil RANS flow prediction.

This module is adapted from the TUM Deep-Flow-Prediction implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(module):
    """Initialize convolution and batch-normalization layers."""
    classname = module.__class__.__name__
    if classname.find("Conv") != -1:
        module.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        module.weight.data.normal_(1.0, 0.02)
        module.bias.data.fill_(0)


def block_unet(
    in_channels,
    out_channels,
    name,
    transposed=False,
    batch_norm=True,
    relu=True,
    size=4,
    padding=1,
    dropout=0.0,
):
    """Create one encoder or decoder block for the U-Net."""
    block = nn.Sequential()
    if relu:
        block.add_module(f"{name}_relu", nn.ReLU(inplace=True))
    else:
        block.add_module(f"{name}_leakyrelu", nn.LeakyReLU(0.2, inplace=True))

    if not transposed:
        block.add_module(
            f"{name}_conv",
            nn.Conv2d(in_channels, out_channels, kernel_size=size, stride=2, padding=padding, bias=True),
        )
    else:
        block.add_module(f"{name}_upsam", nn.Upsample(scale_factor=2, mode="bilinear"))
        block.add_module(
            f"{name}_tconv",
            nn.Conv2d(in_channels, out_channels, kernel_size=size - 1, stride=1, padding=padding, bias=True),
        )

    if batch_norm:
        block.add_module(f"{name}_bn", nn.BatchNorm2d(out_channels))
    if dropout > 0.0:
        block.add_module(f"{name}_dropout", nn.Dropout2d(dropout, inplace=True))
    return block


class TurbNetG(nn.Module):
    """Generator network mapping boundary conditions to pressure and velocity fields."""

    def __init__(self, channel_exponent=6, dropout=0.0):
        super().__init__()
        channels = int(2**channel_exponent + 0.5)

        self.layer1 = nn.Sequential()
        self.layer1.add_module("layer1_conv", nn.Conv2d(3, channels, 4, 2, 1, bias=True))

        self.layer2 = block_unet(channels, channels * 2, "layer2", relu=False, dropout=dropout)
        self.layer2b = block_unet(channels * 2, channels * 2, "layer2b", relu=False, dropout=dropout)
        self.layer3 = block_unet(channels * 2, channels * 4, "layer3", relu=False, dropout=dropout)
        self.layer4 = block_unet(channels * 4, channels * 8, "layer4", relu=False, dropout=dropout, size=4)
        self.layer5 = block_unet(channels * 8, channels * 8, "layer5", relu=False, dropout=dropout, size=2, padding=0)
        self.layer6 = block_unet(
            channels * 8,
            channels * 8,
            "layer6",
            batch_norm=False,
            relu=False,
            dropout=dropout,
            size=2,
            padding=0,
        )

        self.dlayer6 = block_unet(
            channels * 8,
            channels * 8,
            "dlayer6",
            transposed=True,
            dropout=dropout,
            size=2,
            padding=0,
        )
        self.dlayer5 = block_unet(
            channels * 16,
            channels * 8,
            "dlayer5",
            transposed=True,
            dropout=dropout,
            size=2,
            padding=0,
        )
        self.dlayer4 = block_unet(channels * 16, channels * 4, "dlayer4", transposed=True, dropout=dropout)
        self.dlayer3 = block_unet(channels * 8, channels * 2, "dlayer3", transposed=True, dropout=dropout)
        self.dlayer2b = block_unet(channels * 4, channels * 2, "dlayer2b", transposed=True, dropout=dropout)
        self.dlayer2 = block_unet(channels * 4, channels, "dlayer2", transposed=True, dropout=dropout)

        self.dlayer1 = nn.Sequential()
        self.dlayer1.add_module("dlayer1_relu", nn.ReLU(inplace=True))
        self.dlayer1.add_module("dlayer1_tconv", nn.ConvTranspose2d(channels * 2, 3, 4, 2, 1, bias=True))

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out2b = self.layer2b(out2)
        out3 = self.layer3(out2b)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)

        dout6 = self.dlayer6(out6)
        dout5 = self.dlayer5(torch.cat([dout6, out5], 1))
        dout4 = self.dlayer4(torch.cat([dout5, out4], 1))
        dout3 = self.dlayer3(torch.cat([dout4, out3], 1))
        dout2b = self.dlayer2b(torch.cat([dout3, out2b], 1))
        dout2 = self.dlayer2(torch.cat([dout2b, out2], 1))
        return self.dlayer1(torch.cat([dout2, out1], 1))


class TurbNetD(nn.Module):
    """Discriminator from the original project. It is kept for compatibility."""

    def __init__(self, in_channels1, in_channels2, channels=64):
        super().__init__()
        self.c0 = nn.Conv2d(in_channels1 + in_channels2, channels, 4, stride=2, padding=2)
        self.c1 = nn.Conv2d(channels, channels * 2, 4, stride=2, padding=2)
        self.c2 = nn.Conv2d(channels * 2, channels * 4, 4, stride=2, padding=2)
        self.c3 = nn.Conv2d(channels * 4, channels * 8, 4, stride=2, padding=2)
        self.c4 = nn.Conv2d(channels * 8, 1, 4, stride=2, padding=2)

        self.bnc1 = nn.BatchNorm2d(channels * 2)
        self.bnc2 = nn.BatchNorm2d(channels * 4)
        self.bnc3 = nn.BatchNorm2d(channels * 8)

    def forward(self, x1, x2):
        h = self.c0(torch.cat((x1, x2), 1))
        h = self.bnc1(self.c1(F.leaky_relu(h, negative_slope=0.2)))
        h = self.bnc2(self.c2(F.leaky_relu(h, negative_slope=0.2)))
        h = self.bnc3(self.c3(F.leaky_relu(h, negative_slope=0.2)))
        return torch.sigmoid(self.c4(F.leaky_relu(h, negative_slope=0.2)))
