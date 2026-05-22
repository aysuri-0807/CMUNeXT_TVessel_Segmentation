"""Vanilla CMUNeXt architecture.

This follows the official CMUNeXt PyTorch implementation:
https://github.com/FengheTan9/CMUNeXt/blob/main/network/CMUNeXt.py
"""

from __future__ import annotations

import torch
from torch import nn


class Residual(nn.Module):
    def __init__(self, fn: nn.Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x) + x


class ConvBlock(nn.Module):
    def __init__(self, ch_in: int, ch_out: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class CMUNeXtBlock(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, depth: int = 1, k: int = 3) -> None:
        super().__init__()
        self.block = nn.Sequential(
            *[
                nn.Sequential(
                    Residual(
                        nn.Sequential(
                            nn.Conv2d(
                                ch_in,
                                ch_in,
                                kernel_size=(k, k),
                                groups=ch_in,
                                padding=(k // 2, k // 2),
                            ),
                            nn.GELU(),
                            nn.BatchNorm2d(ch_in),
                        )
                    ),
                    nn.Conv2d(ch_in, ch_in * 4, kernel_size=(1, 1)),
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in * 4),
                    nn.Conv2d(ch_in * 4, ch_in, kernel_size=(1, 1)),
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in),
                )
                for _ in range(depth)
            ]
        )
        self.up = ConvBlock(ch_in, ch_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        return self.up(x)


class UpConv(nn.Module):
    def __init__(self, ch_in: int, ch_out: int) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class FusionConv(nn.Module):
    def __init__(self, ch_in: int, ch_out: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, groups=2, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(ch_in),
            nn.Conv2d(ch_in, ch_out * 4, kernel_size=(1, 1)),
            nn.GELU(),
            nn.BatchNorm2d(ch_out * 4),
            nn.Conv2d(ch_out * 4, ch_out, kernel_size=(1, 1)),
            nn.GELU(),
            nn.BatchNorm2d(ch_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class CMUNeXt(nn.Module):
    def __init__(
        self,
        input_channel: int = 3,
        num_classes: int = 1,
        dims: list[int] | None = None,
        depths: list[int] | None = None,
        kernels: list[int] | None = None,
    ) -> None:
        super().__init__()
        dims = dims or [16, 32, 128, 160, 256]
        depths = depths or [1, 1, 1, 3, 1]
        kernels = kernels or [3, 3, 7, 7, 7]

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = ConvBlock(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])

        self.up5 = UpConv(ch_in=dims[4], ch_out=dims[3])
        self.up_conv5 = FusionConv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.up4 = UpConv(ch_in=dims[3], ch_out=dims[2])
        self.up_conv4 = FusionConv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.up3 = UpConv(ch_in=dims[2], ch_out=dims[1])
        self.up_conv3 = FusionConv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.up2 = UpConv(ch_in=dims[1], ch_out=dims[0])
        self.up_conv2 = FusionConv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.maxpool(x4)
        x5 = self.encoder5(x5)

        d5 = self.up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.up_conv5(d5)
        d4 = self.up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.up_conv4(d4)
        d3 = self.up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.up_conv3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.up_conv2(d2)
        return self.conv_1x1(d2)


def cmunext(
    input_channel: int = 3,
    num_classes: int = 1,
    dims: list[int] | None = None,
    depths: list[int] | None = None,
    kernels: list[int] | None = None,
) -> CMUNeXt:
    return CMUNeXt(
        input_channel=input_channel,
        num_classes=num_classes,
        dims=dims or [16, 32, 128, 160, 256],
        depths=depths or [1, 1, 1, 3, 1],
        kernels=kernels or [3, 3, 7, 7, 7],
    )


def cmunext_s(input_channel: int = 3, num_classes: int = 1) -> CMUNeXt:
    return CMUNeXt(
        input_channel=input_channel,
        num_classes=num_classes,
        dims=[8, 16, 32, 64, 128],
        depths=[1, 1, 1, 1, 1],
        kernels=[3, 3, 7, 7, 9],
    )


def cmunext_l(input_channel: int = 3, num_classes: int = 1) -> CMUNeXt:
    return CMUNeXt(
        input_channel=input_channel,
        num_classes=num_classes,
        dims=[32, 64, 128, 256, 512],
        depths=[1, 1, 1, 6, 3],
        kernels=[3, 3, 7, 7, 7],
    )


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
