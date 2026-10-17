# Adapted from https://github.com/megvii-research/NAFNet (MIT License).
# Copyright (c) 2022 megvii-model.
# Modified for packed Bayer-to-RGB output in RawNIND.

import torch
import torch.nn.functional as F
from torch import nn


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, input):
        mean = input.mean(1, keepdim=True)
        variance = (input - mean).pow(2).mean(1, keepdim=True)
        output = (input - mean) / torch.sqrt(variance + self.eps)
        return output * self.weight[None, :, None, None] + self.bias[
            None, :, None, None
        ]


class SimpleGate(nn.Module):
    def forward(self, input):
        first, second = input.chunk(2, dim=1)
        return first * second


class NAFBlock(nn.Module):
    def __init__(self, channels, dw_expand=2, ffn_expand=2, dropout=0.0):
        super().__init__()
        dw_channels = channels * dw_expand
        self.conv1 = nn.Conv2d(channels, dw_channels, 1)
        self.conv2 = nn.Conv2d(
            dw_channels, dw_channels, 3, padding=1, groups=dw_channels
        )
        self.conv3 = nn.Conv2d(dw_channels // 2, channels, 1)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channels // 2, dw_channels // 2, 1),
        )
        self.simple_gate = SimpleGate()

        ffn_channels = channels * ffn_expand
        self.conv4 = nn.Conv2d(channels, ffn_channels, 1)
        self.conv5 = nn.Conv2d(ffn_channels // 2, channels, 1)
        self.norm1 = LayerNorm2d(channels)
        self.norm2 = LayerNorm2d(channels)
        self.dropout1 = nn.Dropout(dropout) if dropout else nn.Identity()
        self.dropout2 = nn.Dropout(dropout) if dropout else nn.Identity()
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, input):
        output = self.conv1(self.norm1(input))
        output = self.conv2(output)
        output = self.simple_gate(output)
        output = output * self.channel_attention(output)
        output = self.dropout1(self.conv3(output))
        intermediate = input + output * self.beta

        output = self.conv4(self.norm2(intermediate))
        output = self.simple_gate(output)
        output = self.dropout2(self.conv5(output))
        return intermediate + output * self.gamma


class NAFNet(nn.Module):
    def __init__(
        self,
        in_channels=4,
        funit=16,
        preupsample=False,
        enc_blk_nums=(2, 2, 4, 8),
        middle_blk_num=12,
        dec_blk_nums=(2, 2, 2, 2),
    ):
        super().__init__()
        if in_channels != 4:
            raise ValueError("Bayer NAFNet requires four packed RGGB channels")
        if preupsample:
            raise ValueError("Bayer NAFNet does not support preupsample")
        if len(enc_blk_nums) != len(dec_blk_nums):
            raise ValueError("encoder and decoder stage counts must match")

        self.intro = nn.Conv2d(in_channels, funit, 3, padding=1)
        self.ending = nn.Conv2d(funit, 12, 3, padding=1)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        channels = funit
        for block_count in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(*[NAFBlock(channels) for _ in range(block_count)])
            )
            self.downs.append(nn.Conv2d(channels, channels * 2, 2, stride=2))
            channels *= 2

        self.middle_blocks = nn.Sequential(
            *[NAFBlock(channels) for _ in range(middle_blk_num)]
        )

        for block_count in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels * 2, 1, bias=False),
                    nn.PixelShuffle(2),
                )
            )
            channels //= 2
            self.decoders.append(
                nn.Sequential(*[NAFBlock(channels) for _ in range(block_count)])
            )

        self.padder_size = 2 ** len(enc_blk_nums)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, input):
        height, width = input.shape[-2:]
        output = self.intro(self._pad(input))
        skip_connections = []

        for encoder, down in zip(self.encoders, self.downs):
            output = encoder(output)
            skip_connections.append(output)
            output = down(output)

        output = self.middle_blocks(output)

        for decoder, up, skip in zip(
            self.decoders, self.ups, reversed(skip_connections)
        ):
            output = decoder(up(output) + skip)

        output = self.ending(output)[..., :height, :width]
        return self.pixel_shuffle(output)

    def _pad(self, input):
        height, width = input.shape[-2:]
        pad_height = (self.padder_size - height % self.padder_size) % self.padder_size
        pad_width = (self.padder_size - width % self.padder_size) % self.padder_size
        return F.pad(input, (0, pad_width, 0, pad_height))
