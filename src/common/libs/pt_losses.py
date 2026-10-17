# -*- coding: utf-8 -*-

from functools import partial

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torchvision import models

# import piqa  # disabled due to https://github.com/francois-rozet/piqa/issues/25
import pytorch_msssim
import sys

sys.path.append("..")
# from common.extlibs import DISTS_pt

# class MS_SSIM_loss(piqa.MS_SSIM):
#     def __init__(self, **kwargs):
#         r""""""
#         super().__init__(**kwargs)
#     def forward(self, input, target):
#         return 1-super().forward(input, target)


class MS_SSIM_loss(pytorch_msssim.MS_SSIM):
    def __init__(self, data_range=1.0, **kwargs):
        r""""""
        super().__init__(data_range=data_range, **kwargs)

    def forward(self, input, target):
        return 1 - super().forward(input, target)


class MS_SSIM_metric(pytorch_msssim.MS_SSIM):
    def __init__(self, data_range=1.0, **kwargs):
        r""""""
        super().__init__(data_range=data_range, **kwargs)


class L1MSSSIMLoss(torch.nn.Module):
    def __init__(self, l1_weight=0.16, msssim_weight=0.84):
        super().__init__()
        self.l1_weight = l1_weight
        self.msssim_weight = msssim_weight
        self.msssim_loss = MS_SSIM_loss()

    def forward(self, input, target):
        l1 = torch.nn.functional.l1_loss(input, target)
        return self.l1_weight * l1 + self.msssim_weight * self.msssim_loss(
            input, target
        )


class ShadowL1MSSSIMLoss(L1MSSSIMLoss):
    def __init__(self, alpha=0.2, beta=5.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta

    def shadow_weighted_l1(self, input, target):
        luminance = (
            0.2627 * target[:, 0:1]
            + 0.6780 * target[:, 1:2]
            + 0.0593 * target[:, 2:3]
        )
        shadow_weight = self.alpha + (1.0 - self.alpha) * torch.exp(
            -self.beta * luminance
        )
        return (shadow_weight * torch.abs(input - target)).mean()

    def forward(self, input, target):
        shadow_l1 = self.shadow_weighted_l1(input, target)
        return self.l1_weight * shadow_l1 + self.msssim_weight * self.msssim_loss(
            input, target
        )


class VGG16PerceptualL1(torch.nn.Module):
    def __init__(self, layers=(0, 5, 10, 19, 28)):
        super().__init__()
        self.layers = frozenset(layers)
        self.features = models.vgg16(
            weights=models.VGG16_Weights.IMAGENET1K_V1
        ).features[: max(layers) + 1]
        self.features.eval()
        self.features.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.features.eval()
        return self

    def forward(self, input, target):
        loss = input.new_zeros(())
        for index, layer in enumerate(self.features):
            input = layer(input)
            target = layer(target)
            if index in self.layers:
                loss = loss + F.l1_loss(input, target)
        return loss


class RawForgeLoss(torch.nn.Module):
    def __init__(
        self,
        alpha=0.2,
        beta=5.0,
        l1_weight=0.16,
        msssim_weight=0.84,
        perceptual_weight=0.025,
        perceptual_loss=None,
        perceptual_chunk_size=1,
        checkpoint_perceptual=True,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.l1_weight = l1_weight
        self.msssim_weight = msssim_weight
        self.perceptual_weight = perceptual_weight
        self.perceptual_loss = perceptual_loss or VGG16PerceptualL1()
        self.perceptual_chunk_size = perceptual_chunk_size
        self.checkpoint_perceptual = checkpoint_perceptual
        self.msssim_loss = MS_SSIM_loss()

    @staticmethod
    def apply_gamma(input):
        return input.clamp(1e-6, 1.0).pow(1.0 / 2.2).clamp(1e-6, 1.0)

    def shadow_weighted_l1(self, input, target):
        luminance = (
            0.2126 * target[:, 0:1]
            + 0.7152 * target[:, 1:2]
            + 0.0722 * target[:, 2:3]
        )
        shadow_weight = self.alpha + (1.0 - self.alpha) * torch.exp(
            -self.beta * luminance
        )
        return (shadow_weight * torch.abs(input - target)).mean()

    def chunked_perceptual_loss(self, input, target):
        total = input.new_zeros(())
        batch_size = input.shape[0]
        for start in range(0, batch_size, self.perceptual_chunk_size):
            end = min(start + self.perceptual_chunk_size, batch_size)
            input_chunk = input[start:end]
            target_chunk = target[start:end]
            if self.checkpoint_perceptual and input_chunk.requires_grad:
                chunk_loss = checkpoint(
                    self.perceptual_loss,
                    input_chunk,
                    target_chunk,
                    use_reentrant=False,
                )
            else:
                chunk_loss = self.perceptual_loss(input_chunk, target_chunk)
            total = total + chunk_loss * (end - start)
        return total / batch_size

    def forward(self, input, target):
        input = self.apply_gamma(input)
        target = self.apply_gamma(target)
        shadow_l1 = self.shadow_weighted_l1(input, target)
        perceptual = self.chunked_perceptual_loss(input, target)
        return (
            self.l1_weight * shadow_l1
            + self.msssim_weight * self.msssim_loss(input, target)
            + self.perceptual_weight * perceptual
        )


# class SSIM_loss(piqa.SSIM):
#     def __init__(self, **kwargs):
#         r""""""
#         super().__init__(**kwargs)
#     def forward(self, input, target):
#         return 1-super().forward(input, target)


# class DISTS_loss(DISTS_pt.DISTS):
#     def __init__(self, **kwargs):
#         super().__init__()
#
#     def forward(self, x, y):
#         return super().forward(x, y, require_grad=True, batch_average=True)


losses = {
    "mse": torch.nn.MSELoss,
    "msssim_loss": MS_SSIM_loss,
    "l1_msssim_loss": L1MSSSIMLoss,
    "shadow_l1_msssim_loss": ShadowL1MSSSIMLoss,
    "rawforge_loss": RawForgeLoss,
    "rawforge_detail_loss": partial(RawForgeLoss, perceptual_weight=0.035),
}
# metrics = losses | {"msssim": MS_SSIM_metric}  # python 3.8 / 3.10 compat
metrics = {
    "msssim": MS_SSIM_metric,
    "mse": torch.nn.MSELoss,
    "msssim_loss": MS_SSIM_loss,
    # "dists": DISTS_loss,
}  # python 3.8 / 3.10 compat
