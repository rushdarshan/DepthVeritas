"""Photometric and geometric self-supervision losses."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from depthlab.geometry import sample_at_pixels


def _ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mu_x, mu_y = F.avg_pool2d(x, 3, 1, 1), F.avg_pool2d(y, 3, 1, 1)
    sigma_x = F.avg_pool2d(x * x, 3, 1, 1) - mu_x.square()
    sigma_y = F.avg_pool2d(y * y, 3, 1, 1) - mu_y.square()
    sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y
    ssim = ((2 * mu_x * mu_y + 0.01 ** 2) * (2 * sigma_xy + 0.03 ** 2)) / ((mu_x.square() + mu_y.square() + 0.01 ** 2) * (sigma_x + sigma_y + 0.03 ** 2))
    return ((1 - ssim) / 2).clamp(0, 1)


class PhotometricConsistencyLoss(nn.Module):
    def __init__(self, ssim_weight: float = 0.85) -> None:
        super().__init__()
        self.ssim_weight = ssim_weight

    def forward(self, target: torch.Tensor, source: torch.Tensor, pixels: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        warped = sample_at_pixels(source, pixels)
        error = self.ssim_weight * _ssim(target, warped).mean(1) + (1 - self.ssim_weight) * (target - warped).abs().mean(1)
        return error[mask].mean() if mask is not None and mask.any() else error.mean()


class TemporalConsistencyLoss(nn.Module):
    def forward(self, depth_t: torch.Tensor, depth_next: torch.Tensor,
                projected_depth: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        error = (depth_next - projected_depth).abs() / depth_next.clamp_min(1e-3)
        return error[mask].mean() if mask is not None and mask.any() else error.mean()
