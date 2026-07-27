"""Pose and reprojection components for self-training depth on video clips."""

from __future__ import annotations

import torch
import torch.nn as nn


class PosePipeline(nn.Module):
    """Small relative-pose regressor producing an SE(3) transform per frame pair."""
    def __init__(self, channels: int = 3, hidden: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(channels * 2, hidden, 7, 2, 3), nn.ReLU(), nn.Conv2d(hidden, hidden, 5, 2, 2), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.pose = nn.Linear(hidden, 6)
        nn.init.zeros_(self.pose.weight)
        nn.init.zeros_(self.pose.bias)

    def forward(self, image: torch.Tensor, next_image: torch.Tensor) -> torch.Tensor:
        params = self.pose(self.encoder(torch.cat((image, next_image), dim=1)).flatten(1))
        rotation, translation = params[:, :3], params[:, 3:]
        rx, ry, rz = rotation.unbind(1)
        zeros, ones = torch.zeros_like(rx), torch.ones_like(rx)
        skew = torch.stack((zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros), dim=1).reshape(-1, 3, 3)
        transform = torch.eye(4, device=image.device, dtype=image.dtype).unsqueeze(0).repeat(image.shape[0], 1, 1)
        transform[:, :3, :3] = torch.eye(3, device=image.device, dtype=image.dtype) + skew
        transform[:, :3, 3] = translation
        return transform
