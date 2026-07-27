"""Differentiable pinhole-camera geometry for reprojection experiments."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def pixels_to_camera(depth: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """Unproject depth [B,H,W] to homogeneous camera points [B,3,H,W]."""
    b, h, w = depth.shape
    yy, xx = torch.meshgrid(torch.arange(h, device=depth.device), torch.arange(w, device=depth.device), indexing="ij")
    pixels = torch.stack((xx, yy, torch.ones_like(xx)), dim=0).float().reshape(1, 3, -1).expand(b, -1, -1)
    rays = torch.linalg.solve(intrinsics, pixels).reshape(b, 3, h, w)
    return rays * depth.unsqueeze(1)


def camera_to_pixels(points: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    b, _, h, w = points.shape
    projected = intrinsics @ points.reshape(b, 3, -1)
    xy = projected[:, :2] / projected[:, 2:].clamp_min(1e-8)
    return xy.reshape(b, 2, h, w)


def reproject_depth(depth: torch.Tensor, intrinsics: torch.Tensor, transform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return projected pixel coordinates and positive target-frame z-depth."""
    points = pixels_to_camera(depth, intrinsics).reshape(depth.shape[0], 3, -1)
    homogeneous = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
    transformed = (transform @ homogeneous)[:, :3].reshape(depth.shape[0], 3, *depth.shape[-2:])
    return camera_to_pixels(transformed, intrinsics), transformed[:, 2]


def sample_at_pixels(source: torch.Tensor, pixels: torch.Tensor) -> torch.Tensor:
    _, _, h, w = source.shape
    grid = pixels.permute(0, 2, 3, 1).clone()
    grid[..., 0] = 2 * grid[..., 0] / max(w - 1, 1) - 1
    grid[..., 1] = 2 * grid[..., 1] / max(h - 1, 1) - 1
    return F.grid_sample(source, grid, padding_mode="border", align_corners=True)
