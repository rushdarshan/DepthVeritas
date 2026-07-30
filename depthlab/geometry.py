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
    # ponytail: border padding for out-of-bounds pixels — fine for RGB
    # sampling inside the in-bounds mask, but callers must mask invalid
    # coords explicitly before using sampled values in losses.
    return F.grid_sample(source, grid, padding_mode="border", align_corners=True)


def projected_coords_in_bounds(pixels: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Binary mask [B, 1, H, W] for projected pixel coordinates inside [0, W-1] x [0, H-1]."""
    px = pixels[:, 0:1]
    py = pixels[:, 1:2]
    return (px >= 0) & (px <= W - 1) & (py >= 0) & (py <= H - 1)


def positive_z_mask(z: torch.Tensor) -> torch.Tensor:
    """Binary mask [B, 1, H, W] for positive target-frame depth."""
    z_ = z.unsqueeze(1) if z.ndim == 3 else z
    return z_ > 0


def occlusion_mask(
    depth_ref: torch.Tensor,
    depth_sampled: torch.Tensor,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Binary mask [B, 1, H, W]: True where depth_sampled is NOT significantly shallower.

    Marks occluded pixels (sampled depth << expected depth) as False.
    Both tensors shape [B, 1, H, W].
    """
    return (depth_sampled >= depth_ref / threshold) | (depth_ref < 1e-6)


def sample_target_depth(depth_target: torch.Tensor, pixels: torch.Tensor) -> torch.Tensor:
    """Sample target depth at projected pixel coordinates.

    depth_target: [B, 1, H, W]
    pixels: [B, 2, H, W] — projected coords from reproject_depth
    Returns: [B, 1, H, W] — depth sampled at projected locations
    """
    return sample_at_pixels(depth_target, pixels)


def minimum_reprojection_mask(
    source_images: list[torch.Tensor],
    target_image: torch.Tensor,
    pixels_list: list[torch.Tensor],
) -> torch.Tensor:
    """Per-pixel mask where the source with minimum photometric error is valid.

    Returns mask [B, 1, H, W] — True where at least one source has low error.
    Handles dynamic objects by not penalizing regions where any single source
    reconstructs the target well.
    """
    B, _, H, W = target_image.shape
    min_err = None
    for src, pix in zip(source_images, pixels_list):
        warped = sample_at_pixels(src, pix)
        err = (target_image - warped).abs().mean(1, keepdim=True)
        min_err = err if min_err is None else torch.minimum(min_err, err)
    # ponytail: fixed threshold — revisit with learned or adaptive threshold
    return min_err < 0.15
