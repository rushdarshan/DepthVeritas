"""Calibrated 3D plane fitting for independently supplied floor candidates."""

from __future__ import annotations

from typing import Dict

import torch

from depthlab.geometry import pixels_to_camera


def fit_plane_from_depth(depth: torch.Tensor, intrinsics: torch.Tensor, candidate_mask: torch.Tensor, *, min_points: int = 32) -> Dict[str, torch.Tensor]:
    """Fit one camera-coordinate plane per image; abstain when support is weak."""
    if depth.ndim != 3:
        raise ValueError("depth must have shape [B, H, W]")
    if candidate_mask.shape != depth.shape:
        raise ValueError("candidate_mask must match depth")
    points = pixels_to_camera(depth, intrinsics).permute(0, 2, 3, 1)
    planes, confidences, residuals = [], [], []
    for image_points, mask in zip(points, candidate_mask.bool() & torch.isfinite(depth) & (depth > 0)):
        selected = image_points[mask]
        if selected.shape[0] < min_points:
            planes.append(depth.new_zeros(4)); confidences.append(depth.new_zeros(())); residuals.append(depth.new_full((), float("inf"))); continue
        center = selected.mean(0)
        _, _, vh = torch.linalg.svd(selected - center, full_matrices=False)
        normal = vh[-1]
        normal = normal / normal.norm().clamp_min(1e-8)
        offset = -(normal * center).sum()
        residual = (selected @ normal + offset).abs().median()
        area = mask.float().mean()
        confidence = (area * torch.exp(-residual)).clamp(0, 1)
        planes.append(torch.cat((normal, offset.unsqueeze(0))))
        confidences.append(confidence); residuals.append(residual)
    return {"plane": torch.stack(planes), "confidence": torch.stack(confidences), "median_residual": torch.stack(residuals)}


def planar_huber_loss(points: torch.Tensor, plane: torch.Tensor, confidence: torch.Tensor, *, delta: float = 0.05, max_weight: float = 0.25) -> torch.Tensor:
    """Bounded robust residual; caller supplies a validated floor-candidate mask."""
    residual = (points * plane[..., :3].unsqueeze(-2)).sum(-1) + plane[..., 3].unsqueeze(-1)
    absolute = residual.abs()
    huber = torch.where(absolute <= delta, 0.5 * residual.square() / delta, absolute - 0.5 * delta)
    weight = confidence.clamp(0, 1).unsqueeze(-1) * max_weight
    return (huber * weight).mean()
