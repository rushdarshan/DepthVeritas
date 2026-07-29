"""Relative-scale point-cloud export used by the companion dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch

from depthlab.geometry import pixels_to_camera


def relative_point_cloud(depth: torch.Tensor, rgb: torch.Tensor, intrinsics: torch.Tensor, valid_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Return camera points and RGB for a single relative-depth image."""
    if depth.ndim != 2 or rgb.shape != (3, *depth.shape):
        raise ValueError("depth must be [H,W] and rgb must be [3,H,W]")
    mask = torch.isfinite(depth) & (depth > 0) if valid_mask is None else valid_mask.bool() & torch.isfinite(depth) & (depth > 0)
    points = pixels_to_camera(depth.unsqueeze(0), intrinsics.unsqueeze(0)).squeeze(0).permute(1, 2, 0)[mask]
    colors = rgb.permute(1, 2, 0)[mask].clamp(0, 1)
    return points, colors


def colored_ply_text(points: torch.Tensor, colors: torch.Tensor) -> str:
    """Serialize a relative-scale point cloud as an ASCII PLY payload."""
    rows = ["ply", "format ascii 1.0", "comment DepthLab relative scale", f"element vertex {len(points)}", "property float x", "property float y", "property float z", "property uchar red", "property uchar green", "property uchar blue", "end_header"]
    for point, color in zip(points.detach().cpu(), colors.detach().cpu()):
        red, green, blue = (color * 255).round().to(torch.uint8).tolist()
        rows.append(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {red} {green} {blue}")
    return "\n".join(rows) + "\n"


def write_colored_ply(path: str | Path, points: torch.Tensor, colors: torch.Tensor) -> Path:
    """Write an ASCII PLY explicitly suitable only for relative-scale inspection."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(colored_ply_text(points, colors), encoding="ascii")
    return destination


def write_prediction_metadata(path: str | Path, metadata: Dict[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {"scale": "relative", **metadata}
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return destination
