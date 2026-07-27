"""Frame-pair metrics used by video depth and benchmark scoring."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def flow_warp(image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Warp ``image`` [B,C,H,W] into the next frame using pixel-space flow."""
    b, _, h, w = image.shape
    yy, xx = torch.meshgrid(torch.arange(h, device=image.device), torch.arange(w, device=image.device), indexing="ij")
    grid = torch.stack((xx, yy), dim=-1).float().unsqueeze(0) + flow.permute(0, 2, 3, 1)
    grid[..., 0] = 2 * grid[..., 0] / max(w - 1, 1) - 1
    grid[..., 1] = 2 * grid[..., 1] / max(h - 1, 1) - 1
    return F.grid_sample(image, grid, mode="bilinear", padding_mode="border", align_corners=True)


def temporal_metrics(depth_t: torch.Tensor, depth_next: torch.Tensor,
                     flow_t_to_next: Optional[torch.Tensor] = None,
                     mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
    if flow_t_to_next is not None:
        depth_t = flow_warp(depth_t.unsqueeze(1) if depth_t.ndim == 3 else depth_t, flow_t_to_next).squeeze(1)
    delta = depth_next - depth_t
    if mask is not None:
        delta = delta[mask]
    if not delta.numel():
        return {"flicker_variance": 0.0, "temporal_consistency_error": 0.0}
    return {"flicker_variance": float(delta.var(unbiased=False).item()), "temporal_consistency_error": float(delta.abs().mean().item())}
