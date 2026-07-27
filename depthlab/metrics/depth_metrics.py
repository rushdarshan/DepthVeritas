"""
depthlab/metrics/depth_metrics.py — Core Depth Estimation Metric Definitions.
"""

import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn


def align_depth_scale_shift(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Aligns predicted relative depth to target metric depth using least-squares scale and shift.
    Solves min_{s, t} sum || (s * pred + t) - target ||^2
    """
    if mask is None:
        mask = (target > 0) & (~torch.isnan(target)) & (~torch.isinf(target))

    pred_masked = pred[mask]
    target_masked = target[mask]

    if pred_masked.numel() == 0:
        return pred

    pred_masked = torch.clamp(pred_masked, min=1e-3)
    target_masked = torch.clamp(target_masked, min=1e-3)

    n = float(pred_masked.numel())
    sum_p = torch.sum(pred_masked)
    sum_t = torch.sum(target_masked)
    sum_pp = torch.sum(pred_masked ** 2)
    sum_pt = torch.sum(pred_masked * target_masked)

    det = sum_pp * n - sum_p * sum_p
    if torch.abs(det) < 1e-8:
        scale = torch.median(target_masked) / (torch.median(pred_masked) + 1e-8)
        shift = torch.tensor(0.0, device=pred.device)
    else:
        scale = (sum_pt * n - sum_p * sum_t) / det
        shift = (sum_pp * sum_t - sum_p * sum_pt) / det

    return scale * pred + shift


def compute_depth_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    align: bool = False,
    min_depth: float = 1e-3,
    max_depth: float = 80.0
) -> Dict[str, float]:
    """
    Computes depth evaluation metrics on valid masked pixels.

    Args:
        pred: Predicted depth map (B, 1, H, W) or (B, H, W)
        target: Target depth map (B, 1, H, W) or (B, H, W)
        mask: Optional boolean mask (B, 1, H, W) where True = valid pixel
        align: Whether to compute least-squares scale & shift alignment
        min_depth: Minimum valid depth threshold
        max_depth: Maximum valid depth threshold

    Returns:
        Dictionary containing scalar float metric values.
    """
    if pred.ndim == 3:
        pred = pred.unsqueeze(1)
    if target.ndim == 3:
        target = target.unsqueeze(1)

    valid_mask = (
        (target > min_depth)
        & (target < max_depth)
        & torch.isfinite(target)
        & torch.isfinite(pred)
    )
    if mask is not None:
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        valid_mask = valid_mask & mask

    if not valid_mask.any():
        return {
            "abs_rel": 0.0, "sq_rel": 0.0, "rmse": 0.0, "rmse_log": 0.0,
            "silog": 0.0, "log10": 0.0, "delta1": 0.0, "delta2": 0.0, "delta3": 0.0,
            "d1": 0.0, "d2": 0.0, "d3": 0.0
        }

    if align:
        pred = align_depth_scale_shift(pred, target, valid_mask)

    pred = torch.clamp(pred, min=min_depth, max=max_depth)

    p = pred[valid_mask]
    t = target[valid_mask]

    thresh = torch.max(p / t, t / p)
    d1 = torch.mean((thresh < 1.25).float()).item()
    d2 = torch.mean((thresh < 1.25 ** 2).float()).item()
    d3 = torch.mean((thresh < 1.25 ** 3).float()).item()

    diff = p - t
    abs_rel = torch.mean(torch.abs(diff) / t).item()
    sq_rel = torch.mean((diff ** 2) / t).item()
    rmse = torch.sqrt(torch.mean(diff ** 2)).item()

    log_p = torch.log(p)
    log_t = torch.log(t)
    log_diff = log_p - log_t
    rmse_log = torch.sqrt(torch.mean(log_diff ** 2)).item()

    silog = torch.sqrt(torch.mean(log_diff ** 2) - (torch.mean(log_diff) ** 2) + 1e-8).item() * 100.0
    log10 = torch.mean(torch.abs(torch.log10(p) - torch.log10(t))).item()

    def finite(value: float) -> float:
        value = float(value)
        return value if math.isfinite(value) else 0.0

    return {
        "abs_rel": finite(abs_rel),
        "sq_rel": finite(sq_rel),
        "rmse": finite(rmse),
        "rmse_log": finite(rmse_log),
        "silog": finite(silog),
        "log10": finite(log10),
        "delta1": finite(d1),
        "delta2": finite(d2),
        "delta3": finite(d3),
        "d1": finite(d1),
        "d2": finite(d2),
        "d3": finite(d3),
    }
