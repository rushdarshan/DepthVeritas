"""Uncertainty calibration and risk-coverage metrics."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch


def _valid(confidence: torch.Tensor, correct: torch.Tensor, mask: Optional[torch.Tensor]):
    if mask is None:
        mask = torch.ones_like(confidence, dtype=torch.bool)
    return confidence[mask].float(), correct[mask].float()


def expected_calibration_error(confidence: torch.Tensor, correct: torch.Tensor,
                               mask: Optional[torch.Tensor] = None, bins: int = 15,
                               adaptive: bool = False) -> float:
    confidence, correct = _valid(confidence, correct, mask)
    if confidence.numel() == 0:
        return 0.0
    if adaptive:
        edges = torch.quantile(confidence, torch.linspace(0, 1, bins + 1, device=confidence.device))
    else:
        edges = torch.linspace(0, 1, bins + 1, device=confidence.device)
    ece = confidence.new_zeros(())
    for index in range(bins):
        in_bin = (confidence >= edges[index]) & ((confidence <= edges[index + 1]) if index == bins - 1 else (confidence < edges[index + 1]))
        if in_bin.any():
            ece += in_bin.float().mean() * (correct[in_bin].mean() - confidence[in_bin].mean()).abs()
    return float(ece.item())


def negative_log_likelihood(probabilities: torch.Tensor, target_bins: torch.Tensor,
                            mask: Optional[torch.Tensor] = None) -> float:
    chosen = probabilities.gather(-1, target_bins.long().unsqueeze(-1)).squeeze(-1).clamp_min(1e-8)
    if mask is not None:
        chosen = chosen[mask]
    return float((-chosen.log()).mean().item()) if chosen.numel() else 0.0


def area_under_risk_coverage(uncertainty: torch.Tensor, errors: torch.Tensor,
                             mask: Optional[torch.Tensor] = None) -> float:
    if mask is None:
        mask = torch.ones_like(uncertainty, dtype=torch.bool)
    u, error = uncertainty[mask].flatten(), errors[mask].float().flatten()
    if u.numel() == 0:
        return 0.0
    ordered = error[torch.argsort(u)]
    risk = ordered.cumsum(0) / torch.arange(1, ordered.numel() + 1, device=u.device)
    return float(risk.mean().item())


def uncertainty_metrics(probabilities: torch.Tensor, target_bins: torch.Tensor,
                        mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
    confidence, prediction = probabilities.max(dim=-1)
    correct = prediction.eq(target_bins)
    entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=-1)
    return {
        "ece": expected_calibration_error(confidence, correct, mask),
        "adaptive_ece": expected_calibration_error(confidence, correct, mask, adaptive=True),
        "aurc": area_under_risk_coverage(entropy, ~correct, mask),
        "nll": negative_log_likelihood(probabilities, target_bins, mask),
        "sharpness": float(confidence[mask].mean().item()) if mask is not None and mask.any() else float(confidence.mean().item()),
    }


def _effective_validity_mask(
    pred: torch.Tensor, target: torch.Tensor,
    caller_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    finite = pred.isfinite() & target.isfinite()
    positive = (pred > 0) & (target > 0)
    valid = finite & positive
    if caller_mask is not None:
        valid = valid & caller_mask
    return valid


def depth_error_event(
    pred: torch.Tensor, target: torch.Tensor,
    threshold_ratio: float = 0.1,
    align_scale_shift: bool = False,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if pred.ndim != target.ndim:
        raise ValueError("pred and target must have the same number of dimensions")
    if align_scale_shift:
        from depthlab.metrics.depth_metrics import align_depth_scale_shift
        pred = align_depth_scale_shift(pred, target, mask)
    valid = _effective_validity_mask(pred, target, mask)
    rel = (pred - target).abs() / target.clamp_min(1e-6)
    err = rel > threshold_ratio
    err = err & valid
    return err


def _tiles(tensor: torch.Tensor, tile_size: int, agg: str = "mean") -> torch.Tensor:
    if tensor.ndim not in (2, 3):
        raise ValueError("tensor must be 2D (H,W) or 3D (B,H,W)")
    if tile_size < 1:
        raise ValueError(f"tile_size must be >=1, got {tile_size}")
    stack = False
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
        stack = True
    B, H, W = tensor.shape
    H_pad = (tile_size - H % tile_size) % tile_size
    W_pad = (tile_size - W % tile_size) % tile_size
    if H_pad > 0 or W_pad > 0:
        tensor = torch.nn.functional.pad(tensor, (0, W_pad, 0, H_pad), mode="replicate")
    padded_H, padded_W = tensor.shape[1], tensor.shape[2]
    Ht = padded_H // tile_size
    Wt = padded_W // tile_size
    tiles = tensor.view(B, Ht, tile_size, Wt, tile_size).permute(0, 1, 3, 2, 4).reshape(B, Ht * Wt, tile_size * tile_size)
    if agg == "mean":
        result = tiles.mean(dim=-1)
    elif agg == "max":
        result = tiles.amax(dim=-1)
    elif agg == "any":
        result = tiles.any(dim=-1).float()
    elif agg == "sum":
        result = tiles.sum(dim=-1)
    else:
        raise ValueError(f"Unknown agg: {agg}")
    if stack:
        result = result.squeeze(0)
    return result


def risk_coverage_curve(
    risk: torch.Tensor, errors: torch.Tensor,
    tile_size: int = 32, n_steps: int = 100,
) -> Dict[str, torch.Tensor]:
    tile_risk = _tiles(risk, tile_size, agg="mean")
    tile_err = _tiles(errors.float(), tile_size, agg="any")
    if tile_risk.ndim == 1:
        tile_risk = tile_risk.unsqueeze(0)
        tile_err = tile_err.unsqueeze(0)
    B, N = tile_risk.shape
    coverages = torch.linspace(1.0, 0.01, n_steps, device=risk.device)
    curves = torch.zeros(B, n_steps, device=risk.device)
    baselines = torch.zeros(B, n_steps, device=risk.device)
    for i in range(B):
        order = torch.argsort(tile_risk[i], descending=False)
        sorted_err = tile_err[i][order]
        kept = torch.arange(1, N + 1, device=risk.device).float()
        running_err = sorted_err.cumsum(0) / kept
        cov = kept / N
        interpolated = torch.zeros(n_steps, device=risk.device)
        for j, c in enumerate(coverages):
            idx = (cov >= c).nonzero(as_tuple=True)[0]
            interpolated[j] = running_err[idx[0]] if idx.numel() > 0 else running_err[-1]
        curves[i] = interpolated
        baselines[i] = tile_err[i].mean()
    return {"risk_selective": curves.squeeze() if B == 1 else curves,
            "random_baseline": baselines.squeeze() if B == 1 else baselines,
            "coverages": coverages}
