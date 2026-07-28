"""Uncertainty calibration and risk-coverage metrics."""

from __future__ import annotations

from typing import Dict, Optional

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
