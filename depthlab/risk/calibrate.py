from __future__ import annotations

from enum import Enum
from typing import Dict, Optional, Tuple

import torch


class TriageLabel(Enum):
    USABLE = 0
    REVIEW = 1
    ABSTAIN = 2


class TriagePolicy:
    def __init__(self, t_usable: float, t_abstain: float) -> None:
        if t_usable < 0 or t_abstain < 0 or t_usable > 1 or t_abstain > 1:
            raise ValueError("thresholds must be in [0, 1]")
        if t_usable >= t_abstain:
            raise ValueError("t_usable must be < t_abstain")
        self.t_usable = t_usable
        self.t_abstain = t_abstain

    def classify(self, risk: torch.Tensor) -> torch.Tensor:
        labels = torch.full_like(risk, TriageLabel.REVIEW.value, dtype=risk.dtype)
        labels[risk < self.t_usable] = TriageLabel.USABLE.value
        labels[risk >= self.t_abstain] = TriageLabel.ABSTAIN.value
        return labels

    def state_dict(self) -> Dict[str, float]:
        return {"t_usable": self.t_usable, "t_abstain": self.t_abstain}

    @classmethod
    def from_state_dict(cls, d: Dict[str, float]) -> TriagePolicy:
        return cls(d["t_usable"], d["t_abstain"])


def calibrate_thresholds(
    risk: torch.Tensor,
    tile_errors: torch.Tensor,
    target_fur: float = 0.01,
    n_steps: int = 100,
) -> Optional[Tuple[float, float]]:
    if risk.ndim != 1 or tile_errors.ndim != 1:
        raise ValueError("risk and tile_errors must be 1D tensors")
    if risk.shape != tile_errors.shape:
        raise ValueError("risk and tile_errors must have the same shape")
    thresholds = torch.linspace(0.0, 1.0, n_steps, device=risk.device)
    best_pair = None
    best_usable_cov = -1.0
    for t_u in thresholds:
        usable_mask = risk < t_u
        if not usable_mask.any():
            continue
        fu_rate = tile_errors[usable_mask].float().mean().item()
        if fu_rate > target_fur:
            continue
        for t_a in thresholds:
            if t_a <= t_u:
                continue
            covered = risk < t_a
            if not covered.any():
                continue
            cov = covered.float().mean().item()
            if cov > best_usable_cov:
                best_usable_cov = cov
                best_pair = (float(t_u), float(t_a))
    if best_pair is None:
        return (1.0, 1.0)
    return best_pair
