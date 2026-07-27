"""Losses for categorical Surface Existence Field depth prediction."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEFLoss(nn.Module):
    def __init__(self, center_weight: float = 0.1, entropy_weight: float = 0.0) -> None:
        super().__init__()
        self.center_weight = center_weight
        self.entropy_weight = entropy_weight

    def forward(self, preds: Dict[str, torch.Tensor], target: torch.Tensor,
                valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        probabilities = preds["bin_probs"]
        centers = preds["bin_centers"]
        if valid_mask is None:
            valid_mask = torch.isfinite(target) & (target > 0)
        if not valid_mask.any():
            return probabilities.sum() * 0
        distances = (target[..., None] - centers[:, None, None, :]).abs()
        labels = distances.argmin(dim=-1)
        log_probs = probabilities.clamp_min(1e-8).log()
        cross_entropy = F.nll_loss(log_probs[valid_mask], labels[valid_mask])
        center_grid = centers[:, None, None, :].expand_as(probabilities)
        nearest_centers = center_grid.gather(3, labels[..., None]).squeeze(-1)
        center_error = (nearest_centers[valid_mask] - target[valid_mask]).abs().mean()
        entropy = -(probabilities * log_probs).sum(dim=-1)[valid_mask].mean()
        return cross_entropy + self.center_weight * center_error + self.entropy_weight * entropy
