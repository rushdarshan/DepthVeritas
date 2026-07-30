"""DINO-style feature affinity probe against deterministic baselines."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def affinity_probe(features: torch.Tensor, token_depth: torch.Tensor, *, top_k: int = 1) -> Dict[str, float]:
    """Measure whether nearest feature neighbors have smaller depth differences.

    ``features`` is [B,N,C] and ``token_depth`` is [B,N]. This probe is
    intentionally unsigned: affinity alone cannot tell which point is nearer.
    """
    if features.ndim != 3 or features.shape[1] < 2 or token_depth.shape != features.shape[:2] or top_k < 1:
        raise ValueError("features [B,N,C], token_depth [B,N], and positive top_k are required")
    normalized = F.normalize(features, dim=-1)
    affinity = normalized @ normalized.transpose(1, 2)
    affinity.diagonal(dim1=1, dim2=2).fill_(-torch.inf)
    neighbors = affinity.topk(min(top_k, features.shape[1] - 1), dim=-1).indices
    selected_depth = token_depth.gather(1, neighbors.reshape(features.shape[0], -1)).reshape_as(neighbors)
    affinity_error = (selected_depth - token_depth.unsqueeze(-1)).abs().mean()
    adjacent = (token_depth[:, 1:] - token_depth[:, :-1]).abs().mean() if features.shape[1] > 1 else token_depth.new_zeros(())
    random_index = torch.arange(features.shape[1] - 1, -1, -1, device=features.device)
    random_error = (token_depth - token_depth[:, random_index]).abs().mean()
    return {"affinity_abs_depth_difference": float(affinity_error.item()), "adjacent_abs_depth_difference": float(adjacent.item()), "reversed_index_abs_depth_difference": float(random_error.item())}
