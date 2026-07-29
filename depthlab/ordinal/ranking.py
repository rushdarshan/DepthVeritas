"""Valid-pair ordinal depth objectives."""

from __future__ import annotations

from typing import Tuple

import torch


def sample_ordinal_pairs(depth: torch.Tensor, valid_mask: torch.Tensor | None = None, *, margin: float = 0.05, max_pairs: int = 2048) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return deterministic flattened index pairs whose target separation is meaningful."""
    flat = depth.reshape(depth.shape[0], -1)
    valid = (torch.isfinite(flat) & (flat > 0)) if valid_mask is None else valid_mask.reshape_as(flat).bool() & torch.isfinite(flat) & (flat > 0)
    left, right, direction = [], [], []
    for batch_index in range(flat.shape[0]):
        indices = valid[batch_index].nonzero(as_tuple=True)[0]
        if indices.numel() < 2:
            continue
        # Never materialize N choose 2 pairs for image-sized masks. This
        # deterministic lattice covers short and long separations within the
        # bounded training budget.
        pair_count = min(max_pairs, indices.numel() * (indices.numel() - 1) // 2)
        order = torch.arange(pair_count, device=depth.device)
        first_position = order.remainder(indices.numel() - 1)
        offset = order.div(indices.numel() - 1, rounding_mode="floor") + 1
        second_position = (first_position + offset).clamp_max(indices.numel() - 1)
        pairs = torch.stack((indices[first_position], indices[second_position]), dim=1)
        difference = flat[batch_index, pairs[:, 1]] - flat[batch_index, pairs[:, 0]]
        keep = difference.abs() > margin
        pairs, difference = pairs[keep][:max_pairs], difference[keep][:max_pairs]
        left.append(torch.stack((torch.full_like(pairs[:, 0], batch_index), pairs[:, 0]), dim=1))
        right.append(torch.stack((torch.full_like(pairs[:, 1], batch_index), pairs[:, 1]), dim=1))
        direction.append(difference.sign())
    if not left:
        empty = torch.empty((0, 2), dtype=torch.long, device=depth.device)
        return empty, empty, torch.empty((0,), device=depth.device)
    return torch.cat(left), torch.cat(right), torch.cat(direction)


def ordinal_ranking_loss(predicted_depth: torch.Tensor, target_depth: torch.Tensor, valid_mask: torch.Tensor | None = None, *, separation: float = 0.05, margin: float = 0.1, max_pairs: int = 2048) -> torch.Tensor:
    left, right, direction = sample_ordinal_pairs(target_depth, valid_mask, margin=separation, max_pairs=max_pairs)
    if direction.numel() == 0:
        return predicted_depth.sum() * 0
    predicted = predicted_depth.reshape(predicted_depth.shape[0], -1)
    signed_difference = predicted[right[:, 0], right[:, 1]] - predicted[left[:, 0], left[:, 1]]
    return torch.relu(margin - direction * signed_difference).mean()


def ordinal_accuracy(predicted_depth: torch.Tensor, target_depth: torch.Tensor, valid_mask: torch.Tensor | None = None, *, separation: float = 0.05, max_pairs: int = 2048) -> float:
    left, right, direction = sample_ordinal_pairs(target_depth, valid_mask, margin=separation, max_pairs=max_pairs)
    if direction.numel() == 0:
        return 0.0
    predicted = predicted_depth.reshape(predicted_depth.shape[0], -1)
    difference = predicted[right[:, 0], right[:, 1]] - predicted[left[:, 0], left[:, 1]]
    return float((difference.sign() == direction).float().mean().item())
