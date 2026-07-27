"""Differentiable local uncertainty-guided Laplace depth refinement."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def refine_depth(depth: torch.Tensor, uncertainty: torch.Tensor, features: torch.Tensor,
                 quantile: float = 0.8, temperature: float = 10.0, window_size: int = 7) -> torch.Tensor:
    """Blend uncertain pixels with feature-similar, confident local neighbours."""
    if window_size % 2 != 1:
        raise ValueError("window_size must be odd")
    b, _, h, w = features.shape
    threshold = torch.quantile(uncertainty.flatten(1), quantile, dim=1).view(b, 1, 1)
    gate = torch.sigmoid(temperature * (uncertainty - threshold))
    padding = window_size // 2
    padded_features = F.pad(features, (padding, padding, padding, padding), mode="reflect")
    padded_depth = F.pad(depth.unsqueeze(1), (padding, padding, padding, padding), mode="reflect")
    patches_f = F.unfold(padded_features, window_size).reshape(b, features.shape[1], window_size * window_size, h, w)
    patches_d = F.unfold(padded_depth, window_size).reshape(b, window_size * window_size, h, w)
    centre = features.unsqueeze(2)
    distance = (patches_f - centre).square().mean(1)
    local_uncertainty = F.unfold(F.pad(uncertainty.unsqueeze(1), (padding, padding, padding, padding), mode="reflect"), window_size).reshape(b, window_size * window_size, h, w)
    weights = F.softmax(-distance - local_uncertainty, dim=1)
    replacement = (weights * patches_d).sum(1)
    return depth * (1 - gate) + replacement * gate
