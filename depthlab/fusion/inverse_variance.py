"""Sequential inverse-variance fusion with an explicit disagreement gate."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch


def fuse_depth_measurements(
    depths: Sequence[torch.Tensor], variances: Sequence[torch.Tensor], *, gate_sigma: float = 3.0,
    valid_masks: Sequence[torch.Tensor] | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse aligned depth measurements and reject disagreements/invalid inputs."""
    if len(depths) < 1 or len(depths) != len(variances):
        raise ValueError("depths and variances must be non-empty sequences of equal length")
    if gate_sigma <= 0:
        raise ValueError("gate_sigma must be positive")
    mean = depths[0].clone()
    variance = variances[0].clamp_min(1e-8).clone()
    accepted = torch.isfinite(mean) & torch.isfinite(variance) & (mean > 0)
    if valid_masks is not None:
        if len(valid_masks) != len(depths):
            raise ValueError("valid_masks must match depths")
        accepted &= valid_masks[0].bool()
    for index, (depth, measurement_variance) in enumerate(zip(depths[1:], variances[1:]), start=1):
        measurement_variance = measurement_variance.clamp_min(1e-8)
        valid = torch.isfinite(depth) & torch.isfinite(measurement_variance) & (depth > 0)
        if valid_masks is not None:
            valid &= valid_masks[index].bool()
        consistent = (depth - mean).abs() <= gate_sigma * (variance + measurement_variance).sqrt()
        use = accepted & valid & consistent
        precision = variance.reciprocal() + measurement_variance.reciprocal()
        fused_variance = precision.reciprocal()
        fused_mean = fused_variance * (mean / variance + depth / measurement_variance)
        mean = torch.where(use, fused_mean, mean)
        variance = torch.where(use, fused_variance, variance)
    return mean, variance, accepted
