from __future__ import annotations

from typing import Dict, List, Optional

import torch

from depthlab.metrics.calibration import _tiles


def tile_entropy_risk(
    entropy_map: torch.Tensor, tile_size: int = 32,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if valid_mask is not None:
        entropy_map = entropy_map * valid_mask.float()
    risk = _tiles(entropy_map, tile_size, agg="mean")
    return risk


def tile_aleatoric_risk(
    variance_map: torch.Tensor, tile_size: int = 32,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return tile_entropy_risk(variance_map, tile_size, valid_mask)


def tile_ensemble_risk(
    ensemble_depths: List[torch.Tensor], tile_size: int = 32,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    stacked = torch.stack(ensemble_depths, dim=0)
    var = stacked.var(dim=0)
    if valid_mask is not None:
        var = var * valid_mask.float()
    return _tiles(var, tile_size, agg="mean")


class NormalizedCombiner:
    def __init__(self, signals: Dict[str, torch.Tensor]) -> None:
        self.ranges: Dict[str, float] = {}
        for name, tensor in signals.items():
            self.ranges[name] = float(tensor.amax() - tensor.amin())

    def normalize(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        r = self.ranges.get(name, 1.0)
        if r < 1e-8:
            return torch.zeros_like(tensor)
        tmin = float(tensor.amin())
        return (tensor - tmin) / r

    def combine(self, signals: Dict[str, torch.Tensor]) -> torch.Tensor:
        combined = None
        for name, tensor in signals.items():
            norm = self.normalize(name, tensor)
            if combined is None:
                combined = norm
            else:
                combined = combined + norm
        if combined is None:
            raise ValueError("at least one signal required")
        return combined / len(signals)
