from __future__ import annotations

from typing import Dict, List, Optional

import torch

from depthlab.metrics.calibration import _tiles


def tile_entropy_risk(
    entropy_map: torch.Tensor, tile_size: int = 32,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if valid_mask is None:
        return _tiles(entropy_map, tile_size, agg="mean")
    masked = entropy_map * valid_mask.float()
    tile_sum = _tiles(masked, tile_size, agg="sum")
    tile_count = _tiles(valid_mask.float(), tile_size, agg="sum")
    risk = tile_sum / tile_count.clamp_min(1e-8)
    risk[tile_count == 0] = 1.0
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
    if valid_mask is None:
        return _tiles(var, tile_size, agg="mean")
    masked = var * valid_mask.float()
    tile_sum = _tiles(masked, tile_size, agg="sum")
    tile_count = _tiles(valid_mask.float(), tile_size, agg="sum")
    risk = tile_sum / tile_count.clamp_min(1e-8)
    risk[tile_count == 0] = 1.0
    return risk


class NormalizedCombiner:
    def __init__(self, signals: Dict[str, torch.Tensor]) -> None:
        self.mins: Dict[str, float] = {}
        self.ranges: Dict[str, float] = {}
        self.shapes: Dict[str, torch.Size] = {}
        for name, tensor in signals.items():
            self.mins[name] = float(tensor.amin())
            self.ranges[name] = float(tensor.amax() - tensor.amin())
            self.shapes[name] = tensor.shape

    def normalize(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        tmin = self.mins.get(name)
        r = self.ranges.get(name, 1.0)
        if tmin is None:
            raise KeyError(f"unknown signal '{name}', known: {list(self.mins)}")
        if r < 1e-8:
            return torch.zeros_like(tensor)
        return (tensor - tmin) / r

    def combine(self, signals: Dict[str, torch.Tensor]) -> torch.Tensor:
        combined = None
        for name, tensor in signals.items():
            if name not in self.mins:
                raise KeyError(f"unknown signal '{name}', known: {list(self.mins)}")
            if tensor.shape != self.shapes[name]:
                raise ValueError(f"signal '{name}' shape {tensor.shape} != expected {self.shapes[name]}")
            norm = self.normalize(name, tensor)
            if combined is None:
                combined = norm
            else:
                combined = combined + norm
        if combined is None:
            raise ValueError("at least one signal required")
        return combined / len(signals)

    def state_dict(self) -> Dict:
        return {"mins": dict(self.mins), "ranges": dict(self.ranges),
                "shapes": {k: list(v) for k, v in self.shapes.items()}}

    @classmethod
    def from_state_dict(cls, d: Dict) -> NormalizedCombiner:
        obj = cls.__new__(cls)
        obj.mins = dict(d["mins"])
        obj.ranges = dict(d["ranges"])
        obj.shapes = {k: torch.Size(v) for k, v in d.get("shapes", {}).items()}
        return obj
