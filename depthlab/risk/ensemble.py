"""3-member compact head ensemble for calibrated uncertainty.

Each ensemble member shares the backbone but uses an independent head with
distinct initialization, data ordering, and saved checkpoint. Variance across
members provides ensemble uncertainty.

Usage:
    ensemble = HeadEnsemble(head_factory, n_members=3, seeds=[42, 73, 91])
    outputs = ensemble(feature_bundle)  # returns {"depths": [...], "variance": Tensor}
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn


class HeadEnsemble(nn.Module):
    def __init__(
        self,
        head_factory: Callable[[int], nn.Module],
        n_members: int = 3,
        seeds: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        if seeds is None:
            seeds = [42 + 31 * index for index in range(n_members)]
        if len(seeds) != n_members:
            raise ValueError(f"expected {n_members} seeds, got {len(seeds)}")
        self.members = nn.ModuleList()
        for seed in seeds:
            torch.manual_seed(seed)
            self.members.append(head_factory(seed))

    def forward(self, feature_bundle: Any) -> Dict[str, torch.Tensor]:
        depths = []
        for member in self.members:
            out = member(feature_bundle)
            depths.append(out["depth"] if isinstance(out, dict) else out)
        stacked = torch.stack(depths, dim=0)
        return {
            "depths": depths,
            "mean": stacked.mean(dim=0),
            "variance": stacked.var(dim=0),
            "std": stacked.std(dim=0),
        }

    def load_member_state(self, state_dicts: List[Dict[str, torch.Tensor]]) -> None:
        if len(state_dicts) != len(self.members):
            raise ValueError(f"expected {len(self.members)} state dicts, got {len(state_dicts)}")
        for member, sd in zip(self.members, state_dicts):
            member.load_state_dict(sd)


def tile_ensemble_risk_from_members(
    ensemble_outputs: List[torch.Tensor],
    tile_size: int = 32,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    from depthlab.risk.risk_score import tile_ensemble_risk
    return tile_ensemble_risk(ensemble_outputs, tile_size, valid_mask)
