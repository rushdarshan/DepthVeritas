"""Tiny positive scale predictor used by the promptable metric-depth prototype."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class ScalePredictor(nn.Module):
    def __init__(self, input_dim: int = 768, use_patch_features: bool = False) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.use_patch_features = use_patch_features
        hidden = 32 if use_patch_features else 8
        self.network = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        nn.init.constant_(self.network[-1].bias, 0.54132485)  # softplus(bias) ~= 1
        if self.num_params > (60_000 if use_patch_features else 10_000):
            raise ValueError("ScalePredictor exceeds its intentional parameter budget")

    @property
    def num_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(self.network(features)).squeeze(-1)

    def save(self, checkpoint_path: str | Path) -> None:
        torch.save({"state_dict": self.state_dict(), "input_dim": self.input_dim,
                    "use_patch_features": self.use_patch_features}, checkpoint_path)

    @classmethod
    def load(cls, checkpoint_path: str | Path, map_location: str = "cpu") -> "ScalePredictor":
        state = torch.load(checkpoint_path, map_location=map_location)
        model = cls(state.get("input_dim", 768), state.get("use_patch_features", False))
        model.load_state_dict(state.get("state_dict", state))
        return model
