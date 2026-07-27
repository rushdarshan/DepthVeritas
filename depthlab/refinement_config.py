"""Typed settings for uncertainty-guided depth refinement."""

from dataclasses import dataclass


@dataclass(frozen=True)
class UncertaintyRefinementConfig:
    hidden_dim: int = 64
    beta: float = 10.0
    tau: float = 0.8
    neighbours: int = 5
    window_size: int = 7
