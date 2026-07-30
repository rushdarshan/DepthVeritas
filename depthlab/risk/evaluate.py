from __future__ import annotations

from typing import Dict, Optional

import torch

from depthlab.metrics.calibration import _tiles
from depthlab.risk.artifact import CalibrationArtifact
from depthlab.risk.calibrate import TriageLabel


def evaluate_locked_test(
    artifact: CalibrationArtifact,
    combined_risk: torch.Tensor,
    tile_errors: torch.Tensor,
) -> Dict[str, float]:
    if combined_risk.ndim != 1:
        raise ValueError("combined_risk must be 1D (per-tile)")
    if tile_errors.ndim != 1:
        raise ValueError("tile_errors must be 1D (per-tile)")
    if combined_risk.shape != tile_errors.shape:
        raise ValueError("risk and tile_errors must have the same length")
    policy = artifact.to_triage_policy()
    labels = policy.classify(combined_risk)
    usable = labels == TriageLabel.USABLE.value
    review = labels == TriageLabel.REVIEW.value
    abstain = labels == TriageLabel.ABSTAIN.value
    fur = tile_errors[usable].float().mean().item() if usable.any() else 1.0
    coverage = (combined_risk < artifact.threshold_abstain).float().mean().item()
    usable_cov = usable.float().mean().item()
    abstain_rate = abstain.float().mean().item()
    review_rate = review.float().mean().item()
    return {
        "false_usable_rate": fur,
        "coverage": coverage,
        "usable_rate": usable_cov,
        "review_rate": review_rate,
        "abstain_rate": abstain_rate,
        "artifact_version": artifact.version,
    }


def evaluate_on_tiles(
    artifact: CalibrationArtifact,
    risk_map: torch.Tensor,
    error_map: torch.Tensor,
    tile_size: int = 32,
) -> Dict[str, float]:
    tile_risk = _tiles(risk_map, tile_size, agg="mean")
    tile_err = _tiles(error_map.float(), tile_size, agg="any")
    return evaluate_locked_test(artifact, tile_risk, tile_err)
