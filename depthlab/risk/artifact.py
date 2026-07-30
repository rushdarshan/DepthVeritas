from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import torch

from depthlab.risk.calibrate import TriagePolicy, calibrate_thresholds
from depthlab.risk.risk_score import NormalizedCombiner


@dataclass
class SplitMetadata:
    calibration: str
    development: str
    test: str
    calibration_size: int = 0
    development_size: int = 0
    test_size: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> SplitMetadata:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class CalibrationArtifact:
    version: str
    created_at: str
    model_id: str
    combiner_state: dict
    threshold_usable: float
    threshold_abstain: float
    splits: SplitMetadata
    target_false_usable_rate: float
    achieved_coverage: float
    calibration_curve_risk: list = field(default_factory=list)
    calibration_curve_error: list = field(default_factory=list)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        data["splits"] = asdict(self.splits)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> CalibrationArtifact:
        data = json.loads(path.read_text(encoding="utf-8"))
        data["splits"] = SplitMetadata.from_dict(data["splits"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_triage_policy(self) -> TriagePolicy:
        return TriagePolicy(self.threshold_usable, self.threshold_abstain)


def build_calibration_artifact(
    risk: torch.Tensor,
    tile_errors: torch.Tensor,
    combiner: NormalizedCombiner,
    model_id: str,
    split_names: Dict[str, str],
    target_fur: float = 0.01,
    n_steps: int = 100,
) -> Optional[CalibrationArtifact]:
    result = calibrate_thresholds(risk, tile_errors, target_fur, n_steps)
    if result is None:
        return None
    t_u, t_a = result
    covered = (risk < t_a).float().mean().item()
    sorted_idx = torch.argsort(risk, descending=False)
    sorted_risk = risk[sorted_idx].cpu().tolist()
    sorted_err = tile_errors[sorted_idx].float().cpu().tolist()
    return CalibrationArtifact(
        version="1.0.0",
        created_at=datetime.now(timezone.utc).isoformat(),
        model_id=model_id,
        combiner_state=combiner.state_dict(),
        threshold_usable=t_u,
        threshold_abstain=t_a,
        splits=SplitMetadata(
            calibration=split_names.get("calibration", ""),
            development=split_names.get("development", ""),
            test=split_names.get("test", ""),
        ),
        target_false_usable_rate=target_fur,
        achieved_coverage=covered,
        calibration_curve_risk=sorted_risk,
        calibration_curve_error=sorted_err,
    )
