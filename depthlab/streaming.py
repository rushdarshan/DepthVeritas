"""Inspectable temporal controls used before a trainable memory decoder."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

import torch


REQUIRED_VIDEO_FIELDS = {"scene_id", "source_image_path", "target_image_path", "intrinsics", "T_target_from_source", "valid_mask"}


def validate_video_manifest(records: Iterable[Dict[str, Any]]) -> List[str]:
    errors: List[str] = []
    for index, record in enumerate(records):
        missing = sorted(field for field in REQUIRED_VIDEO_FIELDS if record.get(field) is None)
        if missing:
            errors.append(f"record {index}: missing {', '.join(missing)}")
    return errors


class MaskedEMA:
    """Alignment-ready EMA control that keeps current values outside valid priors."""
    def __init__(self, decay: float = 0.7) -> None:
        if not 0 <= decay < 1:
            raise ValueError("decay must be in [0, 1)")
        self.decay = decay
        self.value: torch.Tensor | None = None

    def reset(self) -> None:
        self.value = None

    def update(self, current: torch.Tensor, valid_prior: torch.Tensor | None = None, *, scene_cut: bool = False) -> torch.Tensor:
        if scene_cut or self.value is None:
            self.value = current.clone()
            return self.value
        valid = torch.ones_like(current, dtype=torch.bool) if valid_prior is None else valid_prior.bool()
        fused = self.decay * self.value + (1 - self.decay) * current
        self.value = torch.where(valid, fused, current)
        return self.value
