"""Configuration and deterministic prior selection for scale anchoring."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AnchoringConfig:
    camera_height: float | None = None
    scene_type: str | None = None
    fallback_enabled: bool = True
    plane_fit_threshold: float = 0.05
    focal_length_guess: float | None = None
    ransac_iterations: int = 128
    grounding_dino_confidence: float = 0.3

    @staticmethod
    def from_scene_type(scene: str) -> float:
        values = {"indoor": 1.5, "street": 1.6, "aerial": 20.0, "drone": 10.0}
        if scene not in values:
            raise ValueError(f"Unknown scene type '{scene}'. Available: {sorted(values)}")
        return values[scene]

    def resolved_camera_height(self) -> float:
        return self.camera_height if self.camera_height is not None else self.from_scene_type(self.scene_type or "street")
