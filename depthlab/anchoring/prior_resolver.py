"""Explicit priority order for camera-height priors."""

from __future__ import annotations

from .config import AnchoringConfig


def resolve_camera_height(user_override: float | None, scene_type: str | None) -> float:
    if user_override is not None:
        return user_override
    return AnchoringConfig.from_scene_type(scene_type or "street")
