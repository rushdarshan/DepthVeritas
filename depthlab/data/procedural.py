"""Validation for reproducible procedural-render manifests, independent of Blender."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


REQUIRED_PROCEDURAL_FIELDS = {"scene_id", "seed", "renderer", "render_version", "asset_licenses", "depth_convention"}


def validate_procedural_manifest(records: Iterable[Dict[str, Any]]) -> List[str]:
    """Return deterministic validation errors for a procedural dataset manifest."""
    errors: List[str] = []
    seen: set[str] = set()
    for index, record in enumerate(records):
        missing = sorted(field for field in REQUIRED_PROCEDURAL_FIELDS if not record.get(field))
        if missing:
            errors.append(f"record {index}: missing {', '.join(missing)}")
        scene_id = str(record.get("scene_id", ""))
        if scene_id in seen:
            errors.append(f"record {index}: duplicate scene_id '{scene_id}'")
        seen.add(scene_id)
        if record.get("depth_convention") != "first_visible_surface":
            errors.append(f"record {index}: depth_convention must be first_visible_surface")
    return errors
