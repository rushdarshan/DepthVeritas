"""Dependency-light depth and entropy PNG visualizations."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _colorize(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros((*values.shape, 3), dtype=np.uint8)
    lo, hi = np.percentile(values[finite], (2, 98))
    normalized = np.clip((values - lo) / max(hi - lo, 1e-8), 0, 1)
    return np.stack((255 * normalized, 255 * (1 - np.abs(2 * normalized - 1)), 255 * (1 - normalized)), axis=-1).astype(np.uint8)


def _save(array: np.ndarray, path: str | Path) -> None:
    from PIL import Image
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def save_depth_visualization(depth: np.ndarray, path: str | Path) -> None:
    _save(_colorize(np.asarray(depth)), path)


def save_entropy_visualization(entropy: np.ndarray, path: str | Path) -> None:
    _save(_colorize(np.asarray(entropy)), path)
