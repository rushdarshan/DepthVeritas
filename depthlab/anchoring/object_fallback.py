"""Object-prior fallback helpers. Detection is injected to avoid a hard GDINO dependency."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np


KNOWN_HEIGHTS_M = {"person": 1.7, "car": 1.5, "bus": 3.2, "truck": 3.0}


def solve_from_object_depth(depth_map: np.ndarray, bbox: Iterable[float], class_label: str,
                            focal_length: float) -> Dict[str, float]:
    x0, y0, x1, y1 = map(float, bbox)
    pixel_height = max(y1 - y0, 1.0)
    if class_label not in KNOWN_HEIGHTS_M:
        raise ValueError(f"No physical-size prior for '{class_label}'")
    observed = float(np.median(depth_map[int(y0):int(y1), int(x0):int(x1)]))
    expected_depth = KNOWN_HEIGHTS_M[class_label] * focal_length / pixel_height
    return {"scale": expected_depth / max(observed, 1e-8), "shift": 0.0, "residual": abs(expected_depth - observed)}


def select_best_candidate(candidates: Iterable[Dict[str, float]]) -> Optional[Dict[str, float]]:
    return min(candidates, key=lambda candidate: candidate["residual"], default=None)
