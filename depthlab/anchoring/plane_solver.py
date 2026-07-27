"""Robust ground-plane fitting and affine depth anchoring."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .config import AnchoringConfig


def fit_ground_plane(depth_map: np.ndarray, iterations: int = 128, seed: int = 42) -> Dict[str, np.ndarray | float]:
    """Fit ``z = ax + by + c`` to the lower image region with RANSAC."""
    depth = np.asarray(depth_map, dtype=np.float64)
    h, w = depth.shape
    yy, xx = np.mgrid[int(h * 0.55):h, 0:w]
    valid = np.isfinite(depth[int(h * 0.55):]) & (depth[int(h * 0.55):] > 0)
    x, y, z = xx[valid], yy[valid], depth[int(h * 0.55):][valid]
    if z.size < 3:
        raise ValueError("Not enough valid lower-image depth samples for plane fitting")
    rng = np.random.default_rng(seed)
    points = np.column_stack((x, y, np.ones_like(x)))
    best_coefficients, best_inliers = None, np.zeros(z.size, dtype=bool)
    scale = np.median(np.abs(z - np.median(z))) + 1e-6
    for _ in range(iterations):
        sample = rng.choice(z.size, 3, replace=False)
        coefficients, *_ = np.linalg.lstsq(points[sample], z[sample], rcond=None)
        inliers = np.abs(points @ coefficients - z) < max(0.05 * np.median(z), 2.5 * scale)
        if inliers.sum() > best_inliers.sum():
            best_coefficients, best_inliers = coefficients, inliers
    coefficients, *_ = np.linalg.lstsq(points[best_inliers], z[best_inliers], rcond=None)
    residual = float(np.mean(np.abs(points[best_inliers] @ coefficients - z[best_inliers])))
    return {"coefficients": coefficients, "residual": residual, "inlier_ratio": float(best_inliers.mean())}


def solve_scale_shift(plane: Dict[str, np.ndarray | float], depth_map: np.ndarray,
                      camera_height: float, focal_length_guess: float | None = None) -> Tuple[float, float]:
    """Solve affine depth scale from plane intercept and camera-height prior.

    The plane's bottom-centre depth provides a physically stable anchor; the
    returned shift is zero because one ground-height prior constrains scale only.
    """
    a, b, c = np.asarray(plane["coefficients"])
    h, w = depth_map.shape
    observed = a * (w - 1) / 2 + b * (h - 1) + c
    if not np.isfinite(observed) or observed <= 0:
        raise ValueError("Invalid plane-derived depth for anchoring")
    return float(camera_height / observed), 0.0


def compute_horizon_line(plane: Dict[str, np.ndarray | float], width: int) -> np.ndarray:
    a, b, c = np.asarray(plane["coefficients"])
    if abs(b) < 1e-8:
        return np.full(width, np.nan)
    x = np.arange(width)
    return -(a * x + c) / b


def anchor_scale(depth_map: np.ndarray, config: AnchoringConfig = AnchoringConfig()) -> Dict[str, object]:
    plane = fit_ground_plane(depth_map, config.ransac_iterations)
    scale, shift = solve_scale_shift(plane, depth_map, config.resolved_camera_height(), config.focal_length_guess)
    return {"depth": np.asarray(depth_map) * scale + shift, "scale": scale, "shift": shift, "plane": plane,
            "used_fallback": bool(plane["residual"] > config.plane_fit_threshold)}
