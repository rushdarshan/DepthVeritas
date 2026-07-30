---
title: feat: Physics-Constrained Depth
type: feat
status: active
date: 2026-07-29
---

# Physics-Constrained Depth

## Overview

Soft confidence-gated regularization constrains depth to respect physical priors (floor planarity). Use calibrated intrinsics to unproject candidates to 3D and fit plane in camera coordinates.

## Key Decisions
- Soft constraint (not hard projection) — hard constraint can flatten valid stairs/ramps
- Use calibrated intrinsics to unproject valid candidate pixels and fit 3D plane in camera coordinates (not pixel-coordinate affine fit)
- RANSAC inlier selection treated as detached (no gradient); compute residual in PyTorch
- Floor candidates from independently validated frozen segmenter or reviewed labels, not from lower-image heuristic
- Require inlier count, residual, area, and orientation gates; otherwise abstain
- Loss target defined with gradient path and bounded robust weight so it cannot dominate depth supervision
- No gravity-aligned normals in v1 (needs trained normal estimator)

## Implementation Units

- U0. **First slice: synthetic geometry test**
  Unproject synthetic planar floor with known intrinsics, recover its plane, produce zero residual. Then add detached-mask PyTorch residual on labelled floor pixels before using RANSAC on real imagery.

- U1. **Floor-plane detector**: Use calibrated intrinsics to unproject valid candidate pixels to 3D camera coordinates. Fit plane via PyTorch on inliers (RANSAC inlier selection detached). Output: plane params (camera coords), inlier ratio, plane_confidence.

- U2. **Planar regularization loss**: Huber penalty on depth residuals from fitted 3D plane. Gated by plane_confidence + inlier/area/orientation gates — only applies when fit is stable. Bounded robust weight so it cannot dominate depth supervision.

- U3. **Evaluation metrics**: Score against independently annotated/ground-truth planes (not plane re-fit from prediction). Report: plane violation rate, false accepts, false rejects, plane residual (median/p95), applicability fraction, depth guardrails (AbsRel, RMSE, δ1).
