---
title: feat: Sparse Depth Fields — Radar-Style Depth Detections
type: feat
status: active
date: 2026-07-29
---

# Sparse Depth Fields

## Overview

Replace dense per-pixel depth with K=256 learned (x, y, depth, confidence) tuples. Model decides WHERE to predict and abstains elsewhere.

## Key Decisions
- Feature-grid head (not DETR) — reuses DA2 patch grid
- K=256 max, conf threshold can emit fewer
- Train with dense supervision before NMS — NMS is inference-only (non-differentiable)
- Add coverage/diversity objective so confidence cannot collapse to easy regions
- Primary metric: retained accuracy at fixed K and fixed spatial coverage, not sparse AbsRel alone
- Register head through BaseHead interface

## Implementation Units

- U0. **First slice: one-point-per-patch-cell (no offsets, no NMS)**
- Fixed one candidate per patch cell at cell center. Measure depth error and confidence calibration on valid cells at K=64/128/256. Add learned offsets and variable output count only if fixed-cell results show useful accuracy-coverage tradeoff.

- U1. **Feature-grid head**: Per cell (dx, dy, log_depth, confidence, uncertainty). CNN from DA2 patch grid. Candidate at each cell with bounded intra-cell offsets, positive depth, confidence, and uncertainty. Define mapping back to image pixels for non-square inputs.

- U1b. **Trainable objective (before NMS)**: Train candidate depth against dense valid ground truth. Add coverage/diversity objective — fixed-cell sampling so confidence cannot collapse to a few easy regions.

- U2. **Non-maximum suppression** (inference only): Rank by confidence, suppress near-duplicates, emit top K. Keep out of backward path. Include deterministic NMS tests.

- U3. **Abstention metrics**: Retained accuracy at fixed K and fixed spatial coverage, calibration of confidence, empty-output rate, coverage near edges/thin structures. Compare with dense DA2/SEF map thresholded by its own calibrated risk.

- U4. **Ablation**: K=64, 128, 256, 512. Compare against dense DA2/SEF at equal coverage.
