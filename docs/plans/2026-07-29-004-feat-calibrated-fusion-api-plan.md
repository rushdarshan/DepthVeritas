---
title: feat: Calibrated Fusion API — GPS-Style Depth
type: feat
status: active
date: 2026-07-29
---

# Calibrated Fusion API

## Overview

Depth output includes aleatoric + epistemic uncertainty for Kalman fusion. Over video, fused depth converges, uncertainty shrinks. v1 fuses two known-correspondence depth measurements in synthetic scene before scaling to real frames.

## Key Decisions
- Build on existing `depthlab/heads/uncertainty.py`
- Define fusion state explicitly: inverse depth for reprojected tracks, camera-frame point cloud, or world-frame voxel/point map. Specify transform and scale convention.
- Require known poses/intrinsics for v1, valid/occlusion masks, and reset on tracking loss
- No correlation_length in v1 — use scalar variance inflation factor. Model process noise, measurement variance, variance inflation, and gating. Fit calibration parameters on development split and freeze before testing.
- Ensemble variance empirically calibrated — shared-backbone heads are correlated, not automatically epistemic
- Compare: single-frame, temporal average, aleatoric-only fusion, ensemble+aleatoric fusion on held-out tracks with ground truth

## Implementation Units

- U0. **First slice: synthetic 2-frame fusion test**
  Fuse two known-correspondence depth measurements in synthetic scene with known noise and ground truth. Verify that uncertainty decreases only when measurements agree and occluded/outlier observation is rejected. Scale to 8 frames only after 2-frame test is correct.

- U1. **Measurement head**: Extend existing uncertainty head to output (mean_depth, aleatoric_log_variance, epistemic_variance_from_ensemble)
- U2. **3-member ensemble**: Train 3 compact heads with distinct initialization, data order, saved checkpoints on shared backbone. Empirically calibrate inter-head variance.
- U3. **Fusion**: 8-frame fusion with defined fusion state (inverse depth tracks / point cloud). Include process noise, measurement variance, variance inflation, gating. Report fused vs single-frame error, risk-coverage, uncertainty calibration.
