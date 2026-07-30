---
title: feat: DINOv2 Feature-Geometry Affinity Probe
type: feat
status: active
date: 2026-07-29
---

# Feature-Geometry Affinity Probe

## Overview

Reframed: single-image DINOv2 patch affinities do not encode ordinal depth direction (cosine/attention affinity is symmetric — it doesn't state which pixel is closer). Instead, probe whether nearest DINO-token neighbors have lower absolute depth difference and preserve depth discontinuities better than image-distance and random baselines.

## Key Decisions
- NOT an ordinal depth method — single-image affinity carries no signed direction
- Positioned as representation probe, not head replacement
- High-affinity pairs are often same-surface/same-object ties — not useful near/far comparisons
- For actual ordinal supervision, need two-frame feature matches + trusted pose + triangulation (sparse multi-view geometry, not zero-training)

## Implementation Units

- U0. **Reframed probe**: Measure whether nearest DINO-token neighbors have lower absolute ground-truth depth difference and preserve depth discontinuities better than image-distance and random baselines. Only explore multi-view correspondence if this narrower representation result is positive.

- U1. **Affinity probe**: Extract DINOv2 patch features. Compute pairwise cosine affinity matrix. Measure absolute depth difference of top-k affinity pairs vs random and image-distance baselines. Report boundary preservation at depth discontinuities.

- U2. **Baseline comparison**: Compare against image-distance baseline and random baseline on pre-registered tied threshold, cosine normalization, resize/token mapping.

## Evaluation Gate
Stop if affinity does not beat simple baselines by a pre-registered margin on depth-difference and boundary-preservation metrics.
