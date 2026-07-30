---
title: feat: Streaming 3D Memory
type: feat
status: active
date: 2026-07-29
---

# Streaming 3D Memory

## Overview

Feature queue with confidence-gated fusion, producing stable video depth. Needs temporal decoder to consume fused features — cannot simply average DA2 feature maps.

## Key Decisions
- Feature queue (not ConvLSTM) — inspectable, resettable on scene cut
- Needs temporal decoder that consumes fused features and is initialized against per-frame baseline
- Alignment inherits same pose-scale, out-of-bounds, occlusion issues as test-time adaptation — PosePipeline untrained
- Needs distinct claim vs Video Depth Anything (calibrated gating, consumer-memory operating point)
- Establish video-manifest schema with trusted pose/flow, transform direction, resized intrinsics, masks
- v2 candidate: scene-cut detection, longer memory

## Implementation Units

- U0. **First slice: deterministic video evaluation harness + EMA control**
  Build harness on known-flow clips. Verify warping and masks. Implement 2-frame EMA as control: align only valid correspondences, reset on manual scene boundary, report lag/dynamic object errors + flicker. EMA is a non-claiming baseline. Promote to feature queue only after EMA exposes measurable stability/lag tradeoff.

- U1. **Video manifest schema**: Define schema with trusted pose/flow, transform direction (`T_target_from_source`), resized intrinsics, valid masks, scene identifiers. Build on `VideoManifestDataset` but add semantic validation.

- U2. **Temporal decoder**: Define trainable decoder that consumes fused features. Training loss, memory storage budget, stop-gradient rules, fallback when no valid prior exists.

- U3. **Confidence-gated fusion**: Fuse current + queued features based on uncertainty-weighted average. Accept only valid reprojections (in-bounds, non-occluded).

- U4. **Evaluation**: Flicker variance, temporal consistency error, per-frame depth quality, latency, memory, moving-object failure cases. Compare against per-frame DA2/SEF and EMA control. Video Depth Anything as related baseline — do not claim novelty unless distinct contribution demonstrated.
