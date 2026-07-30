---
title: feat: Self-Adapting Depth — Test-Time Scene Adaptation
type: feat
status: active
date: 2026-07-29
origin: docs/brainstorms/2026-07-28-self-adapting-depth-requirements.md
---

# Self-Adapting Depth

## Overview

Build a tiny external correction network (<2M params) that adapts depth predictions to any new scene in ~10 seconds using a short phone video, with zero labels and no pretraining.

---

## Problem Frame

Monocular depth fails on unseen scenes due to domain gap. Existing solutions need labeled data or offline training. This project adapts at deployment time using geometric consistency.

---

## Requirements Trace

- R1. External correction network (<2M params) predicts log-depth residual from frozen DA2 features
- R2. Hybrid loss: photometric reprojection (textured) + temporal depth consistency (flat regions)
- R3. Reset per scene — no cross-scene weight carryover
- R4. Use 5 keyframes from 10s capture with known poses
- R5. Adaptation completes within 10s on RTX 4050, peak VRAM <4GB
- R6. After adaptation, AbsRel improves ≥10% relative vs zero-shot DA2

---

## Scope Boundaries

- No training of PosePipeline — use external pose source for v1
- No metric depth — relative depth only
- No video dataset collection — use synthetic or existing short clips with known poses

---

## Context & Research

### Existing Code
- `depthlab/geometry.py` — differentiable reprojection, unprojection, sampling
- `depthlab/temporal.py` — PosePipeline (untrained)
- `depthlab/losses/temporal.py` — PhotometricConsistencyLoss, TemporalConsistencyLoss
- `depthlab/backbone/adapter.py` — feature extraction interface

---

## Key Technical Decisions

- **Pose source:** Use manifest-based known poses (VideoManifestDataset pattern). PosePipeline not used in v1.
- **Scale strategy:** Optimize one per-scene scale factor during adaptation. Do not use metric pose with relative depth without alignment. Unanchored output must not be called metric depth.
- **Keyframe selection:** By translation/parallax magnitude, not uniform timestamps. Reject pairs with tiny baseline or pure rotation.
- **Correction net:** Identity-initialized final layer → unadapted = exact base depth
- **Frozen backbone:** Cache DA2 features + base depth once per frame under inference mode before adaptation. No backprop through backbone.
- **Reset:** Manual scene reset (button or CLI flag). Feature-shift auto-detection deferred to v2.
- **Best-checkpoint:** Track aggregate unlabeled validation-pair loss; select best checkpoint, not per-step rollback.
- **Auto-masking:** Use minimum reprojection loss + auto-mask for dynamic objects, reflections, exposure changes.

---

## Implementation Units

- U1. **Correction network module**

**Goal:** Implement the external correction network.

**Files:**
- Create: `depthlab/adaptation/correction_net.py`
- Test: `tests/test_correction_net.py`

**Approach:**
- 1×1 conv projects DA2 features (384→64)
- Concatenate normalized log-depth + image gradient channels (2 ch)
- 3 depthwise-separable residual blocks (64 ch)
- Two heads: tanh-bounded log-depth residual + confidence gate (sigmoid)
- Identity-initialized final conv → zero output at init = exact base depth preserved
- Bilinear upsample from patch resolution to full resolution
- Forward: `corrected_depth = base_depth * exp(gate * residual)`

**Test scenarios:**
- Happy path: forward pass with random input produces output shape [B,1,H,W]
- Edge case: zero-initialized → output exactly equals base_depth
- Edge case: single-channel input still works

---

- U2. **Manifest-based frame and pose loader**

**Goal:** Load video frames with known poses and intrinsics using existing VideoManifestDataset pattern.

**Files:**
- Create: `depthlab/adaptation/manifest_loader.py`
- Test: `tests/test_manifest_loader.py`

**Approach:**
- Build on existing `VideoManifestDataset` which accepts image pairs, intrinsics, optional transforms, optional flow, optional depth
- Create manifest JSON with: frame paths, intrinsics matrix, pose matrix (`T_target_from_source`), resize-crop intrinsics update, quaternion convention, camera-coordinate convention
- Select 5 keyframes by translation magnitude (reject baseline < threshold)
- Missing intrinsics = error for v1 (not silent fallback)
- Output: (rgb, features_cached, base_depth_cached, pose, intrinsics) tuples — DA2 features extracted once at load time

**Test scenarios:**
- Happy path: load 5-frame manifest with known poses
- Edge case: fewer frames than requested → raise clear error
- Edge case: missing intrinsics → raise error (not silent default focal length)
- Edge case: all pairs have tiny baseline → raise error, no adaptation attempted

---

- U3. **Adaptation loop**

**Goal:** Run the optimization loop that adapts correction net weights to a scene.

**Files:**
- Create: `depthlab/adaptation/adapt.py`
- Test: `tests/test_adapt.py`

**Approach:**
- Use cached DA2 features + base depth (extracted once per frame under inference mode, no grad)
- Clone correction net from CPU initial state to GPU
- Optimize one per-scene scale factor alongside correction net weights
- For each keyframe pair (t, t+1):
  - Correction net → residual → corrected depth
  - External pose → SE3 transform (must define `T_target_from_source` convention)
  - `reproject_depth` → projected coords + target-frame z
  - Sample target depth at projected coords using `sample_at_pixels` (with proper in-bounds mask, not border padding)
  - Auto-mask: minimum reprojection loss across source frames; reject dynamic objects, reflections
  - `PhotometricConsistencyLoss` on valid (in-bounds, non-occluded) textured regions
  - `TemporalConsistencyLoss` comparing sampled target depth vs projected depth on valid regions
  - Weighted combined loss → backward → optimizer step
- Track best state by aggregate unlabeled validation-pair loss on held-out pair (not training pair)
- After N steps or patience, restore best checkpoint
- Write adapted weights + scale factor to scene directory

**Test scenarios:**
- Happy path: 5-frame adaptation converges (loss drops) within 100 steps
- Integration: full loop from frame load → adapted weights in <10s on RTX 4050
- Error path: NaN loss → skip step, continue
- Rollback: if step worsens consistency, previous state is restored

---

- U3b. **Geometry validity module**

**Goal:** Provide correct in-bounds mask, occlusion handling, and target depth sampling for the adaptation loss.

**Files:**
- Modify: `depthlab/geometry.py` (add functions)
- Test: `tests/test_geometry_validity.py`

**Approach:**
- `projected_coords_in_bounds(pixels, H, W) → mask`: binary mask for projected pixels inside the target frame
- `positive_z_mask(z) → mask`: binary mask for positive target-frame depth
- `occlusion_mask(depth_ref, depth_sampled, threshold) → mask`: mark pixels where sampled depth is significantly shallower than expected (occluded)
- `sample_target_depth(depth_target, pixels) → Tensor`: sample target depth at projected coordinates (unlike `sample_at_pixels` which samples RGB)
- `minimum_reprojection_mask(source_images, target_image, pixels, K) → mask`: per-pixel minimum over K source frames to handle dynamic objects

**Test scenarios:**
- Happy path: known 2-frame translation → correct in-bounds mask with no false positives
- Edge case: all pixels project outside → mask is all false
- Edge case: occlusion → occluded pixels correctly masked out

**Goal:** Manage scene identity, reset logic, and inference with adapted weights.

**Files:**
- Create: `depthlab/adaptation/scene_manager.py`
- Test: `tests/test_scene_manager.py`

**Approach:**
- `SceneManager` holds initial_state (CPU) and adapted_state (GPU)
- `reset()`: restore adapted_state from initial_state, clear optimizer
- `infer(rgb)`: run DA2 + correction net with current adapted weights
- Scene change detection: optional trigger (user button, large feature shift)
- Keep best adapted state during single capture; rollback on degradation

**Test scenarios:**
- Happy path: reset restores initial weights exactly (output = base depth)
- Edge case: infer before any adaptation → falls back to base depth
- Integration: adapt 5 frames, reset, adapt 5 different frames

---

- U5. **Evaluation script**

**Goal:** Measure AbsRel improvement and wall time.

**Files:**
- Create: `scripts/eval_adaptation.py`

**Approach:**
- Load scene with held-out test frames (not used in adaptation)
- Run DA2 zero-shot → measure AbsRel
- Run adaptation on 5 keyframes → measure wall time + peak VRAM
- Run adapted model on test frames → measure AbsRel
- Report: zero-shot AbsRel, adapted AbsRel, % improvement, wall_time_s, peak_vram_gb
- Ablate: 2, 5, 8 keyframes; with/without photometric loss, depth consistency loss

**Test scenarios:**
- Happy path: script runs end-to-end on synthetic scene
- Ablation: removing either loss term reduces improvement

---

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| Pose source unavailable | Use simulation with known ground-truth poses for development; add ARKit parser later |
| Adaptation >10s | Reduce keyframes from 5 to 3, or reduce optimizer steps |
| VRAM exceeds 4GB | Use fp16, gradient checkpointing, reduce batch size to 1 |
| Identity init not maintained | Verify in U1 test: zero-initialized → output = base depth exactly |
