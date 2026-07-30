---
date: 2026-07-28
topic: self-adapting-depth
---

# Self-Adapting Depth: Requirements

## Problem Frame

Monocular depth models fail on unfamiliar scenes — the domain gap between training data (NYUv2 indoor) and deployment (any new room, lighting, camera) degrades accuracy. Current solutions (fine-tuning, domain adaptation) require labeled data or offline training. This project lets depth adapt to any new scene in seconds using only a short video, with zero labels.

## Key Decisions

- **Architecture:** External correction network (<2M params) predicts a log-depth residual. DA2 backbone stays frozen. Correction network resets per scene.
- **Network design:** 1×1 proj (384→64) + 3 depthwise-separable residual blocks (64ch). Two output heads: tanh-bounded log-depth residual + confidence gate. Corrected depth = base_depth * exp(gate × residual). Identity-initialized final layer → unadapted = exact base depth. Upsamples from patch resolution.
- **Reset:** Reset adapter to CPU-stored initial state per scene. No cross-scene caching in v1. Rolling back to best previous state during a single capture if a step worsens consistency.
- **Loss:** Hybrid — photometric reprojection loss (textured regions) + temporal depth consistency (textureless regions). Both already exist in DepthLab.
- **Use case:** Universal scene adaptation — wave phone camera for ~10 seconds, depth improves for that specific room.
- **Inference:** After adaptation, correction weights are scene-specific. Single-image inference uses adapted correction.

## Existing Infrastructure

- `depthlab/geometry.py` — `reproject_depth`, `pixels_to_camera`, `camera_to_pixels`, `sample_at_pixels` (all differentiable)
- `depthlab/temporal.py` — `PosePipeline` (relative pose regressor, 6-DOF SE3) — **UNTRAINED** (zero-initialized). Not usable for v1.
- `depthlab/losses/temporal.py` — `PhotometricConsistencyLoss` (SSIM + L1), `TemporalConsistencyLoss` (depth consistency over time)

## Dependencies

- **Pose source (critical):** PosePipeline is untrained. v1 prototype must use ARCore/ARKit poses, visual odometry (ORB-SLAM3, DPV-SLAM), or a separately trained pose model.
- Video stream: minimum 2 frames, but start with 5 keyframes selected from 10s capture (0.3-0.8s apart). Ablate at 2, 5, 8 keyframes.
- Camera intrinsics: needed for reprojection. If unknown, use approximate focal length (self-calibration extension).
- RTX 4050 6GB: adaptation must complete in <10s, peak VRAM <4GB (leaving room for backbone).

## Success Criteria

- After 10 seconds of video on an unseen room, AbsRel improves ≥10% relative to zero-shot DA2
- Adaptation completes within 10 seconds on RTX 4050
- Correction network ≤2M params, ≤50MB storage per scene
- No labels, no pretraining — runs entirely at deployment time
