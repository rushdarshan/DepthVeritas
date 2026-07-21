---
date: 2026-07-21
topic: da2-reprojection-consistency
---

# Reprojection-Consistency Temporal Self-Training (Zero Flow)

## Problem Frame

Depth Anything V2 produces strong per-frame monocular depth, but each frame is independent — no temporal smoothness constraint. This causes flickering, scale drift, and inconsistent geometry across video sequences. Existing temporal consistency methods rely on optical flow networks (+~6GB VRAM), making them impractical on a 6GB RTX 4050. Meanwhile, VeloDepth (Dec 2025) achieves SOTA temporal consistency but requires flow, pushing VRAM past consumer GPU limits.

Can we enforce temporal consistency *without* flow? Depth → 3D point cloud → PnP pose across frames → reproject depth → consistency loss. Depth itself provides the correspondences for pose estimation, creating a virtuous cycle. DPV-SLAM (CVPR 2024) validates that geometric consistency alone suffices for tracking.

---

## Requirements

### R1: Inference-only depth → pose pipeline
Implement a pre-training pass that, given DA2 depth predictions on a video clip, produces relative camera poses between adjacent frames via:
- Unproject depth to 3D points (known intrinsics)
- PnP (Perspective-n-Point) pose estimation between frame pairs
- Output: 6-DoF relative pose per adjacent pair

Must handle outlier rejection (RANSAC) and depth noise robustly. Batch processing, not real-time.

### R2: Depth reprojection & consistency loss
For each adjacent frame pair (A, B):
- Reproject frame A's depth into frame B's viewpoint using the estimated pose
- Define `L_consistency` = per-pixel depth residual between reprojected A→B depth and DA2's predicted depth at B
- Mask out pixels where reprojection error exceeds a configurable threshold (these are likely dynamic objects, occlusions, or pose failures — reuse as a motion mask)
- Loss applied only where mask is valid

### R3: Self-training loop on unlabeled video
- Sample clips from any unlabeled video dataset (YouTube, DAVIS, Sintel)
- Forward pass DA2 to get per-frame depth
- Run (R1) to get poses, (R2) to compute consistency loss
- Update DA2 via gradient descent on `L_total = L_mono + λ * L_consistency` (where L_mono is DA2's original per-frame depth loss — kept as the primary supervisory signal)
- Freeze pose estimation branch (no gradients through PnP)
- Iterate over epochs; periodically evaluate on held-out validation sequences

### R4: Evaluation suite
- **Temporal consistency:** flip count (per-pixel depth sign changes across frames), flow-warp error (using pre-computed flow as ground truth — flow not needed during training)
- **Per-frame accuracy:** AbsRel, δ1, RMSE on standard depth benchmarks (KITTI, NYUv2)
- **Ablations:** with/without motion mask, with/without consistency loss, varying λ
- **Comparison:** vs. per-frame DA2 baseline, vs. VeloDepth (using reported numbers or re-run if compute allows)

### R5: 6GB VRAM constraint
- DA2-Small (25M params) as the backbone — fits comfortably within 6GB
- No optical flow network at any point (saves ~6GB vs. VeloDepth)
- Batch size of 4–8 for training clips; gradient accumulation if needed
- Mixed-precision training (fp16)
- Model checkpointing to resume interrupted runs

---

## Success Criteria

1. Temporal consistency (flip count) on DAVIS/Sintel improves ≥15% over per-frame DA2 baseline.
2. Per-frame accuracy (AbsRel) on KITTI does not regress more than 1% relative (self-training should not hurt single-image quality).
3. Motion mask qualitatively identifies moving objects without supervision.
4. Full training pipeline fits within 6GB VRAM (RTX 4050).
5. Ablation confirms L_consistency contributes >50% of the temporal improvement.

---

## Scope Boundaries

### In scope
- DA2-Small as the only backbone (no DA2-Large, no ViT-g)
- PnP pose estimation via OpenCV (`solvePnPRansac`)
- Self-training on DAVIS + YouTube-VOS clips (easy to download, moderate size)
- Fixed camera intrinsics per dataset (no online calibration)
- Evaluation on DAVIS, Sintel (temporal), KITTI, NYUv2 (per-frame)

### Out of scope
- Optical flow networks (the whole point)
- End-to-end learned pose estimation (keep PnP fixed)
- Real-time / online inference
- Depth completion or inpainting
- Multi-GPU or distributed training
- Deploying to mobile / edge
- Video diffusion or video-depth models
- Pre-training from scratch (start from DA2 weights)

### Deferred (post-8-week)
- Scaling to DA2-Large
- Test-time adaptation
- Online fine-tuning on live camera feeds
- Learned motion segmentation from accumulated masks
- Integration with V-SLAM systems

---

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Backbone | DA2-Small (25M) | Fits 6GB VRAM with room for training |
| Pose estimation | PnP + RANSAC (OpenCV) | Zero learned parameters, no VRAM cost |
| Motion mask | Reprojection-error threshold | Emerges naturally, no extra supervision |
| Training data | DAVIS + YouTube-VOS | Curated video, moderate scale, per-frame GT for eval |
| Consistency loss | L1 depth residual | Simple, robust, standard |
| λ weighting | 0.1 (initial, tunable) | Keep mono loss dominant; schedule if needed |
| No optical flow | Core design constraint | Saves ~6GB VRAM, enables single-GPU training |
| Loss only on static regions | From motion mask | Prevents dynamic objects from corrupting depth |

---

## Dependencies / Assumptions

- OpenCV (Python bindings) for PnP + RANSAC
- DA2 weights available via Hugging Face / GitHub
- DAVIS 2017 train/val split (~90 videos)
- YouTube-VOS 2019 subset (~200 videos)
- PyTorch 2.x, CUDA 12.x
- Camera intrinsics known per dataset (or assumed pinhole with 70° FOV default)
- Assumption: depth predictions are consistent enough for PnP to find reasonable pose on *most* frames (self-training can bootstrap from imperfect pose)
- Assumption: static regions dominate the scene (>50%) so consistency loss has sufficient signal
- Assumption: scale ambiguity between frames is handled by aligning median depth (DA2 already produces relative depth; align via scale/shift per frame)

---

## Outstanding Questions

1. Does PnP converge on DA2-Small depth quality? Need a quick sanity test on 100 KITTI odometry pairs before committing.
2. What is the optimal motion-mask threshold? Adaptive (per-frame percentile) or fixed?
3. Does L_consistency help on rapid camera motion (large baseline) where reprojection overlap is small?
4. How many training frames per clip? 5? 9? 17? Tradeoff: more frames = more reprojection pairs = better consistency, but more VRAM.
5. Curriculum: start with easy clips (slow motion, static scenes) and gradually mix in harder ones?
6. Is scale alignment (median matching) sufficient, or do we need a learned scale head?
7. Does self-training collapse if pose estimation fails on a majority of frames in a clip?

---

## Next Steps

-> /ce-plan for structured implementation planning
