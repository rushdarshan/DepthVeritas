---
title: feat: Reprojection-Consistency Temporal Self-Training for DA2
type: feat
status: active
date: 2026-07-21
origin: docs/brainstorms/2026-07-21-da2-reprojection-consistency-requirements.md
---

# feat: Reprojection-Consistency Temporal Self-Training for DA2

## Overview

Train DA2-Small to produce temporally consistent depth on video by enforcing a geometric consistency loss — without optical flow. Depth predictions are unprojected to 3D, relative pose is estimated via PnP + RANSAC between adjacent frames, and depth is reprojected across views. Per-pixel residual between reprojected and predicted depth forms a self-supervised consistency signal. A motion mask (pixels above a reprojection-error threshold) excludes dynamic objects and occlusions. The pose branch is frozen; only DA2 is updated via `L_total = L_mono + λ * L_consistency`. Entire pipeline fits in 6GB VRAM on an RTX 4050.

## Hypothesis

Temporal consistency can be improved without optical flow by using depth reprojection as a self-supervised signal, because depth provides sufficient geometric correspondences for pose estimation.

Null hypothesis: Reprojection consistency provides no statistically significant improvement over vanilla DA2 on temporal metrics.

## Problem Frame

DA2 produces strong per-frame depth but treats each frame independently — no temporal smoothness, causing flicker and scale drift across video. Existing temporal-consistency methods (VeloDepth) require optical flow (+~6GB VRAM), exceeding consumer GPU budgets. Depth itself can provide the correspondences for pose estimation (DPV-SLAM validates this), creating a virtuous cycle: better depth → better pose → better consistency signal → better depth.

## Requirements Trace

- R1. Inference-only depth→pose pipeline via PnP + RANSAC on DA2 depth predictions
- R2. Depth reprojection + consistency loss with motion mask from reprojection-error threshold
- R3. Self-training loop on unlabeled video: DA2 forward → pose → consistency loss → DA2 update (pose frozen)
- R4. Evaluation suite: temporal metrics (flip count, flow-warp error), per-frame metrics (AbsRel, δ1, RMSE), cost metrics (FPS, VRAM, params, FLOPs, training time), ablation harness
- R5. 6GB VRAM constraint throughout: DA2-Small backbone, no optical flow, fp16, batch size 4–8, gradient accumulation

## Scope Boundaries

### In scope
- DA2-Small (25M params) only
- OpenCV `solvePnPRansac` for pose
- Self-training on DAVIS 2017 + YouTube-VOS 2019 clips
- Fixed pinhole intrinsics per dataset
- Eval on DAVIS/Sintel (temporal), KITTI/NYUv2 (per-frame)

### Out of scope
- Optical flow networks (core constraint for training/inference; pre-computed flow for offline evaluation metrics is separately permitted in U4)
- Learned pose estimation
- Real-time / online inference
- Multi-GPU / distributed
- Depth completion, inpainting, video diffusion
- Mobile/edge deployment
- Pre-training from scratch

### Deferred (post-8-week)
- DA2-Large backbone
- Test-time adaptation / online fine-tuning
- Learned motion segmentation from accumulated masks
- V-SLAM integration

## Implementation Units

### U1. Depth-to-Pose Pipeline

**Goal:** Implement PnP-based relative pose estimation from DA2 depth predictions on adjacent frame pairs.

**Requirements:** R1, R5

**Dependencies:** None (OpenCV + PyTorch only)

**Files:**
- Create: `models/pose_pipeline.py`
- Create: `utils/geometry.py` (unprojection, reprojection, scale alignment helpers)
- Test: `tests/test_pose_pipeline.py`

**Approach:**
1. Unproject depth map D_A to 3D points using known intrinsics K: `P_A = K^{-1} * [u, v, 1]^T * D_A[u, v]`
2. Align depth scales between A and B: compute median depth ratio over sampled pixels, rescale D_B to match D_A's scale. DA2 produces relative depth up to an unknown per-frame scale/shift — median alignment resolves this.
3. Sample a subset of points (e.g. 2000) from the unprojected point cloud using a uniform grid with small random jitter to avoid spatial bias.
4. Run OpenCV `solvePnPRansac` with PnP + RANSAC: given 3D points from D_A and 2D projections at frame B (from DA2 depth and intrinsics), estimate `T_{A→B} ∈ SE(3)`.
5. Return 6-DoF relative pose (rotation matrix + translation vector) plus the inlier mask from RANSAC.
6. Wrap in a `@torch.no_grad()` function — no gradients flow through pose estimation.
7. Batch over frame pairs by iterating sequentially (VRAM-safe, single-pair at a time).

**Key complexity handling:**
- **Failure detection:** if RANSAC inlier count < 20% of sampled points, mark the pair as degenerate (skip in training).
- **Scale ambiguity:** median-depth alignment before PnP; verify residual after alignment.
- **Depth noise:** RANSAC threshold of 3-5 pixels (tunable); sampled point grid spacing of ~8px reduces spatial correlation of errors.

**Test scenarios:**
1. **Synthetic cube rotation:** render a depth map of a cube at two known poses; verify recovered pose matches ground truth within 1° / 1% translation.
2. **KITTI odometry pairs (N=100):** run on pairs with known GPS/IMU pose; measure rotation error (°) and translation error (%). Expect rough alignment, not centimeter accuracy — PnP on monocular depth has inherent scale ambiguity.
3. **Noise injection:** add Gaussian noise to depth (σ = 0.05 * depth); verify RANSAC inlier rate drops gracefully and pose error degrades predictably.
4. **Empty / low-texture scene:** all points on a flat wall at constant depth; verify the pair is correctly flagged as degenerate (RANSAC fails).

**Verification:**
- `pytest tests/test_pose_pipeline.py -v` passes all 4 test scenarios.
- Sanity check: 100 KITTI pairs produce rotation error < 5° median (tolerance flag in test).

---

### U2. Consistency Loss and Motion Mask

**Goal:** Given predicted depth maps D_A, D_B and relative pose T_{A→B}, compute a per-pixel consistency loss with a motion mask that excludes dynamic regions.

**Requirements:** R2, R5

**Dependencies:** U1 (pose pipeline), `utils/geometry.py`

**Files:**
- Create: `losses/consistency_loss.py`
- Test: `tests/test_consistency_loss.py`

**Approach:**
1. Unproject D_A to 3D point cloud P_A (same as U1).
2. Transform P_A into frame B's coordinate system: `P_{A→B} = R * P_A + t` (from T_{A→B}).
3. Project P_{A→B} onto frame B's image plane: `p_B = K * P_{A→B}` (perspective projection).
4. For each valid reprojection (inside image bounds, positive z), sample D_B at the projected (u, v) via bilinear interpolation.
5. Compute per-pixel residual: `r(u,v) = |D_{A→B}(u,v) - D_B(u,v)| / D_B(u,v)` (relative residual, normalized by depth magnitude to handle scale).
6. **Motion mask** `M(u,v) = 1` where `r(u,v) < τ` (configurable threshold, default τ = 0.1 = 10% relative difference), else 0. Pixels above τ are occlusions, dynamic objects, or pose failures.
7. Consistency loss: `L_consistency = mean(r(u,v) * M(u,v))` — mean residual only over masked-in (static) pixels.
8. If valid pixel count < 10% of image area, mark the pair as degenerate (skip in training loop).

**Multiple pairs per clip:** For a clip of N frames, compute L_consistency for every adjacent pair (N-1 pairs) and mean-pool. Optionally add skip-1 pairs (A→C, B→D) at half weight for longer-range consistency — deferred to U5 ablation.

**Key complexity handling:**
- **Invalid reprojections:** pixels projecting outside frame B or with negative z are excluded from both loss and mask.
- **Bilinear interpolation:** use `torch.nn.functional.grid_sample` with `align_corners=False`.
- **Edge boundary:** erode the valid region by 4 pixels to avoid boundary artifacts from bilinear sampling.
- **Threshold τ:** start at 0.1 (10%); make it configurable; add an adaptive variant (per-frame percentile: τ = p85 of r(u,v)) for comparison in ablation.

**Test scenarios:**
1. **Perfect static scene:** synthetic plane at constant depth with zero pose difference; verify L_consistency ≈ 0 and mask is all-ones.
2. **Known translation:** synthetic scene with known 0.5m forward translation; verify loss is low (<0.05) and qualitatively correct.
3. **Dynamic object inserted:** place a small square at different depth in frame B vs. A; verify the square region is masked out (M=0) while static background is preserved (M=1).
4. **Occlusion boundary:** synthetic scene where right edge of frame A is not visible in frame B; verify boundary region is correctly masked out.
5. **Full-image motion (panning camera):** all pixels reproject correctly — verify mask stays near all-ones and loss is low.

**Verification:**
- `pytest tests/test_consistency_loss.py -v` passes all 5 test scenarios.
- Manual visual check: overlay motion mask on DAVIS frame pairs, confirm moving objects (cars, people) are masked.

---

### U3. Self-Training Loop

**Goal:** Implement the full training loop: load unlabeled video clips, compute consistency loss, update DA2 while keeping pose branch frozen.

**Requirements:** R3, R5

**Dependencies:** U1, U2, DA2 model definition + pretrained weights

**Files:**
- Create: `train.py` (main entry point — concise, <300 lines)
- Create: `data/da2_dataset.py` (video clip dataset)
- Create: `configs/train_config.yaml` (hyperparameters)
- Create: `utils/checkpoint.py` (save/load/resume)
- Modify: (none — DA2 is used as-is from its original repo)

**Approach:**
1. **Dataset (`data/da2_dataset.py`):** loads videos from DAVIS 2017 and YouTube-VOS 2019. Each item is a clip of N consecutive frames (N=5 default, tunable). Returns raw RGB frames (no depth GT needed). Frames are center-cropped or resized to DA2's input size (518×518). Intrinsics assumed per-dataset (pinhole, 70° horizontal FOV default). Implement lazy loading — load frames on demand, not all into memory.
2. **Training loop (`train.py`):**
   - Load DA2-Small pretrained weights; set to train mode.
   - Freeze batch norm statistics (DA2-Small uses BN; keep running stats fixed during self-training to avoid collapse on small clips).
   - Optimizer: AdamW (lr=1e-5, weight_decay=0.01). Learning rate found via LR range test on 5 clips.
   - For each batch (batch_size=4 clips of 5 frames = 20 total frames):
     a. Forward DA2 on all frames → depth maps.
     b. For each clip, for each adjacent pair: run U1 (pose, no grad), run U2 (consistency loss).
     c. `L_total = L_mono + λ * L_consistency` where `λ = 0.1` (initial; configurable).
        - U1 produces L_mono via DA2's original photometric or depth loss — actually, for self-training on *unlabeled* video, there is no ground-truth depth. Instead, use DA2's own prediction as the pseudo-target for L_mono: `L_mono = L1(D_pred, D_pred_detached)`. This is equivalent to a self-distillation that preserves the per-frame prior. The consistency loss is the only cross-frame signal. So `L_total = L1(D_pred, D_pred_detached) + λ * L_consistency`.
        - Alternative interpretation: L_mono is the original DA2 training loss (if available from the DA2 repo's training code as a frozen distillation target). Ponytail: use self-distillation (simpler, works without GT). `// ponytail: self-distillation for L_mono; swap for GT distillation if overfitting`.
     d. Backward + optimizer step. Clip gradients at max_norm=1.0.
     e. Log: L_total, L_consistency, motion-mask ratio (% valid pixels), inlier rate from PnP.
   - Mixed-precision (torch.cuda.amp) for forward + backward; pose estimation runs in fp32 on CPU (exported from GPU depth via `.cpu().numpy()`).
3. **Checkpointing:** save model weights, optimizer state, step counter, and best eval metric every 500 steps. Resume from latest checkpoint via `--resume` flag.
4. **Validation loop:** every 1000 steps, run on a held-out set of 10 DAVIS validation clips. Compute temporal metrics (flip count) to track improvement. No per-frame metrics during training (expensive).

**Key complexity handling:**
- **Degenerate pairs:** skip pairs flagged by U1 or U2. If >50% of pairs in a clip are degenerate, skip the entire clip.
- **Pose estimation on CPU:** export depth to numpy, run OpenCV on CPU, convert back. No GPU memory used.
- **VRAM budget breakdown (estimated):** DA2 forward = ~1.5GB, depth maps (5 frames @ 518×518 fp16) = ~2.6MB negligible, pose estimation = 0GB (CPU), optimizer states = ~0.5GB, activations for backward = ~3GB. Total ≈ 5GB, leaving ~1GB headroom on RTX 4050.
- **Data loading:** use PyTorch DataLoader with num_workers=2, pin_memory=True. Stream videos from disk (lazy per-frame loading) rather than pre-loading — DAVIS (~3GB) + YouTube-VOS subset (~10GB) totals ~13GB, leaving minimal headroom on a 16GB system alongside OS and CUDA libraries. Use `VideoFolder` with per-frame lazy reads.

**Test scenarios:**
1. **Overfit single clip:** train on one 5-frame clip for 200 steps; verify L_consistency decreases monotonically and flip count on that clip decreases.
2. **Synthetic static video:** 5-frame clip where all frames are identical (zero motion). Verify loss goes near zero and training doesn't diverge.
3. **Rapid motion clip:** one clip with extreme camera motion. Verify the training loop correctly skips degenerate pairs and doesn't crash.
4. **Resume from checkpoint:** kill training mid-epoch, resume; verify step counter and loss curve are continuous.

**Verification:**
- Training loss curves show L_consistency decreasing over steps (not diverging).
- Flip count on DAVIS val decreases (qualitative check, not a hard threshold — formal verification in U4).
- `python train.py --config configs/train_config.yaml` runs without OOM on RTX 4050 6GB; `nvidia-smi` confirms peak VRAM < 5.5GB.
- Resume works: `python train.py --resume checkpoints/latest.pt` picks up at correct step.

---

### U4. Evaluation Suite

**Goal:** Implement the metric computation for both temporal consistency and per-frame accuracy, and provide an ablation runner for controlled experiments.

**Requirements:** R4

**Dependencies:** U1, DA2 baseline (per-frame inference)

**Files:**
- Create: `eval/eval_temporal.py` (flip count, flow-warp error)
- Create: `eval/eval_perframe.py` (AbsRel, δ1, RMSE on KITTI/NYUv2)
- Create: `eval/ablation.py` (sweep runner)
- Create: `eval/metrics.py` (shared metric implementations with numpy/pure-PyTorch)
- Create: `tests/test_metrics.py`
- Create: `scripts/download_eval_data.py` (download KITTI raw + NYUv2 test splits)

**Approach:**

**Temporal metrics (`eval/eval_temporal.py`):**
- **Flip count:** for each pixel, count sign changes of depth gradient (dx, dy) across time. Normalize by number of frames and image area. Lower = more temporally consistent.
  - Compute depth gradients per frame (Sobel or central difference).
  - For pixel (i,j), count how many times `sign(∇D_t(i,j)) != sign(∇D_{t+1}(i,j))`.
  - Aggregate: mean flip count across all pixels and frame transitions.
  - Why gradients, not raw depth? Raw depth can shift in scale while being consistent; gradients are scale-invariant.
- **Flow-warp error:** use pre-computed optical flow (from PWC-Net or RAFT, run once as a pre-processing step — NOT during training). Warp D_t to frame t+1 using flow; compute EPE against D_{t+1}. This measures temporal consistency using flow as a reference, but flow is never needed during training.
  - Accept an optional `--flow-dir` argument pointing to pre-computed flow `.flo` files.
  - If flow not available, skip this metric (graceful degradation).

**Per-frame metrics (`eval/eval_perframe.py`):**
- Standard depth evaluation: AbsRel, SqRel, RMSE, RMSE_log, δ1/δ2/δ3.
- Align predicted depth to GT via median scaling (standard protocol for monocular depth).
- Evaluated on KITTI Eigen split and NYUv2 test split.

**Ablation runner (`eval/ablation.py`):**
- Sweeps over config parameters via CLI overrides:
  - `--consistency-loss` (on/off)
  - `--motion-mask` (on/off/adaptive)
  - `--lambda` (0.0, 0.05, 0.1, 0.2, 0.5)
  - `--clip-length` (3, 5, 9)
  - `--threshold-tau` (0.05, 0.1, 0.2, adaptive)
- Each run produces a JSON results file with timestamp.
- `--compare` flag loads multiple JSON results and produces a comparison table (stdout + markdown).

**Key design decisions:**
- `eval/metrics.py` contains pure compute functions (no I/O) — testable in isolation.
- Evaluation scripts are standalone CLI tools: `python eval/eval_temporal.py --pred-dir <> --output <>`. They read a directory of `.npy` depth maps + a `meta.json` with intrinsics.
- Flow-warp error is optional and detected by presence of `--flow-dir`.

**Test scenarios:**
1. **Synthetic constant depth:** video where all depths are identical; verify flip count = 0.
2. **Synthetic random noise:** video with i.i.d. random depth per frame; verify flip count ≈ 0.5 (chance-level sign flips).
3. **KITTI known depth:** on a few KITTI samples with GT depth; verify AbsRel matches DA2's reported numbers (~0.08 for DA2-Small).
4. **Ablation dry run:** `python eval/ablation.py --dry-run` prints all sweep combinations without running training.

**Verification:**
- `pytest tests/test_metrics.py -v` passes all scenarios.
- Per-frame metrics on DA2-Small baseline reproduce reported numbers within 0.5% relative.
- Temporal metrics are deterministic (same input → same output).

---

### U5. Full Pipeline and Ablations

**Goal:** Tie U1–U4 together; run the full self-training pipeline on DAVIS + YouTube-VOS; execute the ablation sweep; produce final results and analysis.

**Requirements:** R1, R2, R3, R4, R5 (Integration — individual implementations in U1–U4)

**Dependencies:** U1, U2, U3, U4

**Files:**
- Create: `scripts/run_full_pipeline.sh` (end-to-end training + eval orchestration)
- Create: `scripts/download_data.sh` (data acquisition)
- Create: `configs/ablations.yaml` (ablation sweep definitions)
- Create: `results/` directory structure (gitignored)
- Create: `ANALYSIS.md` (final results + interpretation)

**Approach:**

**Data acquisition (`scripts/download_data.sh`):**
- DAVIS 2017: wget from official server (trainval ~5GB).
- YouTube-VOS 2019: download subset (first 100 videos from train split, ~8GB).
- KITTI raw: download Eigen split subset (~12GB).
- NYUv2 test split: download (~1.5GB).
- Pre-compute optical flow for DAVIS val + Sintel using RAFT-small (run once on CPU or GPU with batch processing). Store as `.flo` files alongside frames. This is only for evaluation, never for training. Total: ~4GB for flow files.
- All downloads go to `data/raw/`; a `data/processed/` directory holds metadata JSONs listing valid clips.

**Full training run (`python train.py`):**
- 10 epochs over combined DAVIS + YouTube-VOS training set (~500 clips total).
- Batch size 4 (4 clips per batch), gradient accumulation ×2 for effective batch of 8.
- Clip length N=5.
- Validation every 1000 steps on DAVIS val (10 clips).
- Expected wall time: ~8 hours on RTX 4050 (10 epochs × ~125 steps/epoch × ~3 min/step = ~6h, with overhead ~8h).

**Checkpoint selection:**
- Choose the checkpoint with the lowest flip count on DAVIS val, not the lowest training loss (to avoid overfitting to the self-training objective at the expense of per-frame quality).

**Ablation sweep (via `eval/ablation.py`):**
- Run 7 training experiments in a controlled chain:
  1. **Baseline:** DA2 (no self-training)
  2. **Baseline + motion mask only:** mask applied, λ=0.0
  3. **Baseline + consistency loss only:** consistency on, no mask
  4. **Full pipeline:** mask + consistency, λ=0.1
  5. λ=0.05
  6. λ=0.2
  7. λ=0.5
- Each run limited to 5 epochs (→ ~4h each) for faster iteration.
- Parallelize across runs when GPU memory allows (swap checkpoints, not simultaneous — single-GPU, sequential).

**Final eval:**
- Each experiment run with 3 random seeds. Report mean ± std across seeds for every metric.
- Compute 95% confidence intervals via bootstrap (10,000 resamples) for primary metrics (flip count, AbsRel).
- For each experiment variant (baseline + 6):
  - Temporal: flip count on DAVIS val + Sintel
  - Flow-warp error on DAVIS val + Sintel (if flow files available)
  - Per-frame: AbsRel, δ1 on KITTI Eigen + NYUv2
  - Cost: FPS (inference, batch=1), peak VRAM, param count, FLOPs per frame, total training time
- Produce comparison table. Format as markdown for `ANALYSIS.md`.

**Success criteria check:**
1. Flip count improvement ≥15% vs DA2 baseline on DAVIS/Sintel.
2. AbsRel regression ≤1% relative on KITTI (i.e., if baseline is 0.080, final must be ≥0.0792).
3. Ablation confirms removing consistency loss drops improvement by >50% relative.
4. Motion mask qualitatively covers moving objects (visual check on 10 frames).

**GPU memory validation:**
- Run `nvidia-smi` every 100 steps during training, log peak VRAM.
- Verify peak < 5500 MB in the log.
- If OOM occurs: reduce batch size from 4 to 2, increase gradient accumulation from 2 to 4.

**Test scenarios:**
1. **Dry-run pipeline:** `python train.py --dry-run --num-steps 10` runs 10 steps without saving, verifies no crashes.
2. **Mini ablation:** `python eval/ablation.py --mini` only runs experiments 1, 2, 4 (fastest) — use during development.
3. **Data integrity:** `scripts/download_data.sh --verify` checks that all expected files exist and are not corrupted.

**Verification:**
- Full pipeline completes without error on RTX 4050 6GB.
- Comparison table shows ≥15% flip count improvement for the full method.
- Ablation shows consistency loss contributes >50% of the gain.
- Per-frame metrics on KITTI are within 1% of baseline DA2.

---

## Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Pose estimation | PnP + RANSAC via OpenCV | Zero learned params, zero VRAM cost (run on CPU) |
| No optical flow | Core design constraint | Enables single 6GB GPU — saves ~6GB vs VeloDepth |
| Motion mask | Reprojection-error threshold τ=0.1 | Emerges naturally, no supervision needed |
| Consistency loss | L1 relative depth residual | Scale-invariant, robust, standard |
| L_mono | Self-distillation (L1 to detached pred) | Works without GT depth; preserves per-frame prior |
| λ weighting | 0.1 initial | Keep consistency signal moderate; sweep in ablation |
| Scale alignment | Median-depth rescaling before PnP | Simple, effective, no learned params |
| PnP point sampling | Grid + jitter, N=2000 points | Balances accuracy vs runtime; grid avoids spatial clustering |
| Clip length | N=5 | Tradeoff: enough temporal context without blowing up VRAM (5 frames × batch 4 = 20 total forward passes) |
| Degenerate pair handling | Skip if RANSAC inliers <20% or valid reprojection <10% | Prevents bad pose from corrupting training |

## Risks & Dependencies

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| PnP fails on DA2 depth quality | Medium | High (blocker) | U1 test on 100 KITTI pairs before proceeding; fallback: use GT pose from KITTI odometry for initial experiments |
| Self-training collapses from noisy pose | Low | High | Degenerate-pair skipping + λ=0.1 keeps mono loss dominant; U3 overfit test catches this early |
| 6GB VRAM overflow | Medium | Medium | VRAM budget tracked in U3; fallback: batch size 2 + grad accum 4 |
| Per-frame accuracy regression | Low | Medium | Success criterion caps regression at 1%; frequent eval during training catches drift |
| No temporal improvement on rapid motion | Medium | Low | Natural limitation — benefit is on slow-to-moderate motion; document in ANALYSIS.md |
| DA2-Small does not benefit from self-training | Low | High | Sanity check: overfit single clip (U3 test 1) first; if no improvement, re-evaluate approach |

### Software Dependencies
- PyTorch ≥2.0, torchvision (record exact version in results)
- OpenCV Python (cv2) — `pip install opencv-python` (record version)
- NumPy, SciPy (record versions)
- DA2 weights: download from `depth-anything/Depth-Anything-V2-Small-hf` (Hugging Face) (record commit hash or model hash)
- Dataset downloads: DAVIS 2017, YouTube-VOS 2019, KITTI raw, NYUv2 test
- **Benchmark versioning:** record `pip freeze` output for every experiment; include CUDA version, cuDNN version, and driver version in results metadata.

### Hardware Dependencies
- Single RTX 4050 6GB (tested target)
- 16GB system RAM
- ~70GB free disk space (datasets + flow files + checkpoints + results)

## Risk Management

| Aspect | Detail |
|--------|--------|
| Expected outcome | 15% lower flicker (flip count), no AbsRel regression |
| Possible failure | PnP pose estimation unstable on DA2 relative depth |
| Fallback experiment | Freeze pose from KITTI GT odometry, evaluate with oracle poses |

## Threats to Validity

**Internal:** PnP success depends on static-scene assumption; moving objects bias the pose estimate.
**External:** Tests on DAVIS/Sintel (CG/video-focused) may not generalize to real-world video (noise, compression, rapid motion).
**Construct:** Flip count and flow-warp error measure temporal smoothness, not depth accuracy — improvement in one may not mean better depth.
**Statistical:** Single training run may reflect favorable initialization; 3 seeds with std reporting required.

## Computational Budget

| Component | Value |
|-----------|-------|
| Inference FPS (batch=1, RTX 4050) | TBD (target ≥15 FPS) |
| Training peak VRAM | <5500 MB |
| Total params (trained) | ~25M (DA2-Small) |
| FLOPs per forward pass | TBD (measure via `ptflops` or `fvcore`) |
| Total training time | ~8 hours (10 epochs, RTX 4050) |
| Pose estimation overhead | ~50 ms per pair (CPU, 2000 points) |

## Sources & References

- Origin: `docs/brainstorms/2026-07-21-da2-reprojection-consistency-requirements.md`
- DA2: Depth Anything V2 (Yang et al., 2024) — `depth-anything/Depth-Anything-V2-Small-hf`
- VeloDepth: Temporal consistency via optical flow (Dec 2025) — reference baseline, not used
- DPV-SLAM: Geometric consistency for tracking (CVPR 2024) — validates the core assumption
- OpenCV `solvePnPRansac`: `docs.opencv.org/4.x/d9/d0c/group__calib3d.html`
- DAVIS 2017: `davischallenge.org/davis2017/code.html`
- KITTI Eigen split: standard depth evaluation protocol
