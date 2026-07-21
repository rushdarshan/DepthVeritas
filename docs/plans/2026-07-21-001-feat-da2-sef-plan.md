---
title: feat: Surface Existence Field for Depth Anything V2
type: feat
status: active
date: 2026-07-21
origin: docs/brainstorms/2026-07-21-da2-sef-requirements.md
---

# feat: Surface Existence Field for Depth Anything V2

## Overview
SEF replaces DA2's scalar depth output with a learned categorical distribution over N depth bins, solving two gaps jointly: transparent-surface handling and calibrated uncertainty.

---

## Problem Frame
DA2 produces high-quality single-depth estimates but fails systematically on transparent surfaces (glass, water) and reflective/specular surfaces where multiple physical depths correspond to one pixel. It also provides zero uncertainty calibration — equally confident on ambiguous reflections as on solid walls. This erodes trust in downstream planners and filters. SEF replaces per-pixel scalar depth with a learned categorical distribution over N depth bins, capturing multi-modal depth ambiguity and producing calibrated uncertainty via softmax entropy.

## Hypothesis

[TBD — finalized after gap analysis]

---

## Requirements Trace
| Req | Description | Verifies | Unit |
|-----|-------------|----------|------|
| R1 | Classification head over N learned depth bins per pixel | Output shape B×H×W×N, softmax sums to 1 | U1 |
| R2 | Learned bin centers via pooled-feature MLP (AdaBins-style) | Bin centers differ per image, monotonic increasing | U1 |
| R3 | Encoder frozen during head warmup, then end-to-end fine-tune | Two-stage training configs | U5 |
| R4 | Cross-entropy + bin-center regression loss | L1 between predicted/optimal bin centers < 0.1 after warmup | U2 |
| R5 | Optional entropy regularization | Config toggle, loss increases on peaked distributions | U2 |
| R6 | Standard depth dataset pre-train + transparent mix fine-tune | Training curves on KITTI, NYUv2, transparent subset | U3, U5 |
| R7 | Transparent/reflective evaluation subset (200–500 frames) | Curated scene list from Hypersim/Replica/TORSD | U3 |
| R8 | Uncertainty metrics: ECE, AURC, NLL | Script produces all three per evaluation run | U4 |
| R9 | Per-pixel entropy map visualization | Entropy overlay images saved during eval | U4 |
| R10 | Comparison table vs DA2 + MC Dropout | AbsRel, RMSE, δ1.25, ECE, AURC across all methods | U4 |
| R11 | Inference depth = marginal expectation over bins | Forward pass produces both depth map and entropy map | U5 |
| R12 | N=96 default, ~4M added params, fits 6GB VRAM | fp16 + grad accum ≥ 2 at 384×384 | U1, U5 |
| R13 | Report FPS, peak VRAM, param count, FLOPs, training wall time alongside accuracy | Measured during eval | U4 |
| R14 | Run every experiment with 3 seeds; report mean ± std; include 95% CI where feasible | Metrics show variance | U4 |

---

## Scope Boundaries
- No changes to the DINOv2 encoder or DPT feature-fusion core — only the final prediction head and loss
- No real-time / mobile deployment target — single-RTX-4050 offline evaluation only
- No synthetic-data generation pipeline for transparent surfaces — only existing dataset curation
- No comparison against Deep Ensembles, SDE, or other SOTA uncertainty methods — MC Dropout only
- No video / temporal consistency — single-frame depth only

---

## Implementation Units

### U1. SEF Decoder Head Implementation
**Goal:** Replace DA2's final regression layer with a classification head that outputs per-pixel softmax probabilities over N depth bins with learned bin centers.

**Requirements:** R1, R2, R3 (head architecture), R12 (4M param budget)

**Dependencies:** None (standalone module, testable with dummy features)

**Files:**
- Create: `models/sef_head.py` — SEFHead (AdaBins-style MLP + bin-width predictor + depth classification head)
- Create: `models/sef_utils.py` — bin helper functions (center-to-boundary, sample bin centers from features, monotonicity projection)
- Modify: `models/da2_wrapper.py` — add `mode='regression' | 'sef'` config, instantiate SEFHead when mode='sef'
- Create: `tests/test_sef_head.py` — unit tests

**Approach:**
1. Build `SEFHead(n_bins=96, hidden_dim=256)` with three submodules:
   - `BinCenterPredictor`: global avg pool → 2-layer MLP (hidden_dim → n_bins) → softmax over bin widths → cumulative sum → N+1 bin boundaries per image. Repeated to per-pixel bin centers via broadcasting.
   - `DepthClassifier`: 1×1 conv from DPT feature channels (256) → N logits + softmax per pixel.
   - Forward: returns `(bin_probs: B×H×W×N, bin_centers: B×N)`.
2. Bin center computation: predict N+1 boundaries b_0...b_N where b_0=min_depth, b_N=max_depth. Learn normalized widths δ_k via MLP, then `b_k = min_depth + (max_depth - min_depth) * sum(exp(δ_1:k)) / sum(exp(δ_1:N))`. Clamp boundaries to [min_depth, max_depth] and project to monotonic increasing.
3. Parameter budget: BinCenterPredictor (~100K params from MLP), DepthClassifier (4N + 1×1 conv weights ≈ 256×N), total ~4M at N=96.
4. Integration: DA2Wrapper detects `head_mode='sef'` and routes DPT fusion features through SEFHead instead of the original regression conv. Original DA2 decoder remains loadable via `head_mode='regression'`.
5. fp16 compatibility: all SEF ops are standard torch ops (linear, conv2d, softmax, cumsum) — no custom CUDA kernels needed.

**Patterns to follow:** DA2 official repo conventions (config dicts, `build_*` factory functions, `nn.Sequential` for simple stacks). AdaBins `AdaBinHead` MIT-licensed reference for bin-center MLP structure.

**Test scenarios:**
- Happy path: forward pass with dummy features (B=2, C=256, H=32, W=32) returns `bin_probs` shape (2, 32, 32, 96) with softmax summing to 1.0 along last dim. `bin_centers` shape (2, 96) with monotonic increasing values.
- Edge case: N=1 collapses to a single bin → softmax always 1.0, marginal expectation = bin center value, entropy = 0. Equivalent to regression baseline.
- Edge case: N=96 uniform logits → max entropy posterior.

**Verification:** `python tests/test_sef_head.py` passes all 3 scenarios.

---

### U2. Training Harness and Loss
**Goal:** Implement the combined loss function (R4), optional entropy regularization (R5), and a training harness for the SEF head warmup stage.

**Requirements:** R4, R5

**Dependencies:** U1 (head must be instantiable)

**Files:**
- Create: `models/sef_loss.py` — SEFLoss (CE + bin-center regression + optional entropy regularization)
- Create: `train_warmup_sef.py` — warmup training script (encoder frozen, head only)
- Create: `configs/train_warmup_sef.yaml` — training config
- Create: `tests/test_sef_loss.py` — loss function unit tests

**Approach:**
1. `SEFLoss` computes three terms:
   - **Cross-entropy term** `L_ce`: For each valid pixel, GT depth d is binned into the predicted bin centers — find nearest bin center via `argmin(|d - centers|)`, then one-hot target. Mask invalid-GT pixels (mask in loss or assign uniform target across bins). Cross-entropy between predicted softmax and one-hot target, averaged over valid pixels.
   - **Bin-center regression term** `L_center`: After assigning each valid GT depth to its nearest predicted bin center, compute L1 distance between GT depth and assigned bin center. This pulls centers toward actual GT depth values. Weight λ_c = 0.1 relative to L_ce.
   - **Entropy regularization term** `L_ent` (optional, R5): `mean(H[p])` over valid pixels, weighted by λ_e. Toggle via config. Default λ_e = 0 (off). Enable only if validation ECE > 0.05 after warmup.
2. Total: `L = L_ce + λ_c * L_center + λ_e * L_ent`.
3. Invalid-GT handling: create binary mask from GT depth validity (valid_depth > 0). Masked pixels contribute nothing to L_ce and L_center. For entropy regularization, unmasked pixels contribute normally.
4. `train_warmup_sef.py`: single-GPU training loop:
   - Load pretrained DA2-Small checkpoint, freeze DINOv2 encoder + DPT fusion (requires_grad=False).
   - Instantiate SEFHead, optimizer (AdamW, lr=1e-4) on head params only.
   - Train on standard depth datasets (KITTI, NYUv2) for N_epochs (configurable, default 20).
   - Validate periodically, save best checkpoint by validation loss.
   - Gradient accumulation: start at accum_steps=4, step size 384×384, batch_size=4 → effective batch 16.
5. Mixed precision: torch.cuda.amp.autocast for all forward/backward, scaler for gradient.

**Patterns to follow:** DA2 training scripts (argparse + yaml config pattern), PyTorch `train_one_epoch` / `validate` loop structure.

**Test scenarios:**
- Happy path: dummy predicted distribution (B=2, H=4, W=4, N=10) + random GT depths. Loss is positive, finite, and backprop produces gradients on all head params. No gradient flow into encoder (frozen).
- Edge case: all GT depths masked → loss = 0, no gradient.
- Edge case: uniform predicted distribution → high cross-entropy loss, high entropy.
- Entropy reg: L_ent toggle toggles loss value. With λ_e > 0, peaked distribution gives lower loss than uniform (because L_ent penalizes high entropy less).

**Verification:** `python tests/test_sef_loss.py` passes. Warmup training runs 1 epoch to convergence on a toy dataset (NYU 50-image subset, 5 min on RTX 4050).

---

### U3. Data Pipeline
**Goal:** Prepare training and evaluation data: standard depth datasets (R6) and a curated transparent-surface evaluation subset (R7).

**Requirements:** R6, R7

**Dependencies:** None (data download can parallelize with U1/U2)

**Files:**
- Create: `data/transparent_subset.py` — scene/camera selection from Hypersim/Replica/TORSD metadata
- Create: `data/depth_dataset.py` — unified depth dataset loader (KITTI, NYUv2, transparent subset)
- Create: `data/depth_collate.py` — custom collate with valid-depth masking
- Create: `configs/dataset_paths.yaml` — local paths for all datasets
- Create: `scripts/download_transparent_subset.py` — automated download script
- Create: `tests/test_depth_dataset.py` — dataset loader unit tests

**Approach:**
1. Standard datasets (R6):
   - **KITTI**: Raw dataset, Eigen split. Load depth maps from `proj_depth/groundtruth/`. Resize to target resolution (384×384), crop valid area. ~23K training images.
   - **NYUv2**: Labeled dataset, official split. Load depth from H5 files. ~47K training images. Center crop to 384×384.
   - Both wrapped in `DepthDataset(paths, split, aug_params)` with photometric augmentation (color jitter, gamma, brightness).
2. Transparent subset (R7):
   - **Hypersim**: Search metadata for scenes tagged with `glass`, `mirror`, `window`, `water` in semantic labels. Select 80–150 frames with high glass/mirror pixel fraction.
   - **Replica**: Scan scene list for objects containing transparent surfaces. Render-depth + semantic maps for 50–100 frames.
   - **TORSD (Transparent Object RGB-D)**: Already a transparent-object dataset — includes glass bottles, cups, windows. 50–100 frames.
   - Total target: 200–500 frames. Output: JSON manifest file with (scene_id, frame_idx, dataset, depth_path, rgb_path, glass_pixel_frac).
   - Evaluation split: 20% of transparent subset held out for calibration metrics.
3. Dataset output: `DepthDataset.__getitem__` returns `(rgb_tensor: 3×H×W, depth_tensor: H×W, valid_mask: H×W)`. Resize to training resolution. Normalize RGB to ImageNet stats.
4. Transparent subset integration: during SEF fine-tune stage, mix transparent frames into each batch at ratio r (configurable, default 0.3).

**Patterns to follow:** torchvision `ImageFolder` pattern for dataset discovery; DA2 data loading (PIL → tensor, depth scaling, intrinsics for KITTI crop bounding box).

**Test scenarios:**
- Happy path: `DepthDataset` returns correct shapes for a single KITTI sample.
- Transparent subset JSON: load manifest, verify each entry's depth_path and rgb_path exist on disk.
- Mask validity: valid_mask is a boolean tensor with `True` where depth values are positive and finite.
- Augmentation doesn't affect depth alignment: test that color jitter preserves depth values.

**Verification:** `python tests/test_depth_dataset.py` passes. Transparent subset JSON manifest has ≥200 entries. Dataset loads 1 epoch of KITTI without OOM (~30s).

---

### U4. Evaluation and Visualization
**Goal:** Implement uncertainty calibration metrics (R8), entropy visualization (R9), and comparison table generation (R10).

**Requirements:** R8, R9, R10, R13, R14

**Dependencies:** U1, U2 (needs trained head and loss)

**Files:**
- Create: `eval/calibration_metrics.py` — ECE (adaptive vs uniform binning), AURC, NLL computation
- Create: `eval/entropy_visualizer.py` — per-pixel entropy color overlay + depth map comparison
- Create: `eval/comparison_table.py` — produce LaTeX comparison table (DA2, DA2+MC Dropout, SEF)
- Create: `eval/run_eval.py` — orchestrator: load model → run inference → compute all metrics → save results
- Create: `eval/mc_dropout_baseline.py` — MC Dropout wrapper for DA2
- Create: `configs/eval_config.yaml` — eval parameters
- Create: `tests/test_calibration_metrics.py` — metric unit tests

**Approach:**
1. Calibration metrics (R8):
   - **ECE**: Bin predictions by confidence (max softmax probability) into M bins (M=15). For each bin, compare accuracy (fraction of bins where GT falls into the predicted bin) vs confidence. ECE = sum(w_k * |acc_k - conf_k|). Implement both uniform-width and adaptive-width bin variants.
   - **AURC**: Sort pixels by uncertainty (1 - max prob, or entropy). Compute risk (error rate) as function of coverage fraction. AURC = area under risk-coverage curve (lower is better). Normalize by optimal AURC (Oracle AURC) for AUROC-style comparison = AURC / AURC_oracle.
   - **NLL**: For each valid pixel, NLL = -log(predicted_prob_of_bin_containing_GT). Average over valid pixels. Lower is better.
   - All metrics computed per-image, then macro-averaged over eval split.
2. Entropy visualization (R9):
   - Entropy per pixel `H[p] = -sum(p_k * log(p_k + eps)) / log(N)` (normalized to [0, 1]).
   - Overlay entropy on RGB using jet colormap (red = high uncertainty, blue = low).
   - Output side-by-side comparison: [RGB, predicted depth, GT depth, entropy overlay].
3. MC Dropout baseline (R10):
   - Enable dropout at inference on DA2's DPT fusion layers. Run T stochastic forward passes (T=20), compute mean depth and pixel-wise variance.
   - Variance → uncertainty proxy. Compare uncertainty calibration metrics against SEF.
4. Comparison table (R10):
   - Rows: KITTI Eigen test, NYUv2 test, Transparent subset.
   - Columns: AbsRel ↓, RMSE ↓, δ1.25 ↑, ECE ↓, AURC ↓, NLL ↓.
   - Three method columns: DA2 (baseline), DA2+MC Dropout, SEF (ours).
   - Output as LaTeX table and as CSV.
5. **Cost metrics (R13):** Report FPS (frames/sec at inference), peak VRAM, parameter count, FLOPs, and training wall time alongside accuracy metrics.
6. **Multi-seed reporting (R14):** Run every experiment with 3 different random seeds; report mean ± std for all metrics. Include 95% confidence intervals where feasible.
7. **Ablation chain:** Compare: Baseline → Baseline + SEF head → Baseline + SEF head + entropy reg → Full SEF pipeline. Each ablation step reports all metrics on the frozen benchmark.
8. **Per-stratum analysis:** Report per sub-stratum improvement (glass, mirror, water, etc.) rather than aggregate only.

**Patterns to follow:** uncertainty-quantification-benchmark conventions for ECE/AURC. AdaBins evaluation script for standard depth metrics.

**Test scenarios:**
- ECE: perfect calibration (predicted distribution matches empirical accuracy) → ECE = 0. Overconfident distribution → ECE > 0.
- AURC: perfect predictions (all GT in highest-confidence bin) → AURC = 0. Random predictions → AURC = max.
- NLL: prediction that assigns probability 1 to correct bin → NLL = 0 (per-sample).
- Visualizer: forward pass with dummy output, saved entropy overlay is valid PNG with correct dimensions.

**Verification:** `python tests/test_calibration_metrics.py` passes. `python eval/run_eval.py --model sef --checkpoint <path>` produces a complete metrics JSON + entropy overlay images on a 50-image eval subset within 10 minutes.

---

### U5. Full Pipeline Integration
**Goal:** Wire all components into a complete training→evaluation pipeline, produce final checkpoints, and generate the comparison table.

**Requirements:** R3 (two-stage training), R6 (end-to-end fine-tune), R11 (inference depth = marginal expectation), R12 (memory within 6GB)

**Dependencies:** U1 (head), U2 (loss + warmup), U3 (data), U4 (eval)

**Files:**
- Create: `train_end_to_end.py` — full fine-tune script (all params unfrozen)
- Create: `configs/train_end_to_end.yaml` — fine-tune config
- Create: `inference_sef.py` — inference script: load checkpoint → marginal expectation → entropy → save outputs
- Create: `configs/final_sef.yaml` — combined pipeline config
- Create: `run_pipeline.sh` — orchestrator: warmup → end-to-end → eval → table
- Modify: `models/da2_wrapper.py` — add `SEFWrapper` that combines encoder + DPT + SEFHead with marginal depth and entropy outputs

**Approach:**
1. `SEFWrapper` in da2_wrapper.py:
   - Forward pass: DINOv2 encoder → DPT fusion → SEFHead → (bin_probs, bin_centers).
   - Post-process: `depth = (bin_probs * bin_centers.unsqueeze(1).unsqueeze(2)).sum(dim=-1)` (marginal expectation, R11). `entropy = - (bin_probs * (bin_probs + 1e-8).log()).sum(dim=-1) / log(N)`.
   - Returns dict `{'depth': B×H×W, 'entropy': B×H×W, 'bin_probs': B×H×W×N, 'bin_centers': B×N}`.
2. Two-stage training loop (R3):
   - **Stage 1 (warmup)**: train_head.py from U2. Load DA2-Small, freeze encoder, train only SEFHead + bin-center MLP. 20 epochs on KITTI+NYUv2. Best checkpoint → `checkpoints/sef_head_warmup.pt`.
   - **Stage 2 (end-to-end)**: train_end_to_end.py. Load warmup checkpoint, unfreeze all params (except DINOv2 patch embedding which stays frozen). Fine-tune with lower lr (1e-5), transparent mix in each batch. 10 epochs. Best checkpoint → `checkpoints/sef_end_to_end.pt`.
3. Memory management (R12):
   - Resolution: 384×384 during training. If OOM, fallback to 320×320.
   - fp16: enabled throughout via GradScaler.
   - Gradient accumulation: start at 4, measured empirically. Effective batch ≈ 16.
   - Check VRAM after stage 1 — if >5.5GB, reduce batch size before stage 2 (more activations from unfrozen encoder).
4. Inference script:
   - `python inference_sef.py --checkpoint <path> --input <image_or_dir> --output <dir>`.
   - Saves: depth map (float16 .npy + colorized .png), entropy map (float16 .npy + colorized .png), optional overlay composite.
   - Batch processing for eval datasets.
5. Pipeline orchestrator:
   - `run_pipeline.sh` stage 1, stage 2, eval on all splits, comparison table generation.
   - Each stage checks that previous outputs exist before running.
   - Total wall time estimate: ~6 days (stage 1: ~2d, stage 2: ~3d, eval: ~1d on single 4050).

**Patterns to follow:** DA2's inference script structure. HuggingFace `transformers` pipeline convention for `__call__` interface.

**Test scenarios:**
- Happy path: `SEFWrapper` forward on a single image returns `depth` and `entropy` tensors with correct spatial dimensions.
- Marginal expectation: if bin_probs is one-hot at bin k, depth output equals bin_centers[k]. Verifiable with synthetic data.
- Checkpoint round-trip: save and reload produces identical depth maps (max diff < 1e-5).
- MC Dropout baseline runs T=20 forward passes on a single image and returns uncertainty map.

**Verification:** `python run_pipeline.sh` completes all stages end-to-end on a 50-image toy dataset (subset of NYUv2). Final checkpoint `sef_end_to_end.pt` loads and produces valid depth + entropy outputs. Comparison table populates with real numbers.

---

## Key Technical Decisions
- **Classification over regression**: AdaBins' established ~10% RMSE improvement on classification-based depth, plus free uncertainty calibration, justifies the swap. Regression gives neither multi-modal outputs nor calibration without expensive sampling.
- **Learned bin centers (AdaBins-style) over uniform bins**: Adaptive discretization handles heavy-tailed natural depth distributions better; uniform bins waste capacity on empty far-field ranges.
- **N=96 default**: Matches AdaBins' empirically validated sweet spot. N=128 is a tuning knob for planning (memory vs resolution tradeoff).
- **Frozen encoder warmup, then end-to-end fine-tune**: 4M new params trained in isolation first avoids corrupting rich DINOv2 features; full fine-tune only after head converges.
- **MC Dropout only as uncertainty baseline**: Ensembles require 5× params and 5× training — implausible on 6GB. MC Dropout is pragmatic.
- **λ_c=0.1 for bin-center regression**: Matches AdaBins' relative weighting, prevents center drift during early training.

---

## Deferred Decisions (Tracked)
| Decision | When to resolve | Trigger |
|----------|----------------|---------|
| Bin-center MLP layers (2 vs 3) | During U1 implementation | Measure whether 2-layer MLP can represent full boundary range |
| Entropy regularization on/off | After stage 1 warmup | Validation ECE > 0.05 → enable λ_e=0.01 |
| Training hyperparams (lr, batch, accum) | During stage 1 first run | Based on VRAM usage at 384×384 |
| Invalid-GT handling (mask vs uniform) | During U2 loss implementation | Measure whether mask-only leaves calibration gaps |
| Transparent subset scene selection | During U3 curation | After inspecting Hypersim/Replica metadata |

## Computational Budget

| Resource | Budget |
|----------|--------|
| GPU memory | ≤6 GB |
| Single experiment | ≤24 hours |
| Full ablation suite | ≤7 days |

---

## Risk Management

| Aspect | Detail |
|--------|--------|
| Expected outcome | [what the head should achieve] |
| Possible failure | [what could go wrong] |
| Fallback experiment | [what to try if it fails] |

---

## Threats to Validity

**Internal:** Dataset bias, random initialization sensitivity, metric noise.
**External:** Limited to NYUv2/KITTI domains; may not generalize to medical/aerial/underwater.
**Construct:** ECE measures calibration but does not measure downstream utility of uncertainty.
**Statistical:** 3 seeds provide variance estimate but 5+ would be stronger.

---

## Risks & Dependencies
| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| VRAM OOM at 384×384 | Resolution must drop to 320×320, slight quality loss | Medium | Start at 384×384 with accum=4; fallback to 320×320 if >5.5GB. SEF head is only ~4M → activations are small |
| Transparent data curation takes >3 days | Schedule slip | Low | Use TORSD as immediate eval set (already transparent-labeled). Hypersim/Replica curation is backup |
| SEF does not match DA2 AbsRel | Core hypothesis invalid | Low | DA2 encoder features are rich; SEF head with N=96 has higher capacity than regression conv. If AbsRel degrades >1%: increase λ_c, verify bin-center regression is converging |
| Entropy always low/peaked | No useful uncertainty | Medium | Enable entropy regularization (R5). If still peaked: check ECE — peaked-but-calibrated is still useful for risk. |
| Pretrained DA2-Small unavailable | Cannot start | Low | DA2 weights downloadable from GitHub. If removed: use DA2-Mini (also fits 6GB) |

---

## Sources & References
- Origin: docs/brainstorms/2026-07-21-da2-sef-requirements.md
- AdaBins: Bhatt et al., "Adabins: Depth estimation using adaptive bins" (CVPR 2021), https://arxiv.org/abs/2103.08695
- Depth Anything V2: Yang et al., https://github.com/DepthAnything/Depth-Anything-V2
- TORSD: S. S. Sajjan et al., "Clear Grasp" (ICRA 2020) — transparent-object RGB-D dataset
- MC Dropout baseline: Y. Gal & Z. Ghahramani, "Dropout as a Bayesian Approximation" (ICML 2016)
- ECE reference: C. Guo et al., "On Calibration of Modern Neural Networks" (ICML 2017)
- AURC reference: Y. Geifman & R. El-Yaniv, "SelectiveNet" (CVPR 2019)
