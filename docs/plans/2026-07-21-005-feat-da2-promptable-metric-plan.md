# Plan: Promptable Metric Depth from a Single Reference Object

**Date:** 2026-07-21
**Source:** `docs/brainstorms/2026-07-21-da2-promptable-metric-requirements.md`

## Overview

DA2 produces relative disparity with unknown global scale. A single click on an object whose typical size is known resolves the ambiguity: DINOv2 features at the click site drive a tiny MLP (<10K params) that predicts a global scale factor `s`. Metric depth = `s × relative_depth`. Training requires ~5K RGB-D images with automatically annotated reference pixels; inference is sub-millisecond.

## Implementation Units

### Unit 1: ScalePredictor module — architecture + inference interface

**Files:** `da2_promptable_metric/scale_predictor.py`, `da2_promptable_metric/__init__.py`

- Implement `ScalePredictor` class with 3-layer MLP: `Linear(768 → 8) + ReLU → Linear(8 → 1) + Softplus`
  - Input: DINOv2 [CLS] token only (768-d). Fallback path: Concatenated CLS+mean-pooled-patch (1792-d → 32-d → 1) if CLS-only underperforms.
  - Softplus bias initialised so `s ≈ 1` at init (`bias ≈ 0.54`)
- `forward(cls_token: Tensor) → Tensor` returns scalar scale factor `s > 0`
- `load(ckpt_path)` / `save(ckpt_path)` for serialisation
- `num_params()` property; validate `<10K` in `__init__` assertion
- Unit test: forward pass shape, init scale ≈ 1, param count <10K

**Acceptance:** Module importable, forward pass produces `s > 0`, `<10K` params.

---

### Unit 2: DINOv2 feature extraction at click point

**Files:** `da2_promptable_metric/feature_extractor.py`

- Wraps DA2's cached DINOv2 output. DA2 already runs DINOv2 once per frame — Unit 2 extracts the [CLS] token from the final transformer layer at zero additional encoding cost.
- `get_click_features(dinov2_output, x, y, model_variant="vitl14") → cls_token: Tensor`
  - Handles coordinate-to-patch mapping: `row = y // patch_size`, `col = x // patch_size`
  - Extracts patch-level feature for the clicked cell + [CLS] global token
  - Supports ViT-B/14 (patch=14) and ViT-L/14 variants
- Standalone test: mock DINOv2 output, verify token shapes and coordinate mapping

**Acceptance:** Feature extraction is O(1) per click. Correct token shapes for both ViT variants.

---

### Unit 3: Training pipeline — dataset + training loop

**Files:** `da2_promptable_metric/train.py`, `da2_promptable_metric/dataset.py`, `scripts/prepare_training_data.py`

- **Dataset curation script** (`scripts/prepare_training_data.py`):
  1. Downloads/copies NYUv2 + ScanNet + Hypersim + Taskonom-Obj splits
  2. For each image, runs DA2 → `D_relative`, computes optimal `s*` via least-squares: `s* = (D_relative · D_metric) / (D_relative · D_relative)`
  3. Annotates one reference pixel per image: nearest object to image centre (fall back to largest segmented object if centre is empty)
  4. Outputs a manifest CSV: `rgb_path, depth_gt_path, click_x, click_y, s_star, category`
- **Dataset class** (`dataset.py`):
  - Loads RGB, runs DA2+DINOv2 (or loads precomputed features), returns `(cls_token, s_star)` pairs
  - Augmentation: random flips, colour jitter
  - Validation split: 20%
- **Training loop** (`train.py`):
  - Loss: L1 loss on predicted scale vs. `s*`
  - Optimiser: Adam, `lr=3e-4`, `batch_size=64`, 50 epochs
  - Logs loss curve; saves checkpoint at best validation loss
  - Runs in <2 min on CPU; optionally CUDA

**Acceptance:** Training converges <50 epochs. Checkpoint loads into `ScalePredictor`.

---

### Unit 4: Gradio demo + full pipeline integration

**Files:** `demo/app.py`, `da2_promptable_metric/pipeline.py`

- **Pipeline class** (`pipeline.py`):
  - `Pipeline.compute_metric_depth(rgb: np.ndarray, click_x: int, click_y: int) → metric_depth: np.ndarray`
    1. Run DA2 → `D_relative` + cache DINOv2 output
    2. Extract `cls_token` at click point (Unit 2)
    3. `s = scale_predictor(cls_token)` (Unit 1)
    4. Return `s × D_relative`
  - If no click provided: default `s = 1.0` (raw relative depth)
- **Gradio app** (`demo/app.py`):
  - RGB input → DA2 depth overlay rendered with colourmap
  - User clicks on depth overlay → re-renders in metric units (metres) with colourbar
  - Undo / re-click: click on a different point to update
  - Displays predicted scale factor `s` numerically
  - Optional: compose with SAM2 to show per-object metric depth

**Acceptance:** Full end-to-end: RGB → click → metric depth map in one UI.

---

### Unit 5: Evaluation — NYUv2 benchmark + per-category analysis

**Files:** `scripts/evaluate_nyu.py`, `scripts/per_category_breakdown.py`

- **NYUv2 evaluation** (`scripts/evaluate_nyu.py`):
  - Standard 654-image test split; intrinsics hidden
  - Report: MAE (m), RMSE (m), δ1 accuracy (%)
  - Baselines: `s=1`, oracle `s*`, median-scale from training set
- **Per-category breakdown** (`scripts/per_category_breakdown.py`):
  - Groups test images by object category of the reference click
  - Reports MAE per category; flags categories with high variance (the "traffic cone" ceiling)
- **Zero-shot domain check:** Run on KITTI (outdoor) without retraining, report MAE
- **Ablation:** Compare CLS-only vs. CLS+patch (57K) on the 100-image probe from the requirements document (`docs/brainstorms/2026-07-21-da2-promptable-metric-requirements.md`)

**Acceptance:** MAE < 0.3m gate on NYUv2. Per-category report identifies weak categories.

---

## Dependencies

| Dependency | Unit | Notes |
|-----------|------|-------|
| DA2 (CUDA) | 2, 4 | Pinned version; inference only, no backprop |
| DINOv2 checkpoint | 2 | Shared with DA2 — no extra download |
| torch | 1, 3 | |
| gradio | 4 | pip install |
| NYUv2 / ScanNet / Hypersim | 3, 5 | Public datasets |
| SAM2 | 4 | Optional, for composability demo |

## Key Decisions (from requirements)

1. CLS-only features (768-d → 8 → 1), <10K params. Fallback: 57K 32-d bottleneck if needed.
2. Global scalar scale, not per-pixel. Per-object metric depth via SAM2 composition.
3. Softplus on output for guaranteed positivity.
4. Optimal `s*` from least-squares (whole-image), not per-object annotation.
5. Frozen DINOv2 and DA2 throughout — no backprop.

---

## Hypothesis

| ID | Statement | Null form | Test |
|----|-----------|-----------|------|
| H1 | A 3-layer MLP (<10K params) on a single DINOv2 [CLS] token can predict a global scale factor s with MAE < 0.3m on NYUv2 | H1₀: MAE ≥ 0.3m | `scripts/evaluate_nyu.py` on 654-image test split |
| H2 | Scale factor generalises from a single in-distribution click without per-scene calibration | H2₀: oracle s* outperforms predicted s by >2× | Compare MAE vs. least-squares s* baseline |
| H3 | Category-level failure modes are predictable (high-variance categories have ≥2× MAE of low-variance) | H3₀: MAE variance is uniform across categories | `scripts/per_category_breakdown.py` χ² test on variance |

---

## Risk Management

| Risk | Likelihood | Impact | Mitigation | Trigger | Owner |
|------|-----------|--------|------------|---------|-------|
| CLS-only MLP cannot represent scale from a single visual token | Low | Core failure | Deferred fallback to CLS+mean-patch (1792-d → 32-d → 1, 57K params). Add flag in `ScalePredictor.__init__` | val MAE > 0.3m at epoch 20 | Implementer |
| Reference-pixel annotation (nearest-object-to-centre) yields degenerate s* for textureless or occluded regions | Medium | Biased training signal | Automated quality filter: reject samples where s* deviates >3σ from category median during curation | `scripts/prepare_training_data.py` outputs >10% rejected samples | Implementer |
| Dataset imbalance — NYUv2 dominates training; generalisation to Hypersim/ScanNet fails | Medium | Overfit to indoor NYUv2 distribution | Stratified sampling by dataset in each batch. Zero-shot KITTI eval catches outdoor failure early | KITTI MAE > 0.5m | Reviewer |
| Gradio demo latency is unacceptable (>1s per click) | Low | Poor UX | Profiling gate in CI: `pipeline.py` end-to-end must run <100ms on CPU. If exceeded, add feature caching | CI profile step fails | Implementer |
| Softplus bias initialisation drifts from s≈1 after first SGD step | Low | Slower convergence | Verify s distribution at epoch 1; if mean deviates >20%, add explicit s-normalisation in loss | Training log at epoch 1 | Implementer |

---

## Threats to Validity

### Internal Validity
- **Confounding by object size:** Reference object typical size may correlate with dataset (NYUv2 beds → metric scale leaks via dataset prior). Mitigation: report per-dataset MAE separately.
- **s* oracle is not ground-truth:** Whole-image least-squares s* mixes foreground/background scale. A click on a small foreground object may have a different optimal s than the frame-level least-squares target. Mitigation: add pixel-masked s* (only pixels from the same SAM2 segment as the click) as an additional baseline in evaluation.

### External Validity
- **Single-reference constraint:** Real users may click on non-representative objects (wall, floor). The evaluation protocol always clicks on a known object — this overestimates real-world performance. Mitigation: add a "random click" variant in evaluation (click on any pixel, not just objects).
- **Known-object prior:** Three of four training datasets (NYUv2, ScanNet, Hypersim) are indoor scenes. Zero-shot KITTI outdoor eval bounds outdoor validity but does not guarantee generalisation to aerial, underwater, or medical domains.
- **Category coverage:** Training categories are limited to those present in NYUv2/ScanNet/Hypersim object annotations. Long-tail categories (traffic cones, unusual furniture) are underrepresented — the per-category breakdown explicitly flags these.

### Construct Validity
- **MAE as sole gate:** MAE < 0.3m measures average error but hides systematic bias (all predictions offset by a constant). Add signed bias (mean error) to evaluation output to detect systematic drift.
- **Reference-click location sensitivity:** A click at the centre of an object vs. its edge produces different [CLS] features due to attention pooling. Add a stability test: 5 clicks per image within the same object mask; report s variance.

---

## Cost Metrics

| Metric | Budget | Measurement | Hard ceiling |
|--------|--------|-------------|-------------|
| Training params | <10K | `ScalePredictor.num_params()` | 15K |
| Training time (CPU) | <2 min | Timer around training loop in `train.py` | 5 min |
| Training time (CUDA) | <10 s | Timer around training loop in `train.py` | 30 s |
| Inference (single click) | <1 ms | `timeit` over 1000 forward passes | 5 ms |
| Data download | <15 GB | `du -sh data/` after curation script | 25 GB |
| Disk (features + checkpoints) | <2 GB | `du -sh outputs/` | 5 GB |
| Demo latency (click→render) | <500 ms | Gradio timer widget | 1 s |
| Precomputed feature storage | <1 GB | Per-dataset precomputed DINOv2 tokens | 3 GB |

---

## 3× Seeds

Every training experiment is repeated with 3 random seeds to establish statistical significance:

| Seed | Purpose | Config override |
|------|---------|----------------|
| 42 | Default / primary | None (baseline) |
| 1337 | Reproducibility cross-check | `--seed 1337` |
| 20260721 | Date-anchored seed (no cherry-picking) | `--seed 20260721` |

**Reporting:** Every result table reports `mean ± std` across 3 seeds (e.g., `MAE: 0.27 ± 0.02 m`). If any seed fails to converge (MAE > 2× median), flag and document whether it's a saddle-point or data-order issue.

---

## Ablation Chains

Ordered ablation experiments — each chain adds one modification to isolate its contribution:

| Chain | Step 0 (baseline) | Step 1 | Step 2 | Step 3 |
|-------|-------------------|--------|--------|--------|
| **Input features** | CLS-only (768-d → 8 → 1) | CLS+mean-patch (1792-d → 32 → 1) | CLS+patch-grid (768+768 → 32 → 1) | — |
| **Training data** | NYUv2 only | + ScanNet | + Hypersim | + Taskonom-Obj |
| **Reference selection** | Nearest-object-to-centre | Largest-object-in-frame | Random-pixel (uniform) | Centred-on-SAM2-mask |
| **Scale target** | Whole-image least-squares s* | SAM2-masked s* | Per-object median s* | — |
| **Augmentation** | None | Random flips | Colour jitter | Flip + jitter |
| **Inference** | Single click | 3-click majority vote | 5-click median | — |

**Reporting:** Each chain produces a comparison table. Any step that degrades the metric is recorded as a negative result (equally valuable). Results go into `eval/ablations/` as `.json`.

---

## Computational Budget

| Resource | Budget | Notes |
|----------|--------|-------|
| GPU memory | ≤6 GB | RTX 4050 hard limit |
| Single training run | <2 min (CPU) / <10 s (CUDA) | Verified in Unit 3 acceptance |
| Full ablation suite | ≤1 hour | 6 chains × 3 steps × 3 seeds × 10 s ≈ 9 min total; overhead dominated by feature extraction |
| Feature precomputation | <30 min per dataset | One-time per dataset; cached to disk |
| Data storage | <20 GB | 4 datasets + precomputed features + checkpoints + evaluation outputs |
| Total wall time (all experiments) | ≤3 hours | Including precomputation, training, eval, ablation sweep |
