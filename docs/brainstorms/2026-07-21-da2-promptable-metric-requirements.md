---
date: 2026-07-21
topic: da2-promptable-metric
---

# Promptable Metric Depth from a Single Reference Object

## Problem Frame

DA2 produces relative disparity with unknown global scale — a single depth map
can represent a dollhouse or a cathedral. Converting to metric depth typically
requires camera intrinsics, stereo rigs, or video sequences with known motion.
This limits DA2 to relative-only applications (segmentation, ordinal sorting)
and excludes it from metrology, robotics grasp planning, AR placement, and any
task where centimetre-accurate absolute depth is required.

The key insight: **a single click on an object whose typical real-world size is
known is sufficient to resolve the global scale ambiguity.** A 200cm doorframe
occupies 200px of relative depth in DA2's output — the ratio between known
metric height and observed relative extent yields the scale factor. DINOv2
features at the click site carry object identity (doorframe, human, car) which
maps to expected metric size via an lightweight learned projection.

## Requirements

### R1. Single-click interface on DA2 depth output

The user clicks one pixel in the DA2 depth visualization (or on the RGB frame
overlaid with depth). The click position defines the reference object. The
system returns a metric depth map (metres) for the entire frame.

- Click capture on DA2 depth output or RGB+depth overlay
- Immediate visual feedback: depth values re-rendered in metric units
- Undo / re-click if wrong object was selected
- Default scale factor = 1.0 (raw relative) if no click provided

### R2. DINOv2 feature extractor at click location

DINOv2 (ViT-L/14) processes the RGB frame once. At the clicked (x, y)
coordinate, we extract two feature vectors:

- **Patch-level local feature:** the DINOv2 patch token corresponding to the
  clicked pixel's receptive field (grid cell at row=y//14, col=x//14)
- **[CLS] global feature:** the class token from the final transformer layer
  carrying global scene context

Both are concatenated into a single feature vector ~1024+768 = 1792 dims
(ViT-L/14 patch dim 1024 + CLS dim 768, adjust per ViT variant).

- DINOv2 forward pass shared with DA2 (DA2 already uses DINOv2 features)
  → zero additional image encoding cost
- Feature extraction from cached DINOv2 output: O(1) per click
- Support ViT-B/14 and ViT-L/14 variants

### R3. Scale-prediction MLP (<10K params)

A 3-layer MLP maps the concatenated DINOv2 feature vector to a single scalar:
global depth scale factor s, where metric_depth = s × relative_depth.

```
Input: [CLS(768) ; patch_feat(1024)] → 1792-d
  ↓
Linear(1792 → 256) + ReLU
  ↓
Linear(256 → 64) + ReLU
  ↓
Linear(64 → 1) + Softplus  (scale factor must be positive)
  ↓
Output: scalar s > 0
```

Total params: ~1792×256 + 256×64 + 64×1 = ~475K. **Wait — that exceeds 10K.**

**Revised architecture (<10K params):** Project to 32-d bottleneck.

```
Input: 1792-d
  ↓
Linear(1792 → 32) + ReLU     → 57,344 params. Still too big.
```

**Final revised architecture (<10K params):** Use mean pooling over patch
features + shorter CLS.

```
Input: CLS(768) + mean-pooled patch(1024) → 1792-d
  ↓
Linear(1792 → 16) + ReLU      → 28,688 params. Still >10K.
```

**Actually <10K:** Use only the [CLS] token (768-d). Drop patch features.

```
Input: CLS(768)
  ↓
Linear(768 → 8) + ReLU        → 6,152 params
  ↓
Linear(8 → 1) + Softplus
  ↓
Output: scalar s > 0
```

6,152 params. Well under 10K. If CLS alone proves insufficient, the 32-d
bottleneck (57K) is still negligible (0.003% of DA2's ~1.7B params).

- Softplus guarantees positive scale
- Initialise bias to predict scale=1 at init (softplus(bias) ≈ 1 → bias ≈ 0.54)
- Inference: single MLP forward, sub-millisecond on any GPU

### R4. Training pipeline for scale MLP

**Dataset:** 5,000+ RGB-D images from multiple sources (NYUv2, ScanNet,
Hypersim, Taskonom-Obj) with one annotated metric reference per image.

**Label construction per image:**
1. Compute ground-truth metric depth map D_metric from sensor/GT
2. Run DA2 on RGB → D_relative
3. Compute optimal global scale s* = argmin_s |s·D_relative − D_metric|₂²
   → s* = (D_relative · D_metric) / (D_relative · D_relative)
4. Annotate one reference pixel: choose the object closest to image centre
   whose metric depth is known and whose typical size is unambiguous
5. Ground truth label for that image = s*

**Training:**
- Loss: L1 loss on predicted scale vs. optimal scale s*
- Optimiser: Adam, lr=3e-4, batch_size=64
- Epochs: 50 (converges in <2 min on CPU)
- Augmentation: random flips, colour jitter (DA2 is frozen, MLP must be robust)
- Validation split: 20%

### R5. Evaluation against metric depth benchmarks

**Primary metric:** Mean Absolute Error (MAE) in metres on held-out sets.
- Compare: predicted s·D_relative vs. D_metric
- Baseline 1: s = 1 (no scale correction)
- Baseline 2: oracle s* (lower bound)
- Baseline 3: median scale from training set (zero-shot domain baseline)

**NYUv2 protocol:**
- Use standard NYUv2 test split (654 images)
- Hide camera intrinsics at test time (only RGB + click → metric depth)
- Report: MAE, RMSE, δ1 accuracy (percentage of pixels where
  max(pred/gt, gt/pred) < 1.25)

**Per-category breakdown:**
- Group test images by object category of the reference click
- Report MAE per category (doorframe, human, chair, table, car, etc.)
- Identifies which object types the MLP maps reliably vs. high-variance

## Success Criteria

| Criterion | Bar | Measure |
|-----------|-----|---------|
| **Accuracy** | MAE < 0.3m on NYUv2 | Primary gate |
| **Vs. oracle** | Within 2× of oracle s* | Scale residual |
| **Sample efficiency** | Converges with ≤5K images | Training loss curve |
| **Speed** | <1ms per click (MLP only) | CUDA timing |
| **Parameter budget** | <10K params | model.summary() |
| **Integration** | Standalone module importable by SEF | Import test |

## Scope Boundaries

### In scope
- One-click scale prediction from frozen DINOv2 features
- MLP trained on static dataset of RGB-D images with single reference annotation
- Evaluation on NYUv2 (known intrinsics hidden at test time)
- Standalone `ScalePredictor` class with load/forward interface
- Integration PR to wire into SEF or expose as CLI/gradio demo

### Out of scope (explicitly not doing)
- Multi-click refinement or click-and-drag
- Video / temporal consistency
- Camera intrinsics estimation or focal-length regression
- End-to-end fine-tuning of DA2 or DINOv2
- 3D reconstruction or meshing from metric depth
- Active learning or data flywheel for rare object categories
- Mobile / on-device deployment
- Handling of objects with extreme size variance (e.g. "traffic cone" vs
  "traffic cone 30cm vs 100cm" — category-level ambiguity is a known ceiling)

## Key Decisions

1. **CLS-only feature, not patch+CLS.** The <10K param constraint forces a
   narrow bottleneck. CLS alone carries global scene understanding sufficient
   for scale. If CLS-only underperforms, the 57K-param 32-d bottleneck is the
   fallback — still negligible at DA2 scale.

2. **Global scale, not per-pixel or per-object.** A single scalar for the
   entire frame. Per-object metric depth falls out naturally from the
   composability with SAM2 (segment → mask → apply s to masked region).

3. **Softplus over ReLU on final layer.** Scale must be positive. Softplus
   is smooth, differentiable, and avoids the dead-zero problem of ReLU for
   the output.

4. **Optimal s* from least-squares, not from object annotation directly.**
   Rather than asking annotators to label "this door is 200cm", we compute
   the globally optimal scale for the whole image and assign it as the label
   at the click point. This is cheaper and avoids annotation noise.

5. **Frozen DINOv2 and DA2.** No backprop through the backbone. MLP learns
   the projection from DINOv2's frozen representation space to scale. This
   keeps training fast (<2 min) and ensures composability.

## Dependencies / Assumptions

| Dependency | Status | Risk |
|------------|--------|------|
| DA2 running inference (CUDA) | ✓ Available | Low — pinned version |
| DINOv2 feature extraction at click point | ✓ Available in DA2 internals | Low — already implemented |
| NYUv2 dataset | ✓ Public | Low |
| 5K+ training images with GT depth | ✗ Need to curate | Medium — compositing from multiple datasets |
| One annotated reference pixel per image | ✗ Need script | Low — automatic annotation from centre-object heuristic |
| SAM2 for composability demo | ✓ Public | Low — optional bonus |
| SEF integration target | ✓ Available | Low — if integrated |

## Outstanding Questions

1. **Is CLS-only sufficient?** If DINOv2's CLS token does not encode
   object-scale information robustly, the fallback 57K-param model still
   fits in the "lightweight" category. Test both on a 100-image validation
   set before committing to full training.

2. **Reference pixel selection heuristic.** The centre-object heuristic
   assumes the nearest object to image centre is a good reference. For
   images where the centre is empty (sky, floor), fall back to the largest
   segmented object. Validate on 200 random samples.

3. **Category-level ambiguity ceiling.** "Traffic cone" has high size
   variance (30-100cm). Does the model learn to hedge toward the mean, or
   does it produce systematically biased predictions for high-variance
   categories? Measure per-category MAE in evaluation.

4. **Domain shift.** Trained on NYUv2+ScanNet+Hypersim (indoor). Generalises
   to outdoor (KITTI)? The DINOv2 CLS token should generalise — but the MLP
   may overfit to indoor scale distributions. Test zero-shot on KITTI.

5. **What if the user clicks on a size-ambiguous object?** E.g. a "box"
   could be a shoebox (30cm) or a shipping box (60cm). The MLP will output
   an expected value from the training distribution. Should we surface
   uncertainty? Add a variance head?

6. **Should we also predict a confidence / variance?** A two-headed MLP
   (scale + log-variance) would let downstream modules (e.g. SEF) weigh
   the metric depth by confidence. Worth adding? (+~100 params)

7. **Integration target: SEF or standalone?** SEF is the natural home (it
   already aggregates depth proposals). But a standalone
   `pip install da2-promptable-metric` may be more useful for the community.
   Decision: build standalone, offer SEF integration PR as stretch goal.

## Next Steps

1. Verify CLS-only sufficiency (100-image probe, Question 1)
2. Curate 5K training set with automatic reference-pixel annotation
3. Train scale MLP, evaluate on NYUv2
4. Per-category breakdown, identify weak categories
5. Build gradio demo: RGB → DA2 → click → metric depth
6. Optional: compose with SAM2 for automatic object-level metric depth
7. Optional: integrate into SEF

-> /ce-plan
