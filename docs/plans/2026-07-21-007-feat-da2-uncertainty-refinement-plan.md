# Plan: Closed-Loop Uncertainty-Guided Depth Refinement

**Date:** 2026-07-21
**Tracking:** Issue #007
**Source:** `docs/brainstorms/2026-07-21-da2-uncertainty-refinement-requirements.md`
**Design doc:** `docs/designs/2026-07-21-da2-uncertainty-refinement-design.md`

---

## Problem Frame

Monocular depth estimates are unreliable on textureless regions, reflective surfaces, and object boundaries. Existing DA2 outputs a single depth map with no per-pixel confidence signal — downstream consumers cannot distinguish trustworthy from hallucinated predictions. Closing the loop: predict per-pixel uncertainty, then use confident regions to supervise uncertain ones via a differentiable refinement pass, trained with photometric consistency (no GT depth).

---

## Hypothesis

H1: Per-pixel aleatoric log-variance predicted from DA2 decoder features captures the spatial distribution of depth error — high-variance regions correspond to high-AbsRel regions (correlation ρ ≥ 0.6 against per-pixel error on held-out KITTI frames).

H2: Differentiable Laplace-weighted interpolation using a soft sigmoid-gated mask reduces AbsRel by ≥5% on KITTI Eigen split without GT depth, by propagating information from low-uncertainty to high-uncertainty pixels.

H3: Joint end-to-end training (uncertainty head + decoder) outperforms post-hoc training alone (ΔAbsRel ≥ 2% relative), validating that refinement gradients improve decoder feature quality.

Null hypothesis H₀: Uncertainty-guided refinement produces no statistically significant improvement over uniform-weight interpolation (McNemar's test on δ1.25 at α=0.05).

---

## Requirements Trace

- **R1.** 3-layer MLP uncertainty head on DA2 decoder features, <1M params, outputs per-pixel log-variance
- **R2.** Differentiable refinement pass: uncertainty mask → Laplace-weighted interpolation from low-uncertainty neighbors
- **R3.** Self-supervised training loop: photometric consistency (L1+SSIM) after refinement, gradients through refinement → uncertainty head
- **R4.** KITTI evaluation: AbsRel, δ1.25, ECE, AURC, FPS — before vs. after refinement

---

## Scope Boundaries

- **In scope:** Lightweight aleatoric uncertainty head, Laplace interpolation refinement (no learned refiner), photometric self-supervision, KITTI-only eval, post-hoc then joint training
- **Out of scope:** Training DA2 from scratch, GT-depth supervision for refinement, multiple datasets, modifications to SEF itself, epistemic uncertainty (MC dropout, ensembles), learned refinement network

---

## Risk Management

| Risk | Likelihood | Impact | Mitigation | Trigger |
|------|-----------|--------|-----------|---------|
| Photometric loss is underdetermined (textureless regions have many equally photometric depths) | Medium | High — refinement converges to degenerate solution (uniform depth) | Edge-aware photometric weighting (gradient magnitude as reliability mask); if loss plateaus >2% AbsRel from target, add sparse GT-depth regularization (5% of pixels) | Validation AbsRel not decreasing after 5k steps |
| Uncertainty head collapses to constant map (all pixels equal variance) | Low | High — refinement becomes uniform blur | Gradient clipping (norm=1.0), spectral normalization on first Linear layer; detect via variance of predicted log-variance < 0.01 over batch | Detection check in training loop |
| Refinement pass washes out fine structure (high-frequency detail destroyed by interpolation) | Medium | Medium | Add detail-preservation term: L1(refined_depth, raw_depth) weighted by low-uncertainty mask; enable only if edge-PSNR drops >1dB | Per-frame edge-PSNR monitoring |
| Local 7×7 window has zero low-uncertainty pixels too often (>5%) | Medium | Medium — fallback to global attention is expensive | If fallback rate >5% after 1k steps, increase window to 15×15 or switch to global sparse attention | Fallback rate counter |
| Training instability from sigmoid-gate near-binary gradient (β too high) | Low | Medium — NaN loss, training divergence | β warmup: linear from β=1 to β=10 over first 2k steps. Detect NaN → halve β and restart step | Loss is NaN / Inf |
| DA2 intermediate features not accessible (API changed) | Low | High — blocks IU1 entirely | Pin DA2 commit hash; forward hook as fallback if direct API call unavailable | Integration test fails |

---

## Key Technical Decisions

1. **Post-hoc → joint training:** Start with frozen DA2 encoder/decoder (train uncertainty head only). Switch to end-to-end if refinement signal is weak. (see origin)
2. **Soft threshold via sigmoid-temperature gating:** `σ(t) = sigmoid(β · (u − u_τ))`. High β ≈ hard mask (cleaner but near-binary gradient); low β smoother but leaks. Start β=10, tune.
3. **Local 7×7 neighbor window** for KNN to keep attention weighting O(HWk²). Global sparse attention fallback when local window lacks low-uncertainty pixels.
4. **K=5** (default), validate K=3 and K=10. Quantile τ=0.8 (top 20% uncertain).
5. **Reflection padding** on feature maps for edge-of-frame pixel handling. Gaussian feather on mask boundary to avoid hard seams.

---

## Error Taxonomy

DA2 depth errors fall into distinct categories; the refinement pass should handle each differently. Standard taxonomy adopted from monocular depth literature and validated against DA2's known failure modes:

| Category | Cause | Expected uncertainty signal | Refinement effectiveness | Mitigation if weak |
|----------|-------|---------------------------|------------------------|-------------------|
| **Type A — Textureless regions** (walls, sky, road) | No local gradient for stereo correspondence | High variance, spatially uniform | High — neighbors have correct depth | Increase window size to capture distant edges |
| **Type B — Reflective/transparent surfaces** (windows, mirrors, water) | Specular highlights break photometric consistency | High variance, potentially fragmented | Medium — if reflection occupies >50% of window, no correct neighbors | Add reflection priors (plane-parallel assumption) |
| **Type C — Object boundaries** (edges of cars, poles) | Mixed-depth pixels from foreground/background | High variance, spatially correlated with edge | Medium — interpolation blurs sharp edges | Edge-aware refinement: reduce weight across edges (RGB gradient gating) |
| **Type D — Foliage/fine structure** (trees, fences) | High-frequency detail undersampled at input resolution | High variance, oscillating | Low — interpolation smears structure beyond recognition | Fall through to raw depth (no refinement) when local patch has high RGB variance |
| **Type E — Large low-texture regions with distant horizon** (sky meeting road) | No cue at all — depth is unbounded | High variance, soft transition | High — distant plane prior works well | Parametric plane fit as refinement alternative |
| **Type F — Motion blur / rolling shutter** | Temporal misalignment in training pairs | Moderate variance | Low — refinement can hallucinate; photometric loss unreliable | Skip refinement; flag for temporal consistency module (future work) |

Each refined pixel is logged with its error type (via heuristic classifier: RGB variance → Type D, edge proximity → Type C, etc.) so the ablation sweep can slice metrics by category and identify which types the method actually fixes vs. makes worse.

---

## Implementation Units

### IU1: Uncertainty MLP Head + DA2 Integration

Core module. A 3-layer MLP attached to the DA2 decoder's penultimate feature map that predicts per-pixel log-variance (aleatoric uncertainty).

**Files:**
- `src/da2_uncertainty/head.py` — `UncertaintyHead(hidden_dims=[64,32])`, forward takes DA2 feature map (B×C×H×W) → log-variance map (B×1×H×W). ReLU activations, no batchnorm.
- `src/da2_uncertainty/__init__.py` — exports `UncertaintyHead`, `build_uncertainty_head(cfg)` factory.
- `src/da2_uncertainty/config.py` — dataclass `UncertaintyConfig` with `hidden_dim`, `output_dim=1`, `beta`, `tau`, `K`, `window_size`.
- Integrate into DA2: forward hook or explicit call after decoder, feature map flows into head.
- `tests/test_uncertainty_head.py` — shape assertions, param count (<1M), forward/backward on synthetic features.

**Depends on:** DA2 checkpoint (intermediate features accessible), PyTorch ≥1.13

**Technical design:**
```
Input: DA2 decoder penultimate feature map (B, C, H, W)
  → Linear(C, 64) + ReLU
  → Linear(64, 32) + ReLU
  → Linear(32, 1)
Output: log-variance map (B, 1, H, W)
```

**Test scenarios:**
- Happy path: forward on random feature map of expected shape → output shape (B,1,H,W)
- Param ceiling: model parameters < 1,000,000
- Gradient flow: backward from log-variance → all head params receive non-zero gradient
- Frozen backbone: with DA2 encoder detached, only head params update

**Verification:**
- `pytest tests/test_uncertainty_head.py` passes
- Param count logged and <1M confirmed

---

### IU2: Differentiable Laplace Refinement Pass

Refinement operation. Takes depth map + uncertainty map, produces refined depth via soft-masked Laplace interpolation. Fully differentiable for end-to-end training.

**Files:**
- `src/da2_uncertainty/refinement.py` — `LaplaceRefinement(K=5, tau=0.8, beta=10.0, window_size=7)`.
  - `soft_mask(uncertainty)` — sigmoid-gated threshold
  - `laplace_weights(depth, mask, k)` — spatial distance + inverse-uncertainty weighting over local window
  - `refine(depth, uncertainty, feature_map)` — mask → weights → weighted sum → refined depth
- Reflection padding on feature maps. Gaussian blur on mask edges (kernel=5, σ=2) to feather hard seams.
- Fallback: if local window has zero low-uncertainty pixels, widen window to global (sparse gather).
- `tests/test_refinement.py` — known depth with synthetic uncertainty mask; verify high-uncertainty pixels shift toward low-uncertainty mean.

**Depends on:** IU1 (uncertainty head produces the input uncertainty map)

**Approach:**
- Soft mask replaces hard threshold: pixels pass through partially based on how far above/below τ
- Attention-like weighting over 7×7 local neighborhood: weight = `1/(spatial_dist + ε) * exp(-uncertainty)` — Laplace kernel in both spatial and uncertainty domains
- Implementation as pure PyTorch ops (unfold + einsum) — no custom CUDA kernels needed

**Test scenarios:**
- Happy path: uniform depth + high-uncertainty center pixel → refined value ≈ neighbor mean
- Soft vs hard mask: sigmoid-gated output converges to hard mask as β → ∞ (high-β validation)
- Edge-of-frame: pixels within 3px of border produce valid output (reflection padding)
- Differentiability: backward from refined depth → uncertainty map receives gradients
- Feathering: mask edge shows smooth transition (no hard seam artifact in output)

**Verification:**
- All test scenarios pass
- Known synthetic case: center 10×10 block with u=1.0, surround u=0.0 → refined block matches surround depth within 1e-2

---

### IU3: Self-Supervised Training Loop + Photometric Loss

Training harness. Wraps DA2 + uncertainty head + refinement into a self-supervised loop using photometric consistency from adjacent frames. No GT depth.

**Files:**
- `src/da2_uncertainty/trainer.py` — `SelfSupervisedTrainer(model, uncertainty_head, refinement, dataloader, config)`.
  - Photometric loss: `L1(raw_depth_pred, warped_target) + SSIM(raw_depth_pred, warped_target)` computed **after refinement**.
  - Two-phase schedule: Phase 1 (frozen backbone, train head only, 10k steps), Phase 2 (joint finetune, 5k steps, lower lr).
  - Gradient paths: refined depth → refinement weights → uncertainty head params; refined depth → decoder params (Phase 2).
  - **3 seeds:** Every training run repeated with seeds {42, 1337, 2026}. Metrics reported as mean ± std across seeds. A run must pass all acceptance criteria at mean−1σ (pessimistic bound) to be considered successful.
- `src/da2_uncertainty/losses.py` — `photometric_loss(depth, warped_ref, alpha=0.85)` returns `alpha * L1 + (1-alpha) * SSIM`.
- `configs/train_uncertainty.yaml` — ADAM (lr=1e-4 head, 5e-5 joint), batch=6, res=640×192, KITTI raw.
- `tests/test_trainer.py` — synthetic 2-frame sequence: verify loss decreases over 50 steps, verify gradient flow stops when each phase's requires_grad is correctly set.

**Depends on:** IU1, IU2, KITTI raw dataset accessible, DA2 image pair warping utilities (existing)

**Approach:**
- Photometric loss computed on refined depth creates a closed loop: if refinement improves depth accuracy, loss goes down → gradient updates the uncertainty head to produce better masks → which produces better refinement
- Phase 1 avoids catastrophic forgetting of backbone; Phase 2 allows refinement to shape decoder features
- Early stopping on validation AbsRel if computed periodically (every 1k steps, needs GT — noted as optional)

**Test scenarios:**
- Loss topology: on synthetic data with known perfect depth, photometric loss is approximately zero
- Gradient isolation (Phase 1): head params update, decoder params unchanged
- Gradient flow (Phase 2): all model params update
- Convergence: on synthetic 2-frame sequence, loss ‖ at minimum over 50 steps (non-increasing final 10 steps)

**Verification:**
- Training loop runs 1k steps without OOM (peak GPU < 5.5GB at batch=6)
- Loss traces logged to tensorboard / CSV
- Phase transition: optimizer param groups correctly toggle requires_grad

---

### IU4: KITTI Evaluation Harness + Metrics

End-to-end benchmark. Computes depth accuracy, uncertainty calibration, and timing — before vs. after refinement. All metrics from R4.

**Files:**
- `scripts/eval_kitti_depth.py` — runs DA2 + uncertainty head + refinement on KITTI Eigen split. Reports AbsRel, δ1.25, RMSE before/after refinement.
- `scripts/eval_calibration.py` — ECE (20 bins), AURC, sorted risk-coverage curve. Plots to `results/calibration_curve.png`.
- `scripts/eval_ablations.py` — three runs: (a) no refinement (baseline), (b) refinement with uniform mask (same weight everywhere), (c) refinement with oracle GT-error mask. Outputs comparison table.
- `scripts/benchmark_fps.py` — times each stage: DA2 inference, uncertainty head, refinement pass, total. 100-frame warmup, 500-frame measurement. Writes `results/latency.json`.
- `tests/test_eval.py` — golden values on 10-frame KITTI subset (assert AbsRel, ECE within tolerance).

**Depends on:** IU1, IU2, IU3 (trained checkpoint), KITTI Eigen split, DA2 baseline metrics known

**Test scenarios:**
- Consistency: eval pipeline produces same metrics on same 10-frame subset within 1e-4 tolerance
- Latency budget: refinement (head + pass) < 15% of total DA2 inference time
- Ablation ordering: oracle mask > learned mask > uniform mask > no refinement (AbsRel)

**Verification:**
- `python scripts/eval_kitti_depth.py --checkpoint <path>` produces comparable table
- `python scripts/eval_ablations.py` produces ablation table with expected ordering
- `python scripts/benchmark_fps.py` reports refinement overhead fraction

---

### IU5: Uncertainty Calibration Analysis + Ablation Sweep

Calibration diagnostics and sensitivity analysis on hyperparameters. Produces the ECE ≤ 0.05 commitment and informs τ, K, β tuning.

**Ablation chain (running in order, each entry adds one component):**
1. **A0 — No refinement (baseline):** Raw DA2 depth. No uncertainty head, no refinement.
2. **A1 — Uniform refinement:** Refinement pass with uniform mask (same weight everywhere). Measures improvement from pure interpolation without uncertainty guidance.
3. **A2 — Learned uncertainty refinement:** Full pipeline (uncertainty head + refinement). The proposed method.
4. **A3 — Oracle refinement:** Refinement pass with GT-error mask (computed from known GT depth). Upper-bound ceiling for the refinement design.
5. **A4 — Joint end-to-end:** Full pipeline + decoder fine-tuned (Phase 2). Measures value of end-to-end training.

Each ablation run with 3 seeds. Delta reported as A2 - A1 (value of learned uncertainty) and A3 - A2 (headroom to oracle). If A2 − A1 is not significant (Welch's t-test, p<0.05), the uncertainty head is not adding value over uniform interpolation.

**Files:**
- `scripts/calibration_analysis.py` — reliability diagrams (confidence vs accuracy per bin), sharpness histogram, sorted AURC curve + AUC. Outputs `results/reliability_diagram.png`, `results/sharpness_hist.png`.
- `scripts/sweep_hyperparams.py` — grid over τ ∈ {0.7, 0.8, 0.9}, K ∈ {3, 5, 10}, β ∈ {5, 10, 20}. Writes `results/sweep_results.csv`.
- `results/sweep_results.csv` — committed to repo, generated by CI or manual run.

**Depends on:** IU4 (eval scripts), trained checkpoint from IU3

**Test scenarios:**
- Calibration: ECE < 0.05 on KITTI Eigen split target
- Sweep determinism: re-running sweep with same seed produces same top-τ row
- Monotonicity: oracle-mask refinement strictly dominates learned-mask refinement (AbsRel)

**Verification:**
- `python scripts/calibration_analysis.py --checkpoint <path>` produces ECE ≤ 0.05
- `python scripts/sweep_hyperparams.py` finds top configuration; optimal τ, K, β documented in results

---

## Dependency Order

```
IU1 (uncertainty head)
 └── IU2 (refinement pass) ── requires IU1 output shape
      └── IU3 (training loop) ── requires IU1 + IU2
           └── IU4 (eval harness) ── requires IU3 checkpoint
                └── IU5 (calibration + sweep) ── requires IU4 scripts
```

Implement in IU1→IU2→IU3→IU4→IU5 order. IU4 and IU5 can be parallelized once the IU3 checkpoint exists.

---

## Acceptance

| Criterion | Target |
|-----------|--------|
| Depth AbsRel improvement after refinement | ≥5% relative reduction in AbsRel (new_AbsRel ≤ 0.95 × baseline_AbsRel) on KITTI Eigen split |
| Uncertainty calibration (ECE) | ≤0.05 after training |
| Refinement overhead | <15% additional inference time (trained head only) |
| MLP params | <1M |
| GPU memory (6GB target) | Peak ≤5.5GB during training |
| Self-supervised (no GT depth) | Photometric loss converges, depth quality reaches supervised baseline within 10% relative (α ≥ 0.90 × supervised AbsRel) |

---

## Cost Metrics

| Metric | Target | How measured | When |
|--------|--------|-------------|------|
| **Training FLOPs** | ≤3.2 PFLOPS total (Phase 1 + Phase 2) | `thop.profile` on each forward/backward step × step count | After IU3 completes |
| **Training wall time** | ≤24 hrs on single A4000 (Phase 1: ~8h, Phase 2: ~16h) | `time` wrapper in trainer script | After IU3 |
| **Refinement GPUµs per 640×192 frame** | ≤300 µs (head + pass combined) | CUDA event timing, 500-frame median | After IU4 benchmark |
| **CO₂e per training run** | ≤0.8 kg CO₂e (using `codecarbon` or wattmeter × regional factor) | `codecarbon` tracker in training loop | After IU3 |
| **Storage per checkpoint** | ≤150 MB (model weights + optimizer state) | `os.path.getsize` | After IU3 |

---

## Computational Budget

Total budget for the project: **≤36 A4000-GPU-hours** (or equivalent). Allocation across experiments:

| Activity | GPU-hours | Notes |
|----------|-----------|-------|
| Phase 1 development & debugging | 4h | Synthetic data, small batch, rapid iteration |
| Phase 1 production run (3 seeds) | 3 × 8h = 24h | Frozen backbone, 10k steps |
| Phase 2 production run (3 seeds) | 3 × 2h = 6h | Joint finetune, 5k steps |
| Ablation chain A0–A4 (3 seeds each) | 5 × 0.5h × 3 = 7.5h | Only eval, no training needed for A0–A3 reruns; budget includes one full retrain for A4 per seed |
| Hyperparameter sweep (τ, K, β) | 3 × 3 × 3 = 27 configs × 0.5h = 13.5h | Parallelized over GPUs if available |
| Calibration analysis + figure generation | 1h | Eval-only |
| Contingency (retrain after bugfix, restarts) | 4h | Hard ceiling |

**Hard ceiling:** 36 GPU-hours. If exceeded, triage: skip hyperparameter sweep (fix τ=0.8, K=5, β=10), reduce to 1 seed for ablation A4, or cut Phase 2 entirely (take post-hoc result only).

---

## Threats to Validity

| Threat | Type | Mitigation |
|--------|------|-----------|
| **Overfitting to KITTI Eigen split** (photometric loss exploits dataset-specific biases) | External validity | Final evaluation on KITTI 2015 (unseen scenes) before claiming generalization; flag if AbsRel delta > 2× training gap |
| **Photometric loss degeneracy** — uniform depth minimizes photometric error on textureless regions | Construct validity | Edge-aware loss weighting; monitor depth variance within batch; A0 baseline catches if uniform depth beats DA2 output |
| **Refinement relies on already-good regions** — if no pixel is confident, pass is a no-op | Internal validity | Pre-compute "confident-coverage" fraction on KITTI; if <30% of pixels have predicted confidence >0.8, the approach is not viable for this domain |
| **Metric hacking** — ECE < 0.05 via overconfident low-variance predictions that are still wrong | Internal validity | Report sharpness alongside ECE; require sharpness ≥ 0.8 (predictions span full [0,1] range) |
| **Single-dataset evaluation** does not generalize to indoor (NYUv2), aerial, or medical domains | External validity | Explicitly scoped to KITTI; generalization experiments deferred to follow-up; no KITTI-only result will claim domain-agnostic improvement |
| **3 seeds insufficient for tight confidence intervals** on high-variance metrics (AURC) | Statistical validity | Bootstrap 95% CI from 1,000 resamples of per-frame metrics within each seed; report mean width. If CI width > 20% of effect size, collect 10 seeds |
| **Code/setup errors** — DA2 feature extraction point may not match intended penultimate layer | Internal validity | Verification test: compare feature shapes, confirm output shape (B×C×H×W), integration test in CI before any training run |
| **Photometric loss does not equal depth accuracy** — loss can decrease while AbsRel increases | Construct validity | Periodic (every 1k steps) validation AbsRel check against held-out KITTI GT; if photometric loss ↓ but AbsRel ↑ for 3 consecutive checks, abort & flag as construct failure |

---

## Open Questions

### Resolved During Planning

- **Post-hoc vs. joint training:** Start post-hoc, switch to joint if refinement signal weak. Reflected in IU3 two-phase schedule.
- **Photometric loss weighting:** Standard L1+SSIM (α=0.85). Uncertainty-weighted deferred — prevents coupled learning signals at first.
- **Edge-of-frame handling:** Reflection padding on feature maps during KNN.
- **Mask boundary artifacts:** Gaussian blur (kernel=5, σ=2) on soft mask before refinement.

### Deferred to Implementation

- **Global sparse attention fallback frequency:** Measure how often local 7×7 window lacks low-uncertainty pixels. If >5% of pixels trigger fallback, increase window size or switch to global attention.
- **β temperature tuning:** β=10 initial, tune via calibration ECE on validation set. If ECE is insensitive to β in [5,20], fix at 10.

---

## Sources & References

- **Origin document:** `docs/brainstorms/2026-07-21-da2-uncertainty-refinement-requirements.md`
- **Related plan:** `docs/plans/2026-07-21-006-feat-da2-scale-anchoring-plan.md`
- KITTI Eigen split benchmark protocol (Eigen et al., 2014)
- Kendall & Gal (2017) — aleatoric uncertainty via learned log-variance
- Laplace interpolation for depth inpainting (local KNN weighting)
