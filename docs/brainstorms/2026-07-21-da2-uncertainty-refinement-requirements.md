---
date: 2026-07-21
topic: da2-uncertainty-refinement
---

# Closed-Loop Uncertainty-Guided Depth Refinement

## Problem Frame

Monocular depth estimation outputs are always uncertain — textureless regions, reflective surfaces, motion blur, and object boundaries produce unreliable predictions. Existing uncertainty methods ( Monte Carlo dropout, ensemble variance, learned aleatoric/logit heads) output a per-pixel uncertainty map but **never act on it**. The map is visualized, logged, or used for filtering — never as a signal to improve the depth estimate itself.

**Closing the loop** — predict uncertainty, then use it as a differentiable mask to gate a refinement pass — creates a self-supervised training signal: confident regions supervise uncertain ones via interpolation. No ground truth required.

This work adds a lightweight uncertainty head to DA2 and a refinement stage that replaces high-uncertainty pixels with Laplace-interpolated values from low-uncertainty neighbors, trained end-to-end with photometric consistency.

---

## Requirements

### R1. Uncertainty Prediction Head (3-layer MLP on DA2 intermediate features)

- Attach to DA2 decoder's penultimate feature map (or bottleneck)
- 3-layer MLP: hidden dim 64 → 32 → 1, ReLU activations
- Output: per-pixel log-variance (aleatoric uncertainty)
- <1M params, compatible with DA2's existing architecture
- Training: jointly with DA2 decoder, or post-hoc frozen-encoder finetune
- Loss: learned uncertainty via negative log-likelihood on depth residual, or auxiliary photometric consistency term

### R2. Differentiable Refinement Pass (Laplace Interpolation with Uncertainty Mask)

- Forward pass (inference-only once head is trained):
  1. Threshold uncertainty map at quantile `τ` (e.g., top 20% pixels)
  2. For each high-uncertainty pixel, find K nearest low-uncertainty neighbors in feature space (e.g., K=5)
  3. Replace depth value with weighted average of neighbors (weights = inverse uncertainty / spatial distance — Laplace kernel)
- Must be differentiable for end-to-end training:
  - Hard threshold → soft mask via sigmoid-temperature gating
  - Nearest-neighbor selection → differentiable via attention-like weighting over all low-uncertainty pixels (bottlenecked to local window for memory)
- Compatible with SEF: SEF provides the uncertainty distribution; this module consumes it

### R3. Self-Supervised Training Loop

- Loss: photometric consistency (L1 + SSIM) between warped target image and reference frames, computed **after refinement**
- Gradient flows through refinement pass into uncertainty head → uncertainty head learns to route confident gradients
- No ground-truth depth needed
- Iterative: refine → compute photometric loss → update uncertainty head + decoder (or decoder-only, uncertainty head fixed for stability)

### R4. Evaluation

- **Depth accuracy:** absolute relative error (AbsRel), δ1.25 before vs. after refinement
- **Uncertainty calibration:** Expected Calibration Error (ECE), Area Under the Risk-Coverage Curve (AURC)
- **Ablation:** (a) no refinement, (b) refinement with uniform mask, (c) refinement with oracle mask (GT error)
- **Qualitative:** side-by-side depth maps + uncertainty maps before/after refinement
- Compute budget: FPS before vs. after refinement (refinement should add <15% overhead on trained head)

---

## Success Criteria

| Criterion | Target |
|-----------|--------|
| Depth AbsRel improvement after refinement | ≥5% relative on KITTI Eigen split |
| Uncertainty calibration (ECE) | ≤0.05 after training |
| Refinement overhead | <15% additional inference time (trained head only) |
| MLP params | <1M |
| GPU memory (6GB target) | Peak ≤5.5GB during training |
| Self-supervised (no GT depth) | Photometric loss converges, depth quality reaches supervised baseline within 10% |

---

## Scope Boundaries

| In Scope | Out of Scope |
|----------|-------------|
| Lightweight uncertainty MLP on DA2 features | Training a full depth network from scratch |
| Laplace interpolation refinement | Learned refinement (e.g., second network) |
| Photometric self-supervision | Using GT depth for refinement training |
| KITTI evaluation | Multiple datasets beyond KITTI |
| SEF compatibility | Modifications to SEF itself |
| Aleatoric uncertainty only | Epistemic uncertainty (ensembles, dropout) |

---

## Key Decisions

1. **Post-hoc vs. joint training:** Start post-hoc (freeze DA2 encoder/decoder, train uncertainty head only), switch to joint if refinement signal is weak.
2. **Soft threshold design:** `σ(t) = sigmoid(β · (u − u_τ))` where `u_τ` is the quantile threshold, `β` is temperature. Trade-off: high `β` ≈ hard mask (cleaner but non-differentiable in practice), low `β` is smoother but leaks low-uncertainty pixels into refinement.
3. **Neighbor search space:** Local 7×7 window per pixel to keep attention-like weighting O(HWk²) rather than O(H²W²). Global sparse attention if local window misses low-uncertainty pixels (edge case: large uniform uncertain region).
4. **K in KNN:** K=5 initially, validate with K=3 and K=10. Higher K smooths more but may wash out fine detail.
5. **Quantile τ:** τ = 0.8 (refine top 20% uncertain pixels). Tune on validation set.

---

## Dependencies / Assumptions

- DA2 checkpoint available (intermediate features accessible)
- SEF integrated (provides uncertainty distribution — this module is downstream)
- PyTorch ≥1.13, CUDA ≥11.6
- KITTI raw dataset available for self-supervised training
- 6GB GPU: training batch size ≤8, resolution ≤640×192

---

## Outstanding Questions

1. Does the refinement pass need to be trained end-to-end, or is the uncertainty head trained standalone, then refinement applied as a fixed post-process?
   - Prototype: standalone first. If refinement improves things, unwrap the gradient path.
2. What photometric loss weighting? Standard L1+SSIM or uncertainty-weighted (confident pixels weighted more)?
   - Uncertainty-weighted is more principled but couples two learning signals. Start standard.
3. How to handle edge-of-frame pixels where neighbor window falls partly outside the image?
   - Reflection padding on feature maps during neighbor search.
4. Does the refinement pass introduce artifacts at the boundary of the uncertainty mask?
   - Likely — feather the mask (blur threshold with Gaussian) to avoid hard seams.

---

## Next Steps

→ /ce-plan
