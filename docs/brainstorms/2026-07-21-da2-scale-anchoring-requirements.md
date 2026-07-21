---
date: 2026-07-21
topic: da2-scale-anchoring
---

# Test-Time Scale Anchoring via Physical Priors

## Problem Frame

DA2 (Depth Anything v2) produces affine-invariant depth — correct relative ordering and shape, but unknown global scale and shift. Re-Depth Anything (CVPR 2026 Findings) resolves this via diffusion SDS optimization at ~30s/frame. This is prohibitive for real-time or near-real-time use.

The core insight: most scenes contain at least one known-scale reference in the physical world. A ground plane + horizon line over-constrains (scale, shift) from a single frame, giving a closed-form solution in ~1-2s. No training, no diffusion, no multi-frame SLAM.

We formulate test-time scale anchoring as a convex optimization over (scale, shift) fitting a plane+depth model against a known camera height or horizon prior.

---

## Requirements

**R1. Horizon line / ground plane estimation from depth.**
Extract ground plane normal + camera height from DA2 depth gradient and semantic priors. Use the fact that gravity-aligned ground planes produce a characteristic depth gradient (z = a·u + b·v + c in image coordinates). Horizon line = vanishing line of the ground plane.

**R2. Camera height prior handling.**
Default priors: 1.6m for handheld capture, 1.2m for ground robots, 0.8–1.0m for tabletop. Allow user override. The height prior anchors absolute scale when the ground plane is visible.

**R3. Convex (scale, shift) optimizer.**
Closed-form least squares solution given plane+depth correspondences. Solve:
  argmin_{s, t} Σ_i (s·d_i + t - g(ui, vi))²
where d_i is DA2 disparity at pixel i, g(u,v) = a·u + b·v + c is the ground-plane depth model.
Set up as a 2×2 linear system — direct solve, no iteration.

**R4. Fallback for scenes without ground plane.**
Detect no-planar-ground via plane-fit residual threshold. Fallback: detect known-scale objects — car width (~1.8m), person height (~1.7m), door height (~2.0m). Use DA2 depth contours + a small zero-shot detection model (Grounding DINO) to segment candidates, then solve (scale, shift) from the known physical dimension. Implement two closed-form variants: (a) known absolute depth of one point fixes both s and t under strong prior; (b) known object dimension in world coordinates fixes s alone if t ≈ 0.

**R5. Evaluation against metric depth benchmarks.**
Evaluate on KITTI metric depth split (without using camera intrinsics). Metrics: δ1, δ2, δ3, AbsRel, RMSE. Compare against:
- Re-Depth Anything (test-time diffusion, ~30s/frame)
- Promptable Metric Depth (interactive clicks)
- DA2 + oracle scale/shift (upper bound)

---

## Success Criteria

- **Primary:** AbsRel < 0.10 on KITTI metric depth split without test-time training.
- **Oracle baseline:** Match or exceed DA2 + oracle scale/shift baseline on all KITTI metrics (AbsRel, δ1, δ2, δ3, RMSE).
- **Latency:** ≤ 2s per frame on a single A100 (target ~1s).
- **No-training:** Zero parameter updates. No checkpoint, no gradient descent.
- **Fallback coverage:** ≥ 80% of KITTI frames yield valid scale (plane or object fallback).

---

## Scope Boundaries

- **In scope:** Single-frame test-time scale/shift solving. Ground plane + object fallback. KITTI evaluation harness.
- **Out of scope:** Multi-frame temporal smoothing (future work). End-to-end training of DA2. Intrinsic calibration recovery. Real-time video pipeline.
- **Explicitly not doing:** Diffusion SDS, per-frame optimization with >50 iterations, or learned scale priors.

---

## Key Decisions

- **Decision 1: Convex over iterative.** Closed-form 2×2 linear system vs. 30-step Adam. The convex approach is cheaper, deterministic, and has no hyperparameter sensitivity. Accepts slightly lower accuracy than SDS in exchange for 15-30x speedup.
- **Decision 2: Horizon-first, object-fallback.** Ground plane + camera height is the primary path because it uses the full image and is more robust to occlusion. Object detection fallback only when plane fit residual exceeds threshold.
- **Decision 3: DA2 depth gradient for plane normal.** Use the depth map's own spatial gradient (∂d/∂u, ∂d/∂v) to estimate ground plane orientation, not a separate semantic segmentation network. Keeps the system zero-training.

---

## Dependencies / Assumptions

- **DA2 checkpoint** available via Hugging Face (`depth-anything/Depth-Anything-V2-Large-hf`).
- **Grounding DINO** for object-fallback (zero-shot detection, no fine-tuning).
- **Camera height prior** is approximately correct (±20%): the convex solution degrades gracefully with height error (scale error ≈ height error ratio).
- **Ground plane dominates** at least 30% of the image (typical for driving, handheld indoor, and ground robot views).

---

## Outstanding Questions

- Q1. How sensitive is the plane+depth fit to horizon line mis-estimation? (Run synthetic perturbation tests.)
- Q2. What is the minimum ground-plane pixel fraction for a reliable solve? (Sweep 5–50%.)
- Q3. Does Grounding DINO + DA2 object contour matching work at 512px resolution, or does the fallback need higher-res crops?
- Q4. How does Promptable Metric Depth perform on KITTI when we oracle the click positions? (Get a baseline before claiming improvement.)

---

## Next Steps

-> /ce-plan
