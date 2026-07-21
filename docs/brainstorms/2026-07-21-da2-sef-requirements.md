---
date: 2026-07-21
topic: da2-surface-existence-field
---

# Surface Existence Field (SEF) — Depth as Distribution over Learned Planes

## Problem Frame

Researchers and engineers using Depth Anything V2 for scene understanding in real-world settings (robotics, autonomous driving, AR/VR) are affected. DA2 produces high-quality single-depth estimates but fails systematically on transparent surfaces (glass, water) and reflective/specular surfaces where multiple physical depths correspond to one pixel. It also provides zero uncertainty calibration — the model is equally confident on an ambiguous reflection as on a solid wall. This erodes trust in downstream planners and filters. SEF replaces per-pixel scalar depth with a learned categorical distribution over N depth bins, naturally capturing multi-modal depth ambiguity and producing calibrated uncertainty via softmax entropy.

## Requirements

### Decoder Head
- R1. Replace the DA2 final regression layer with a classification head that outputs a probability vector over N learned depth bins per pixel (target N=64–128).
- R2. Implement learned bin centers via a small MLP that reads pooled image features and predicts bin-width boundaries per image (AdaBins-style), enabling adaptive discretization.
- R3. Keep the change localized to the decoder head — DINOv2 encoder and DPT feature-fusion backbone remain frozen during SEF head warmup, then fine-tuned end-to-end.

### Loss Function
- R4. Loss = cross-entropy between predicted bin probabilities and the ground-truth depth binned into the predicted centers, plus a bin-center regression loss (L1 between predicted and optimal bin centers for each GT depth).
- R5. Add an optional entropy regularization term to discourage overconfident distributions on known-ambiguous pixels during training (only if validation ECE does not improve with CE alone).

### Training & Data
- R6. Pre-train SEF head on standard depth datasets (KITTI, NYUv2) with DA2 encoder frozen; fine-tune end-to-end on a curated mix that includes transparent/reflective scene subsets.
- R7. Curate or scrape a transparent/reflective evaluation subset (at minimum 200–500 labeled frames across indoor and outdoor scenes) from existing datasets (Hypersim, Replica, TORSD) to enable targeted metrics.

### Evaluation Infrastructure
- R8. Implement uncertainty calibration metrics: Expected Calibration Error (ECE), Area Under Risk-Coverage Curve (AURC), and negative log-likelihood (NLL) of the predictive distribution.
- R9. Report per-pixel uncertainty via softmax entropy (H[p]) and compare entropy maps against known ambiguous regions.
- R10. Provide a comparison table against vanilla DA2 and DA2 + MC Dropout on: AbsRel, RMSE, δ1.25 accuracy, ECE, AURC.

### Inference
- R11. Depth prediction at inference = marginal expectation over bins (softmax-weighted sum of bin centers). Entropy computed in the same forward pass with negligible overhead.
- R12. Memory: keep N=96 as default (validated by AdaBins as the sweet spot); the head adds ~4M params and fits within fp16 + grad accum on 6GB VRAM.

## Success Criteria

- SEF matches or improves vanilla DA2's AbsRel/RMSE on KITTI and NYUv2 standard splits, while providing per-pixel calibrated uncertainty.
- ECE ≤ 0.05 on calibration-holdout splits from the transparent/reflective subset.
- Multi-modal depth distributions visually confirmed on at least 3 transparent-surface examples (bimodal peaks at foreground-glass depth and background depth).
- Downstream implementer can reproduce all results from the following artifacts: training config, evaluation script, pretrained checkpoint, and a single-table results summary.
- A 1-page handoff doc fits in an agent's context and conveys: architecture change, loss, data pipeline, evaluation command, expected results.

## Scope Boundaries

- No changes to the DINOv2 encoder or DPT feature-fusion core — only the final prediction head and loss.
- No real-time / mobile deployment target — single-RTX-4050 offline evaluation only.
- No synthetic-data generation pipeline for transparent surfaces — only existing dataset curation.
- No comparison against SOTA uncertainty methods (Ensembles, SDE) — only MC Dropout as baseline.
- No video / temporal consistency — single-frame depth only.

## Key Decisions

- **Classification over regression**: AdaBin's established ~10% RMSE improvement on classification-based depth, plus free calibrated uncertainty via softmax entropy, justifies the swap. Regression gives neither multi-modal outputs nor calibration without expensive sampling.
- **Learned bin centers per image (AdaBins style) over fixed uniform bins**: Adaptive discretization handles the heavy-tailed depth distribution in natural scenes better than uniform bins; uniform bins waste capacity on empty far-field ranges.
- **N=96 default**: Matches AdaBins' empirically validated sweet spot. N=128 is a tuning knob for planning (memory vs. resolution tradeoff).
- **Keep encoder frozen during SEF head warmup**: 4M new params trained in isolation first avoids corrupting the rich DINOv2 features; full fine-tune only after head converges.
- **MC Dropout over Deep Ensembles as baseline**: Ensembles require 5× params and 5× training — implausible on 6GB. MC Dropout is the pragmatic uncertainty baseline within compute budget.

## Dependencies / Assumptions

- DA2-Small pretrained checkpoint available and loadable in PyTorch (DA2-Mini also fits — SEF is model-agnostic across sizes).
- Depth datasets (KITTI, NYUv2) already in standard format; Hypersim/Replica/TORSD require download and labeling effort (~3 days).
- AdaBins bin-center MLP implementation is MIT-licensed and adaptable without legal risk.
- 6GB VRAM can hold DA2-Small + SEF head at 384×384 with fp16 and gradient accumulation step ≥ 2. Validated by DA2's own published memory figures.

## Outstanding Questions

### Resolve Before Planning
- [None — problem scope, approach, and resource constraints are sufficiently defined.]

### Deferred to Planning
- Exact bin-center MLP architecture (2-layer vs 3-layer, hidden dim, output = N bin boundaries per image).
- Whether entropy regularization is needed or CE alone gives sufficient calibration — decide from first SEF-small training run.
- Training hyperparameters (lr, warmup steps, grad accum factor) — depends on batch size that fits 6GB at chosen resolution.
- How to handle pixels where GT depth is invalid/missing in the cross-entropy binning loss (mask them out or assign uniform target).
- Transparent-surface evaluation subset composition — list specific scenes from Hypersim/Replica after inspecting their metadata.

## Next Steps

-> /ce-plan for structured implementation planning
