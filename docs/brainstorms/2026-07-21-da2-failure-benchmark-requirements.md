---
date: 2026-07-21
topic: da2-failure-benchmark
---

# Depth Foundation Model Failure-Mode Benchmark

## Problem Frame

DA2 produces state-of-the-art metric depth across domains, but fails systematically on known failure modes — transparent surfaces, thin structures, extreme FOV, low-light, HDR scenes, specular highlights, and temporal flicker in video. No existing depth benchmark (NYU-Depth-v2, KITTI, DIODE, Sintel, Middlebury) stratifies by failure mode, so reported aggregate metrics mask these blind spots. A curated, per-stratum benchmark would:

1. Expose the true failure profile of any depth model.
2. Measure whether a proposed fix actually improves the targeted stratum (without regressing others).
3. Outlive any single model as a citable community resource — the highest-leverage non-code investment because it steers all research decisions.

---

## Requirements

- **R1. Curate failure-mode strata from existing public datasets.** Extract 100-500 annotated samples per stratum from NYU-Depth-v2, KITTI, DIODE, Sintel, Middlebury, ETH3D, and in-the-wild collections. No new data collection. Strata (priority order):

  | Priority | Stratum | Description | Source Candidates |
  |----------|---------|-------------|-------------------|
  | P0 | Transparent/Reflective | Glass, mirrors, water, polished metal | NYUv2 (glass doors/windows), DIODE (indoor reflections), wild |
  | P0 | Thin Structures | Objects <2px at native resolution: poles, wires, railings, branches | KITTI (pedestrian crossings), Sintel (hair/fur), ETH3D |
  | P1 | Low-Light / Night | ISO > 1600, lux < 10, or near-infrared capture | NYUv2 (dark rooms), DIML, Dark Zurich, night KITTI |
  | P1 | Extreme FOV | >120° horizontal FOV (fisheye, wide-angle) | Sintel (synthetic extreme), WoodScape, wild panoramas |
  | P2 | HDR / Specular | High dynamic range scenes, bright highlights, glossy floors | DIODE (studio lighting), Sintel (specular CG materials) |
  | P2 | Video Flicker | Adjacent frames with unstable depth output | Sintel (temporal), BONN RGB-D, TUM RGB-D, custom phone video |

- **R2. Define per-stratum evaluation protocol.**
  - Per-frame: absolute relative error (AbsRel), delta1 accuracy, RMSE, log10.
  - Per-video: temporal consistency (frame-to-frame depth delta variance), flicker frequency metric.
  - Stratum score: macro-average of per-frame metrics within the stratum.
  - Composite score: weighted macro-average across all strata (weighted by priority; transparent/thin at 2×).
  - Standardized train/test split: no overlap with any model's training data. Report which datasets have known train leakage.
  - Noise-floor note: stratum assignment targets ~90% precision; improvements below that threshold may not be statistically detectable within a stratum.

- **R3. Implement standardized scoring script.**
  - Single Python CLI: `python score.py --pred_dir ./predictions --output results.json`
  - Reads ground-truth depth (where available), predicted depth; aligns scale if metric depth not provided.
  - Outputs per-sample, per-stratum, and composite JSON + markdown table.
  - Handles missing ground truth (e.g., video clips with only sparse GT frames).
  - 1-2 weeks curation + implementation.

- **R4. Document benchmark, baselines, and usage.**
  - README: benchmark motivation, stratum descriptions, download links, evaluation protocol.
  - Baseline reference tables: DA2 per-frame, DA2 + simple per-stratum mitigations (e.g., transparent-surface inpainting, temporal smoothing for flicker, thin-structure-aware loss finetune).
  - Citation format for research use.

- **R5. Release as open-source resource.**
  - GitHub repo: `github.com/openedai/depth-failure-benchmark`. MIT license.
  - arXiv paper (dataset & benchmark track) — ~4 pages.
  - Stratified sample previews (RGB + GT depth + DA2 prediction per stratum).
  - Reproducibility: pinned dataset versions, deterministic evaluation.

---

## Success Criteria

1. Each priority stratum has ≥200 annotated samples (P0) or ≥100 (P1/P2).
2. Scoring script runs on a fresh checkout in ≤2 commands (`pip install -r requirements.txt && python score.py`).
3. Baseline scores for DA2 per-frame published in the README — every stratum measurably worse than aggregate suggests correct stratification.
4. At least one published fix (DA2 variant, post-processing, or alternative method) shows statistically significant improvement on its target stratum without regressing others.
5. Reproducibility verified by a second researcher (friend test) before release.

---

## Scope Boundaries

| In scope | Out of scope |
|----------|--------------|
| Curating existing public dataset samples into strata | Collecting new sensor data or synthetic renders |
| Per-stratum and composite scoring protocol | Training or fine-tuning models |
| Baseline inference with DA2 | Building a leaderboard platform or website |
| Temporal flicker metric for video | Model comparison dashboard (just the benchmark data + script) |
| arXiv paper (dataset track, ~4 pages) | Full conference paper with novel method |

---

## Key Decisions

- **No new data.** All samples come from existing labeled datasets. Document provenance and license per sample.
- **Stratum assignment by heuristics + human verification.** Use scene metadata (e.g., NYUv2 `scene_type`), object-detection tags (glass/mirror labels from COCO-pretrained), and manual verification for ambiguous cases. Accept ~90% precision over algorithmic purity.
- **Weighted composite score.** Transparent/thin structures weighted 2× vs other strata because they are DA2's most visible failures and the hardest to fix.
- **Temporal metric choice.** Frame-to-frame depth variance (σ² of per-pixel depth deltas) as primary flicker metric; secondary: temporal consistency error (TCE) from Kopf et al. Keep both in the script, report σ² as default.
- **Scale alignment.** For models outputting affine-invariant or shifted depth, apply least-squares scale/shift alignment against ground truth per image before computing errors (standard practice). Report both aligned and raw scores.
- **Train leakage disclosure.** NyuV2 and KITTI were used in DA2 training. Report known overlap explicitly per stratum so users can factor it in.

---

## Dependencies / Assumptions

- DA2 inference code is available (existing repo) to produce baselines.
- Public dataset download URLs remain accessible.
- GPU available for baseline inference (single A100 or equivalent, ~4 hrs for 2000 images).
- At least one contributor can dedicate 1-2 weeks for curation + implementation.
- arXiv endorsement or moderation willing to accept dataset-track papers.

---

## Outstanding Questions

- Q1. What is the exact flicker metric formula — is there a published temporal depth consistency metric we should standardize on, or do we define our own?
- Q2. Should we include a zero-shot cross-dataset transfer stratum (train on NYUv2, test on KITTI) as a separate generalization stratum?
- Q3. License compatibility: some Sintel scenes are CC-BY-NC. Can we include them in a benchmark dataset, or do we provide download scripts instead?
- Q4. Should the composite score exclude strata where the model's training data overlaps heavily (e.g., DA2 on NYUv2 transparent), or report both "clean" and "full" composites?
- Q5. Who maintains the benchmark after initial release? Do we accept external strata contributions via PR?

---

## Next Steps

-> /ce-plan
