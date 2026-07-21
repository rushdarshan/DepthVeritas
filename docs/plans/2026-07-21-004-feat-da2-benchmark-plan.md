# Depth Foundation Model Failure-Mode Benchmark — Implementation Plan

**Source:** `docs/brainstorms/2026-07-21-da2-failure-benchmark-requirements.md`
**Date:** 2026-07-21
**Depth:** Standard

---

## Unit 1 — Curation Pipeline: Stratum Extraction from Public Datasets

**Goal:** Extract 100–500 annotated samples per stratum from existing public datasets. No new data collection.

**Files:**
- `scripts/curate.py` — orchestrator: download helpers, per-dataset extractors, stratum-assignment heuristics
- `scripts/assign_stratum.py` — scene-metadata + COCO-pretrained object-detection → stratum label; human-verification flagging
- `data/strata/<stratum>/` — per-stratum symlinks/copies of (RGB, GT depth, stratum label, provenance JSON)
- `data/manifest.json` — master index: sample → dataset, stratum, license, train/test split, leakage flag

**Key decisions (from requirements):**
- No new data. Curate from NYU-Depth-v2, KITTI, DIODE, Sintel, Middlebury, ETH3D, DIML, Dark Zurich, WoodScape, BONN RGB-D, TUM RGB-D.
- Stratum assignment: heuristics + COCO-pretrained object detection (glass/mirror tags) + human verification. Target ~90% precision.
- Train/test split: per-dataset disjoint partitions; flag known leakage (NYUv2, KITTI used in DA2 training).
- License handling: Sintel CC-BY-NC handled via download script, not bundled assets.

**Acceptance:**
- P0 strata ≥200 samples each; P1/P2 ≥100 each.
- `manifest.json` tracks provenance, license, leak status, stratum per sample.
- Reproducible: pinned dataset versions, deterministic split seed.

**ponytail:** Curation uses existing heuristics + one COCO detector — no custom training, no manual labeling beyond verification. Add human-verification UI (e.g. a simple image viewer + accept/reject loop in ~50 lines) only if a single stratum falls below the 90% precision bar.

**Error taxonomy sub-strata:** Extend the 6 primary strata with finer-grained categories for reporting. Each sample gets a `sub_stratum` tag:
- Transparent/Reflective: `glass`, `mirror`, `water`, `polished_metal`
- Thin Structures: `wires`, `poles`, `railings`, `branches`, `hair`
- Low-Light: `indoor_dark`, `night_outdoor`, `near_infrared`
- Extreme FOV: `fisheye`, `wide_angle_120+`, `panoramic`
- HDR/Specular: `studio_lighting`, `specular_cg`, `glossy_floors`
- Video Flicker: `slow_motion`, `fast_motion`, `static_scene`

These sub-strata do not affect the composite score — they enable per-category reporting in papers (e.g., "SEF improves glass +12%, water +18%, mirror +3%").

---

## Unit 2 — Scoring Script + Evaluation Protocol

**Goal:** Single CLI that consumes predicted depth maps and ground truth, outputs per-sample / per-stratum / composite scores.

**Files:**
- `score.py` — entry point: `python score.py --pred_dir ./predictions --output results.json`
- `score/metrics.py` — AbsRel, δ1, RMSE, log10, frame-to-frame depth variance (σ²), temporal consistency error (TCE)
- `score/align.py` — least-squares scale/shift alignment for affine-invariant predictions
- `score/stratum.py` — stratum aggregation logic, weighted composite (transparent/thin at 2×)

**Key decisions:**
- Per-frame: AbsRel, δ1 accuracy, RMSE, log10.
- Per-video: σ² of per-pixel depth deltas (default flicker), TCE (secondary).
- Composite: macro-average across strata, transparent/thin weighted 2×. Report both "clean" (exclude leaked strata) and "full" composites.
- Handle missing GT gracefully (video clips with sparse GT frames).
- Output: JSON + markdown table.

**Acceptance:**
- `pip install -r requirements.txt && python score.py` produces valid JSON and markdown in ≤30s on a 500-sample directory.
- Aligned and raw scores both reported.
- Missing-GT frames silently excluded from affected metrics, logged to stderr.

**ponytail:** Single-file CLI. Alignment is 15 lines of least-squares. Flicker metric is pixel-wise variance — not a video model. Add TCE only if a reviewer with temporal-depth expertise flags the σ² metric as insufficient.

---

## Unit 3 — Baseline Inference (DA2)

**Goal:** Produce DA2 predictions for all curated samples + publish baseline tables.

**Files:**
- `scripts/run_baseline.py` — wraps existing DA2 inference, iterates over `data/manifest.json`, writes per-sample predicted depth to `predictions/`
- `results/baseline.json` — baseline scores (output of `score.py`)
- `results/baseline.md` — formatted baseline tables per stratum

**Key decisions:**
- Single A100 (~4 hrs for 2000 images).
- Report per-frame DA2 scores + at least one simple mitigation per P0 stratum (transparent-surface inpainting, temporal smoothing, thin-structure-aware loss finetune).
- Verifies stratification is correct: every stratum should be measurably worse than the aggregate.

**Acceptance:**
- DA2 predictions computed for every sample in the manifest.
- Baseline tables published in `results/baseline.md`.
- At least one mitigation shows statistically significant improvement on its target stratum without regressing others.

**ponytail:** Mitigations are simple post-processing passes (bilateral filter for temporal smoothing, 3-line inpainting mask from edge detection) — not new model training. Add full finetune only if post-processing can't move the P0 metric.

---

## Unit 4 — Documentation + Release Infrastructure

**Goal:** Benchmark is usable and citable by external researchers.

**Files:**
- `README.md` — motivation, stratum descriptions, evaluation protocol, download/usage instructions, baseline tables, citation format
- `requirements.txt` — pinned dependencies
- `LICENSE` — MIT
- `CONTRIBUTING.md` — PR workflow for new strata / models
- `paper/` — arXiv-ready ~4-page dataset-track paper

**Key decisions:**
- arXiv dataset-track paper (~4 pages).
- Stratified sample previews (RGB + GT depth + DA2 prediction) embedded in README.
- Reproducibility: pinned versions, deterministic eval, friend-test before release.

**Acceptance:**
- README documents all success criteria from requirements.
- Paper compiles to PDF via `cd paper && make` (or equivalent one-command build).
- A second researcher (friend) can reproduce baseline numbers from a fresh checkout.

**ponytail:** Paper is a markdown + pandoc pipeline, not LaTeX with 40 packages. DOI comes from Zenodo GitHub integration after the tag — no manual upload.

---

## Unit 5 — Release + Versioning + Verification

**Goal:** GitHub release, semantic benchmark versioning, Zenodo DOI, verification gate.

### Benchmark versioning policy
- Once frozen (Unit 5 acceptance), the benchmark uses semantic versions: `v1.0.0`, `v1.1.0`, `v2.0.0`.
- Every experiment records `benchmark_version: 1.0` in its `experiment.yaml`.
- No silent changes. Any stratum addition, removal, sample change, or protocol change increments the version.
- `v1.0.0` = initial release. `v1.1.0` = new strata added. `v2.0.0` = protocol-breaking changes.
- The `manifest.json` includes a `benchmark_version` field per release tag.

**Files:**
- `scripts/verify_reproducibility.py` — checksum-based verification that fresh checkout + `pip install -r requirements.txt && python score.py --pred_dir predictions --output results.json` produces identical scores to `results/baseline.json`
- GitHub Actions: `.github/workflows/verify.yml` — CI that runs the reproducibility check on every PR

**Key decisions:**
- Friend-test must pass before public release.
- CI verifies that no PR breaks scoring reproducibility.
- Zenodo auto-DOI on GitHub tag (MIT license).

**Acceptance:**
- CI green on the release commit.
- Friend-test passes (baseline numbers match).
- GitHub release tagged v1.0.0 with DOI registered.
- All strata have ≥200 (P0) / ≥100 (P1/P2) confirmed samples.
- `manifest.json` includes `benchmark_version: 1.0.0`.
- Error taxonomy sub-strata tagged in manifest (included at v1.0.0, not deferred).

**ponytail:** CI is a one-job, 3-minute check (no GPU needed — it checks against cached predictions). Zenodo is a one-time repo admin action, not code.

---

## Dependencies

- DA2 inference code (existing repo, not in this repo)
- Public dataset download URLs (documented in README)
- GPU (1× A100, ~4 hrs) — baseline inference only
- arXiv endorsement for dataset-track paper

## Outstanding Questions (from requirements doc)

| Q | Resolution | Handled In |
|---|------------|------------|
| Q1 — Exact flicker metric formula | Default to σ²; add TCE as secondary | Unit 2 |
| Q2 — Zero-shot cross-dataset stratum | Omitted from v1; reconsider if reviewers request it | N/A (deferred) |
| Q3 — Sintel CC-BY-NC license | Provide download scripts, not bundled assets | Unit 1 |
| Q4 — Clean vs. full composite | Report both | Unit 2 |
| Q5 — Long-term maintenance | Single owner initially; external strata via PR accepted (CONTRIBUTING.md) | Unit 4 |
