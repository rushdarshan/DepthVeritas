---
artifact_contract: ce-unified-plan/v1
artifact_readiness: implementation-ready
product_contract_source: ce-plan-bootstrap
title: "DepthLab Completion: Eleven Planned Capabilities"
date: 2026-07-29
status: active
---

# DepthLab Completion Plan

## Problem Frame

DepthLab has eleven approved directions but only the self-adapting and risk-aware foundations are substantially implemented. Complete the repository-owned, reproducible software surfaces for the remaining directions without representing unavailable datasets, trained checkpoints, or human-reviewed benchmark results as complete.

## Scope and Success Criteria

- Each plan has an importable implementation, deterministic focused tests, and a documented command or dashboard surface where relevant.
- Existing DA2 and SEF behavior remains the baseline; experimental modules are opt-in and preserve relative-depth semantics.
- Benchmark curation, model training, external SfM/segmentation assets, and publication remain explicit delivery gates rather than generated claims.

## Settled Decisions

- Complete the existing eleven plans before pursuing new ideation directions.
- Preserve honest validation boundaries: synthetic fixtures verify algorithms; only licensed, reviewed external data can substantiate real-world benchmark claims.
- Prefer first-slice controls before trainable or high-cost variants: fixed sparse cells, two-frame fusion, synthetic floor geometry, and EMA video control.

## Implementation Units and Sequence

1. **Stabilize existing foundations (Plans 1-2).** Finalize paired-frame adaptation validation and risk calibration inputs in `depthlab/adaptation/`, `depthlab/risk/`, `scripts/calibrate_and_test.py`, and their focused tests. Prevent dashboard or scripts from inventing uncertainty signals.
2. **Sparse and fusion primitives (Plans 3-4).** Add a registered sparse feature-grid head in `depthlab/heads/`, deterministic candidate selection/NMS and sparse metrics in `depthlab/metrics/`; add inverse-variance known-correspondence fusion in `depthlab/fusion/`. Cover fixed-cell output, selection, outlier rejection, and uncertainty reduction in `tests/`.
3. **Geometry and ordering (Plans 5-6).** Add calibrated 3D plane fit and confidence-gated Huber regularization in `depthlab/physics/`; add valid-pair sampling, ordinal loss, and ordinal metrics in `depthlab/ordinal/`. Test synthetic planes, gate abstention, ordering direction, ties, and invalid masks.
4. **Data and temporal controls (Plans 7-8).** Add seeded procedural-scene manifests and provenance validation in `depthlab/data/`; add semantic video-manifest checks, masked EMA control, and reset behavior in `depthlab/temporal/`. This supplies harnesses, not fabricated Blender or real-video evidence.
5. **Representation probe and benchmark gate (Plans 9-10).** Add DINO-feature affinity measurement with random/image-distance baselines in `depthlab/probes/`, plus release-manifest validation and deterministic stratum scoring in `depthlab/benchmark/`. The six-stratum release remains blocked until curation and review are complete.
6. **Interactive export (Plan 11).** Extend `dashboard.py` and reusable export helpers with click inspection, explicitly relative-scale colored PLY, and a prediction metadata bundle. Test exports independently of Streamlit UI plumbing.

## Dependencies and Risks

Plans 3-6 depend on existing `FeatureBundle`, uncertainty-head conventions, and geometry masks. Plans 7-10 depend on fixture schemas but not external assets. Training, 3-head ensemble checkpoints, Blender rendering, DINO weights, trusted poses, and the 1,200-sample benchmark are external runtime deliverables; code must fail clearly when they are absent.

## Verification

Run focused CPU tests for each unit, then `python verify.py` and the relevant dashboard/import smoke checks. Record exact commands and results in the completion report. Do not claim GPU, training, or benchmark success unless they were actually executed with the required inputs.

## Product Contract Preservation

The eleven feature plans remain the source of feature intent; this plan adds execution ordering and external-delivery boundaries without changing their scope.
