---
artifact_contract: ce-unified-plan/v1
artifact_readiness: implementation-ready
execution: code
title: Release-Ready DepthLab Research Platform
date: 2026-07-27
---

# Release-Ready DepthLab Research Platform

## Goal Capsule

Make the existing DepthLab framework reproducibly runnable on a 6 GB RTX 4050
and ready for collaborators to supply licensed datasets, execute experiments,
score results, and open a release PR without undocumented local steps.

## Scope Boundaries

This work does not accept dataset licenses, download account-gated data, invent
benchmark samples, train against absent data, or publish fabricated metrics.
Those are external research operations. The repository must instead fail early
with actionable messages and provide deterministic preparation, training, and
scoring paths once approved inputs are present.

## Requirements

- R1: Validate dataset manifests and assets before any training or scoring run.
- R2: Provide a GPU-aware preflight check for the documented RTX 4050 target.
- R3: Provide deterministic data preparation templates for RGB-D, video, and
  benchmark manifests without bundling licensed assets.
- R4: Make SEF warmup/end-to-end and temporal configs actionable on valid data.
- R5: Document installation, data intake, experiment execution, benchmark
  scoring, verification, and release limitations in a single README.
- R6: Add CPU-safe tests for new validation and configuration behavior.

## Key Technical Decisions

1. Use manifest validation rather than automatic large downloads. This preserves
   dataset-license compliance and supports user-provided local mirrors.
2. Keep preflight non-destructive and report actionable resource constraints;
   training is never started by a readiness check.
3. Treat benchmark version `1.0.0-dev` as explicitly unfrozen until provenance,
   counts, and baseline predictions are present.

## Implementation Units

### U1. Dataset and Hardware Preflight

Create `depthlab/preflight.py` and `scripts/preflight.py` to validate YAML
configs, checkpoint existence, dataset manifests, required sample fields,
referenced files, and CUDA availability/VRAM. Return structured results and a
nonzero exit code only for unmet requirements.

Files:
- Create: `depthlab/preflight.py`
- Create: `scripts/preflight.py`
- Create: `tests/test_preflight.py`

Tests:
- A valid temporary RGB-D manifest passes without CUDA.
- Missing depth and image files are listed deterministically.
- A missing checkpoint is reported without loading a model.

### U2. Data Intake Templates

Create documented CSV/JSON templates and a single preparation CLI that checks
RGB-D and video inputs, normalizes paths relative to the output manifest, and
creates deterministic train/validation splits when none are supplied.

Files:
- Create: `scripts/prepare_depth_manifest.py`
- Create: `data/templates/depth_manifest.csv`
- Create: `data/templates/video_manifest.json`
- Create: `tests/test_prepare_depth_manifest.py`

Tests:
- Input rows get stable IDs and deterministic splits.
- Invalid paths remain visible in validation output rather than being dropped.

### U3. Runnable Experiment and Benchmark Entrypoints

Wire preflight into documented experiment entry points without changing the
default `train.py` behavior. Add explicit `--preflight` modes to training and
benchmark CLIs so resource/data readiness can be validated before expensive
work begins.

Files:
- Modify: `train.py`
- Modify: `score.py`
- Modify: `scripts/run_research_pipeline.ps1`
- Modify: `config/sef_warmup.yaml`
- Modify: `config/sef_end_to_end.yaml`
- Create: `tests/test_cli_preflight.py`

Tests:
- `--preflight` exits successfully for a valid synthetic config.
- Invalid configs fail before model construction.

### U4. Release Documentation and CI

Create a concise project README, contribution guide, license, and CPU-only CI
workflow. The documentation distinguishes verified platform behavior from
dataset-dependent research results and gives the exact manifest-first workflow.

Files:
- Create: `README.md`
- Create: `CONTRIBUTING.md`
- Create: `LICENSE`
- Create: `.github/workflows/test.yml`
- Modify: `IMPLEMENTATION_STATUS.md`

Tests:
- CI runs the focused pytest suite and `verify.py` only when the available
  checkpoint/fixtures are present; otherwise it runs manifest/preflight tests.

## Verification Contract

- `python -m pytest tests -q`
- `python scripts/preflight.py --config config/default.yaml`
- `python train.py --config config/default.yaml --preflight`
- `python verify.py --all`
- `git diff --check`

## Definition of Done

The project has clear data-intake and preflight paths, an executable
manifest-first workflow, CPU-safe coverage, release documentation, and an
automated validation workflow. Empirical deliverables are clearly marked as
blocked until licensed datasets and completed GPU jobs provide real artifacts.
