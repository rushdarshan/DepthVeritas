## Why

Depth Anything V2 provides a strong monocular depth foundation model, but the current project structure (7 parallel extension plans, no shared infrastructure, no reproduction verification) makes it impossible to build publishable research on a reliable foundation. Before any novel research can begin, we need a reproducible research platform with a verified baseline, frozen benchmark, and shared experiment infrastructure — otherwise every experimental result is built on an unvalidated foundation.

## What Changes

- Establish the official DA2 repository as a pinned, immutable upstream dependency
- Build a three-layer architecture: Official DA2 → Compatibility Adapter → Research
- Implement the reproduction protocol (environment validation, inference reproduction, evaluation reproduction with golden samples)
- Create the `DA2Backbone` adapter wrapping the official `DepthAnythingV2` model at the `get_intermediate_layers()` extension boundary
- Define `BaseHead` contract around `FeatureBundle` (typed feature map structure with stage metadata)
- Implement the initial relative-depth head reproducing the official DPT decoder
- Implement EXP-NNN experiment management with automatic metadata capture (commit, seed, CUDA version, benchmark version)
- Build the frozen evaluation benchmark with stratified failure-mode strata and error taxonomy sub-strata
- Create `verify.py` regression test suite referencing golden sample outputs
- Separate source code (`research/`) from generated artifacts (`runs/`)
- Add new directory: `docs/adr/` for architecture decision records

## Capabilities

### New Capabilities

- `backbone-adapter`: DA2Backbone wrapping official DepthAnythingV2 at the get_intermediate_layers() boundary, with FeatureBundle/FeatureStage dataclasses and variant config resolution
- `head-interface`: BaseHead abstract contract consuming FeatureBundle, with head registry for pluggable decoders
- `relative-depth-head`: Reproduction of the official DPT decoder head, verifying the head interface against known metrics
- `experiment-management`: EXP-NNN config-driven experiment system with automatic metadata capture, checkpointing, TensorBoard logging, and one-command reproducibility
- `reproduction-protocol`: Phase 0.1-0.3 environment/inference/evaluation reproduction pipeline with golden samples and verify.py regression suite
- `benchmark-framework`: Frozen evaluation benchmark with stratified failure-mode strata, error taxonomy sub-strata, semantic versioning, and standardized scoring protocol

### Modified Capabilities

None — this is a new project.

## Impact

- Creates `depthlab/` project directory with `research/`, `runs/`, `benchmark/`, `golden/` structure
- Pins `depth-anything-v2-official/` as a git submodule (immutable upstream)
- New artifacts: golden sample fixtures, reproduction report templates, EXP-NNN run directories
- Dependencies: torch ≥2.0, torchvision, numpy, opencv-python, matplotlib, tensorboard, pyyaml, einops (most already installed in environment)
- No modifications to the official DA2 repository — all changes are additive
