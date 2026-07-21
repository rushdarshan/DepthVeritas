# DepthLab Master Orchestration Plan

## Overview
DepthLab is a reproducible research platform and multi-head extension framework for monocular depth estimation built on Depth Anything V2.

## Milestones & Execution Strategy

### Milestone 1: Environment Validation, Golden Fixtures & `verify.py`
- **Goal**: Validate DA2 environment, generate 10 golden fixtures in `golden/`, build `verify.py` for regression & runtime verification.
- **Key Deliverables**:
  - `golden/` fixtures (10 images + depth outputs)
  - `verify.py` (checkpoint loading, shape H×W, non-negative range, runtime ≤1s @ 518×518, VRAM ≤ 2GB, FP16, no NaNs/Infs, MAE < 1e-6 regression check)

### Milestone 2: Evaluation Reproduction & Validation Gate
- **Goal**: Reproduce official NYUv2 Eigen split metrics (AbsRel, δ1, RMSE) within 1% of published/checkpoint values.
- **Key Deliverables**:
  - `docs/reproduction/environment.md` with complete environment details and metric tables.
  - Evaluation reproduction gate verification.

### Milestone 3: Multi-Head Framework & Scaffolding (`depthlab/`)
- **Goal**: Build modular framework per ADR-001/ADR-002 without mutating `depth-anything-v2-official/`.
- **Key Deliverables**:
  - `depthlab/backbone/loader.py` & `backbone/adapter.py` (`DA2Backbone`, `FeatureBundle`, `FeatureStage`, adapter interface)
  - `depthlab/heads/base.py` & `heads/__init__.py` (`BaseHead`, `@register_head`)
  - `depthlab/heads/relative_depth.py` (`RelativeDepthHead`, `SILogLoss`, metrics)
  - `depthlab/data/` (`@register_dataset`, YAML loader configs, augmentations)
  - `depthlab/trainer.py` & `train.py` (AMP FP16, GradScaler, cosine LR schedule, checkpointing, TensorBoard logging, reproducibility metadata saved to `experiment.yaml`)
  - `depthlab/eval.py` & `metrics/` (metric dispatcher, `--ablate-layer` hook)
  - Configs: `config/default.yaml`, `experiments/nyu_relative.yaml`

### Milestone 4: Secondary Head (Uncertainty Estimation) & Extensibility Proof
- **Goal**: Prove head abstraction by adding a second head without mutating any existing framework files.
- **Key Deliverables**:
  - `depthlab/heads/uncertainty.py` (`UncertaintyHead`, registered via `@register_head`)
  - `experiments/nyu_uncertainty.yaml`
  - Successful training & evaluation pass.

## Verification & Integrity Policy
- Explorer → Worker → Reviewer → Challenger → Forensic Auditor loop.
- Zero code mutation in `depth-anything-v2-official/`.
- Mandatory Forensic Auditor check before milestone sign-off.
