# Original User Request

## 2026-07-21T23:13:12+05:30

DepthLab is a reproducible research platform and multi-head extension framework for monocular depth estimation built on Depth Anything V2.

Working directory: C:\Users\rushd\Downloads\prj-res
Integrity mode: development

## Priority Sequence
1. Official reproduction (non-negotiable foundation)
2. Immutable adapter architecture
3. Regression testing (`verify.py`)
4. Config-driven training
5. Head & dataset registries
6. Second head proving extensibility
7. Parameter-efficient adapter support (optional extension)
8. Feature ablation experiment hook (future experiment prep)

## Requirements

### R1. Phase 0.1 & 0.2: Environment Validation & Official DA2 Reproduction
- Validate official Depth Anything V2 environment and setup.
- Generate golden reference fixtures using official `run.py` on 10 sample images saved into `golden/`.
- Build a standalone `verify.py` script that validates checkpoint loading, inference execution, correct output shape (H×W), non-negative range, runtime bounds (≤1s per image at 518×518), and absence of NaNs/Infs under FP16 mixed precision with peak VRAM ≤ 2GB.

### R2. Phase 0.3: Evaluation Reproduction & Validation Gate
- Reproduce official evaluation metrics (AbsRel, δ1, RMSE) on the NYUv2 Eigen split within 1% of published values (paper AbsRel ~0.128 for DA2-Small).
- `verify.py` must enforce regression checks asserting output alignment within configurable numerical tolerance appropriate for selected precision.
- Produce `docs/reproduction/environment.md` and reproduction report before framework extraction.

### R3. Immutable Adapter Architecture & Submodule Protection
- The official Depth Anything V2 repository (`depth-anything-v2-official/`) is **immutable**. No source file inside `depth-anything-v2-official` may be modified.
- All customization, model wrapping, feature extraction, and head execution must occur strictly in the `depthlab/` adapter layer.

### R4. Phase 0.4: Multi-Head Extension Framework & Scaffolding
Construct the `depthlab/` modular framework per ADR-001 and ADR-002:
1. `backbone/loader.py` & `backbone/adapter.py`: Load frozen DA2 backbone with optional parameter-efficient adapter interface supporting pluggable implementations (initial implementation may be LoRA or identity).
2. `heads/base.py` & `heads/__init__.py`: Abstract `BaseHead` interface (`forward`, `compute_loss`, `compute_metrics`) and registry decorator `@register_head`.
3. `heads/relative_depth.py`: DINOv2-head decoder, `SILogLoss`, and relative depth metric suite (AbsRel, δ1-δ3, RMSE).
4. `data/`: Dataset registry with `@register_dataset`, flat YAML loader configs for NYUv2 and KITTI, and augmentation factory.
5. `trainer.py` & `train.py`: Single YAML config-driven training harness with PyTorch AMP FP16, GradScaler, cosine LR schedule, checkpointing, TensorBoard logging, and automatic reproducibility metadata tracking (`git commit`, `checkpoint hash`, `CUDA version`, `PyTorch version`, `seed`, `dataset version`, `benchmark version`).
6. `eval.py` & `metrics/`: Standalone evaluation script with metric dispatcher and architectural support hook for future feature ablation experiments (`--ablate-layer`).

### R5. Secondary Head Proof of Abstraction
- Implement a secondary head (`heads/uncertainty.py` for learned log-variance uncertainty estimation) implemented completely without modifying framework source outside the new head registration file.

## Acceptance Criteria

### Environment & Reproduction Criteria
- [ ] Golden reference fixtures saved in `golden/` for 10 sample images using official `run.py`.
- [ ] `verify.py` passes on a clean checkout (`python verify.py`), asserting outputs within configurable numerical tolerance and peak VRAM ≤ 2GB.
- [ ] NYUv2 evaluation metrics match published DA2-Small values within 1% margin.
- [ ] Reproduction documentation written to `docs/reproduction/environment.md`.

### Framework & Architecture Criteria
- [ ] Zero files modified in `depth-anything-v2-official/`; all integrations strictly reside in `depthlab/`.
- [ ] `python train.py --config config/default.yaml` executes forward/backward passes on dummy data without errors.
- [ ] `python train.py --config experiments/nyu_relative.yaml` completes one successful training epoch and produces checkpoints, TensorBoard logs, and decreasing training loss on NYUv2.
- [ ] Every experiment run automatically saves full reproducibility metadata (`git commit`, `checkpoint hash`, `CUDA version`, `PyTorch version`, `seed`, `dataset version`, `benchmark version`) into `experiment.yaml`.
- [ ] `eval.py --config experiments/nyu_relative.yaml --checkpoint ...` generates per-metric evaluation tables, TensorBoard logs, and depth visual outputs.
- [ ] Evaluation framework supports `--ablate-layer` hook for future feature ablation experiments.

### Head Abstraction Criteria
- [ ] `heads/uncertainty.py` is implemented and registered via `@register_head` without mutating any framework files.
- [ ] `python train.py --config experiments/nyu_uncertainty.yaml` executes training seamlessly.

## 2026-07-21T23:32:12Z — Generation 2 Resume Request
Resume work immediately:
Execute Milestone 3 (Phase 0.4 Multi-Head Extension Framework & Scaffolding in depthlab/) and Milestone 4 (Secondary Head Uncertainty Estimation) following the Project Pattern (Explorer -> Worker -> Reviewer -> Challenger -> Forensic Auditor).
All Milestone 1 and Milestone 2 deliverables have passed with 100% CLEAN audit verdicts. Proceed directly to Milestone 3!

