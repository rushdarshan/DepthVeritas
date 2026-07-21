# Project: DepthLab

## Architecture
DepthLab adopts a three-layer architecture (ADR-001 / ADR-002):
- **Layer 0: Official Depth Anything V2 (Immutable)**: `depth-anything-v2-official/` — zero source files modified.
- **Layer 1: Compatibility Adapter Layer**: `depthlab/backbone/`, `depthlab/data/`, `depthlab/transforms.py`, `depthlab/metrics/` — wraps official models and exposes unified `FeatureBundle` and dataset interfaces.
- **Layer 2: Multi-Head Research & Training Engine**: `depthlab/heads/`, `depthlab/trainer.py`, `train.py`, `eval.py`, `verify.py` — modular heads registered via `@register_head`, experiment tracking, evaluation, and reproducibility recording.

## Milestones
| # | Name | Scope | Dependencies | Status | Key Outputs / Conversation ID |
|---|------|-------|-------------|--------|--------------------------------|
| 1 | M1: Environment Validation & Golden Fixtures | Phase 0.1 & 0.2: validate environment, generate golden fixtures in `golden/`, build `verify.py` | None | DONE | Golden fixtures generated, verify.py passes (10/10), auditor verdict: CLEAN |
| 2 | M2: Evaluation Reproduction & Validation Gate | Phase 0.3: reproduce NYUv2 metrics (AbsRel ~0.128/0.083), write `docs/reproduction/environment.md` | M1 | DONE | `docs/reproduction/environment.md` & `reproduction-report.yaml` written, verify.py passing 11/11, auditor verdict: CLEAN |
| 3 | M3: Multi-Head Framework & Scaffolding | Phase 0.4: construct `depthlab/` framework per ADR-001/002, relative depth head, trainer, eval, reproducibility metadata | M2 | PLANNED | - |
| 4 | M4: Secondary Head (Uncertainty Estimation) | Phase 0.5: implement `depthlab/heads/uncertainty.py` via `@register_head` without touching framework source | M3 | PLANNED | - |

## Interface Contracts
### Layer 0 ↔ Layer 1 (Adapter)
- `DA2Backbone.features(x: Tensor) -> FeatureBundle`: extracts intermediate features via DINOv2 `get_intermediate_layers()` returning 4 stages of `FeatureStage(patch_tokens, cls_token, stage_index, embed_dim)`.
- `OFFICIAL_TRANSFORM`: single shared image preprocessing transform pipeline across train, eval, and verify.

### Layer 1 ↔ Layer 2 (Heads & Trainer)
- `BaseHead`: abstract base class (`forward(FeatureBundle) -> dict`, `compute_loss(preds, batch) -> Tensor`, `compute_metrics(preds, batch) -> dict`).
- `@register_head(name)`: decorator registering head implementations in head registry.
- `@register_dataset(name)`: decorator registering dataset loaders in dataset registry.

## Code Layout
```
depthlab/
├── backbone/
│   ├── loader.py        # DA2 checkpoint loader & frozen backbone wrapper
│   └── adapter.py       # DA2Backbone & adapter interfaces (LoRA / identity)
├── heads/
│   ├── __init__.py      # Head registry & decorator (@register_head)
│   ├── base.py          # BaseHead abstract contract around FeatureBundle
│   ├── relative_depth.py# Relative depth DPT head & SILogLoss
│   └── uncertainty.py   # Secondary head for log-variance uncertainty
├── data/
│   ├── __init__.py      # Dataset registry & decorator (@register_dataset)
│   ├── datasets.py      # NYUv2 & KITTI dataset loaders
│   └── transforms.py    # Augmentation factory & OFFICIAL_TRANSFORM
├── metrics/
│   ├── depth_metrics.py # Metric suite (AbsRel, d1-d3, RMSE, SILog)
│   └── dispatcher.py   # Metric dispatcher & ablation hook
├── trainer.py           # Training loop, AMP FP16, GradScaler, cosine LR, experiment.yaml writer
├── train.py             # Single entry point: python train.py --config ...
├── eval.py              # Evaluation script with --ablate-layer hook
└── verify.py            # Standalone regression & runtime verification script
```
