## Context

This change establishes DepthLab — a reproducible research platform for monocular depth estimation built on Depth Anything V2. The platform must support multiple research extensions while maintaining reproducibility, upstream compatibility, and scientific rigor. The official DA2 repository is cloned at `depth-anything-v2-official/` and serves as the immutable Layer 0.

The architecture is documented in `docs/adr/001-depthlab-architecture.md` (three-layer architecture) and `docs/adr/002-depthlab-api-stability.md` (public API, FeatureBundle, benchmark versioning, Phase 0 success criteria). This design implements those ADRs.

## Goals / Non-Goals

**Goals:**
- Reproduce official DA2-Small metrics within 1% on NYUv2 Eigen split
- Build the DA2Backbone adapter wrapping official DepthAnythingV2
- Define BaseHead contract around FeatureBundle dataclass
- Implement EXP-NNN experiment management with automatic metadata capture
- Build the frozen failure-mode benchmark with strata and error taxonomy
- Create verify.py regression test suite with golden sample fixtures
- All reproducible with one command: `python train.py --config configs/experiments/EXP-XXX.yaml`

**Non-Goals:**
- Building any research heads beyond the initial relative-depth reproduction
- Modifying the official DA2 repository
- Deploying to mobile, web, or edge devices
- Multi-GPU or distributed training support
- Real-time inference pipelines

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Architecture | Three-layer (Official → Adapter → Research) | Isolates coupling surface; upstream updates only change the adapter |
| Extension boundary | `get_intermediate_layers()` output via FeatureBundle | DINOv2's stable API; 4 feature maps at configurable block indices |
| Backend | Official DA2's custom DINOv2 (not timm) | Official repo bundles its own DINOv2 implementation; timm adds unnecessary abstraction |
| Data structure | FeatureBundle dataclass (not raw tuples) | Adding fields later doesn't break heads; stage metadata is self-documenting |
| Experiment system | EXP-NNN with auto-captured metadata | Each experiment self-contained; one-command reproducibility |
| Benchmark | Semantic versioning, frozen after release | Prevents ambiguous comparisons; v1.0.0 before research begins |
| Transforms | Single OFFICIAL_TRANSFORM shared everywhere | Prevents preprocessing drift — the most common reproduction failure cause |
| Source vs artifacts | research/ (versioned) vs runs/ (gitignored) | Generated artifacts don't pollute version history |
| Model config | variant name → architecture params lookup | No duplicated configuration across experiments |

## Component Architecture

```
user command
     │
     ▼
train.py ───config───▶ configs/experiments/EXP-XXX.yaml
     │
     ├──▶ DA2Backbone (research/backbone.py)
     │       └──▶ depth-anything-v2-official/ (immutable submodule)
     │
     ├──▶ BaseHead (research/heads/base.py)
     │       └──▶ research/heads/relative.py (DPT reproduction)
     │
     ├──▶ OFFICIAL_TRANSFORM (research/transforms.py)
     │
     ├──▶ Trainer (research/train.py)
     │       └──▶ runs/EXP-XXX/ (checkpoints, metrics, tensorboard)
     │
     └──▶ verify.py
             └──▶ golden/ (fixtures)
                  benchmark/ (frozen evaluation)
```

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| CUDA 12.6 + PyTorch 2.12 may have compatibility issues with DA2's codebase | Phase 0.1 smoke test catches this within 1 hour |
| Preprocessing pipeline differs between official repo and our implementation | Single shared OFFICIAL_TRANSFORM; golden sample regression test |
| get_intermediate_layers() output shape changes across DINOv2 versions | FeatureBundle isolates heads from raw tensor shapes; adapter absorbs changes |
| Benchmark stratum curation effort underestimated | 1-2 week budget with priority strata first; use existing datasets only |
| DA2-Small metrics on RTX 4050 may not match published numbers exactly | GO/NO-GO gate at 1% tolerance; document hardware-specific deviations |
