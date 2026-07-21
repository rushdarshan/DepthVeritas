---
date: 2026-07-21
status: accepted
superseded_by: adr-002  # ADR-002 refines the public API and data structures
---

> **Note:** ADR-002 builds on this document by defining the stable public API surface, the `FeatureBundle` dataclass, the `research/` vs `runs/` directory split, benchmark versioning, and objective Phase 0 success criteria. Both ADRs together form the complete framework contract.

# ADR 001: DepthLab Three-Layer Architecture

## Context

DepthLab is a reproducible research platform for monocular depth estimation built on Depth Anything V2. The platform must support multiple research extensions (SEF, temporal consistency, uncertainty refinement, etc.) while maintaining reproducibility, upstream compatibility, and scientific rigor.

The original framework plan assumed a generic `timm`-based ViT backbone. Discovery of the official DA2 codebase revealed that the official implementation uses a custom, self-contained DINOv2 with a stable `get_intermediate_layers()` API — not `timm`. This changes the framework's dependency assumptions but confirms the feasibility of a clean extension boundary.

## Decision

Adopt a **three-layer architecture** with strict dependency direction:

```
┌─────────────────────────────────────────────────┐
│ Layer 2: Research                               │
│                                                  │
│  heads/         benchmark/     experiments/      │
│  train.py       eval.py       verify.py          │
│                                                  │
│  Depends on: Layer 1 (Adapter)                   │
│  Never calls: Layer 0 directly                    │
├─────────────────────────────────────────────────┤
│ Layer 1: Compatibility Adapter                    │
│                                                  │
│  backbone.py    transforms.py   metrics.py        │
│  datasets.py    configs/                          │
│                                                  │
│  Owns the coupling surface to Layer 0             │
│  If official code changes, only this layer shifts │
├─────────────────────────────────────────────────┤
│ Layer 0: Official Depth Anything V2 (immutable)  │
│                                                  │
│  depth-anything-v2-official/                      │
│    depth_anything_v2/dpt.py                       │
│    depth_anything_v2/dinov2.py                    │
│    depth_anything_v2/dinov2_layers/               │
│    metric_depth/dataset/                          │
│    metric_depth/util/metric.py                    │
│                                                  │
│  Never modified.  Pinned via git submodule.       │
└─────────────────────────────────────────────────┘
```

## Layer Definitions

### Layer 0: Official DA2 (Immutable)

**Source:** `depth-anything-v2-official/` — pinned git clone of `github.com/DepthAnything/Depth-Anything-V2`.

**Rules:**
- Never modify a file in this directory.
- Never fork or copy code out of it.
- Pin to a specific commit. Upgrade deliberately with release notes.
- The adapter layer wraps, imports, and delegates to this code.

**Stable extension point:** `DINOv2.get_intermediate_layers(x, n, return_class_token=True)` returns 4 tuples of `(patch_tokens, cls_token)` at the configured block indices. This is the official DINOv2 API, not an internal detail. The indices differ by variant:

| Variant | Blocks | Indices | Embed Dim |
|---------|--------|---------|-----------|
| vits    | 12     | [2, 5, 8, 11] | 384 |
| vitb    | 12     | [2, 5, 8, 11] | 768 |
| vitl    | 24     | [4, 11, 17, 23] | 1024 |
| vitg    | 40     | [9, 19, 29, 39] | 1536 |

### Layer 1: Compatibility Adapter

**Files:**

| File | Responsibility |
|------|---------------|
| `backbone.py` | `DA2Backbone` class wrapping `DepthAnythingV2` |
| `transforms.py` | Single shared preprocessing pipeline (never duplicated) |
| `datasets.py` | Re-exports from `metric_depth/dataset/` |
| `metrics.py` | Re-exports `eval_depth()` from `metric_depth/util/metric.py` |
| `losses.py` | Re-exports `SILogLoss` from `metric_depth/util/loss.py` |
| `configs/variants.yaml` | Maps `model.variant: vits` to architecture params |

**The adapter is thin but load-bearing.** It delegates to Layer 0 for all actual computation. If the official repo updates, the adapter is the only file that changes.

**`DA2Backbone` interface:**

```python
class DA2Backbone(nn.Module):
    def __init__(self, variant: str):
        # Loads official DepthAnythingV2 with variant config
        # Freezes encoder by default

    def forward(self, x: Tensor) -> Tensor:
        # Returns depth (for direct inference)

    def features(self, x: Tensor) -> list[tuple[Tensor, Tensor]]:
        # Returns 4x (patch_tokens, cls_token) — the extension boundary
        # Each patch_tokens: (B, N, D)
        # Each cls_token: (B, D)

    def load(self, path: str):
        # Load pretrained checkpoint

    @property
    def embed_dim(self) -> int:
        # Returns D for the current variant
```

**Single transform to rule them all:**

```python
# transforms.py — defined once, imported everywhere
OFFICIAL_TRANSFORM = Compose([
    Resize(width=518, height=518, keep_aspect_ratio=True,
           ensure_multiple_of=14, resize_method='lower_bound',
           image_interpolation_method=cv2.INTER_CUBIC),
    NormalizeImage(mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225]),
    PrepareForNet(),
])
```

Training, inference, evaluation, and the benchmark all import this exact transform. Zero drift across code paths.

### Layer 2: Research

**Structure:**

```
depthlab/
├── heads/
│   ├── base.py          # BaseHead(ABC) — contract around features()
│   ├── relative.py      # Reproduction of official DPT head
│   ├── sef.py           # Surface Existence Field
│   ├── temporal.py      # Reprojection-consistency
│   └── uncertainty.py   # Uncertainty refinement
├── benchmark/
│   ├── protocol.py      # Frozen evaluation protocol
│   └── strata/          # Stratum definitions
├── experiments/
│   ├── EXP-001/         # Experiment folders
│   └── template.yaml    # Config template
├── train.py             # Single entry point
├── eval.py              # Evaluation entry point
├── verify.py            # Regression tests (golden sample)
└── configs/
    └── experiments/     # Per-experiment configs
```

**`BaseHead` contract:**

```python
class BaseHead(nn.Module):
    """Contract around feature maps, not the final depth tensor."""

    def forward(self, features: list[tuple[Tensor, Tensor]]) -> dict:
        """features[0..3] = (patch_tokens, cls_token) from DA2Backbone.features()
           Returns dict with at least {'depth': Tensor}"""
        ...

    def compute_loss(self, preds: dict, batch: dict) -> Tensor:
        ...

    def compute_metrics(self, preds: dict, batch: dict) -> dict[str, float]:
        ...
```

This design means:
- Relative depth head implements the exact DPT head from the official repo (reproduction)
- SEF head replaces the DPT head with a classification-based decoder
- Temporal head adds a recurrent or 3D component before the head
- All heads share the same encoder outputs — no redundant forward passes

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Backend | Official DA2 DINOv2 | Custom implementation, not timm. Using timm would add an unnecessary abstraction layer that doesn't match the upstream project. |
| Extension boundary | `get_intermediate_layers()` output | This is DINOv2's stable API, consistently returning 4 feature maps. It's the designed extension point. |
| Model immutability | Never modify official code | Enables clean upstream updates, easy bug comparison, and reproduction verification. |
| Adapter layer | Mandatory | Prevents research code from becoming coupled to DA2 internals. If upstream changes, only the adapter shifts. |
| Transforms | Single shared pipeline | Prevents preprocessing drift — the most common cause of reproduction failures. |
| Variant config | `model.variant: vits` | Maps to architecture params in a single lookup table. No duplicated configuration. |
| Metric reuse | Import from `metric_depth/` | SILog loss, eval_depth(), and dataset loaders are already implemented and tested. |
| Experiment organization | EXP-NNN folders | Each experiment self-contained with config, metrics, checkpoints, and paper figure references. |

## Experiment Lifecycle

```
python train.py --config configs/experiments/EXP-042.yaml
                          │
                          ▼
            Creates experiments/EXP-042/
                          │
                          ▼
            Saves: experiment.yaml (config snapshot + metadata)
                   checkpoints/ (backbone + head)
                   tensorboard/
                   visualizations/
                   metrics.json
                   _metadata (commit, CUDA, seed, timestamps)
```

Every experiment records `benchmark_version`, `dataset_version`, `git_commit`, and optionally `paper_section`/`figure_number` for figure reproducibility.

## Reproduction Protocol

Before any research begins, execute these steps in order:

1. **Environment validation** — verify Python, PyTorch, CUDA, GPU, driver, checkpoint hash
2. **Inference reproduction** — run official `run.py` on sample images, compare visual output
3. **Evaluation reproduction** — run official evaluation, verify metrics within 1% of published values
4. **Golden sample** — save 10 images with official depth outputs as regression test fixtures
5. **`verify.py`** — checks checkpoint loading, inference shape, output range, no NaNs, runtime bounds

Only after the reproduction report (containing checkpoint hash, commit, metrics, screenshots) is complete does Phase 1 (framework) begin.

## File Structure

```
depthlab/                          # Research platform root
├── research/                      # Source code (version controlled)
│   ├── backbone.py                # DA2Backbone adapter
│   ├── transforms.py              # Single shared preprocessing
│   ├── datasets.py                # Re-exported dataset loaders
│   ├── metrics.py                 # Re-exported evaluation metrics
│   ├── losses.py                  # Re-exported training losses
│   ├── verify.py                  # Regression test suite
│   ├── train.py                   # Single entry point
│   ├── eval.py                    # Evaluation entry point
│   │
│   ├── heads/
│   │   ├── __init__.py            # Head registry
│   │   ├── base.py                # BaseHead abstract class
│   │   ├── relative.py            # DPT head reproduction
│   │   ├── sef.py                 # Surface Existence Field
│   │   ├── temporal.py            # Reprojection-consistency
│   │   └── uncertainty.py         # Uncertainty refinement
│   │
│   └── configs/
│       ├── variants.yaml          # Model architecture params
│       └── experiments/           # Per-experiment configs
│           └── template.yaml
│
├── runs/                          # Generated artifacts (git-ignored)
│   ├── EXP-001/                   # Auto-created by train.py
│   └── .gitkeep
│
├── benchmark/                     # Frozen evaluation resource
│   ├── __init__.py                # BENCHMARK_VERSION
│   ├── protocol.py                # Evaluation protocol
│   └── strata/                    # Stratum definitions
│
├── golden/                        # Golden sample fixtures (version controlled)
│
├── scripts/
│   ├── download_checkpoints.py
│   └── prepare_datasets.py
│
├── docs/
│   ├── adr/                       # Architecture decision records
│   ├── reproduction/              # Reproduction report artifacts
│   ├── brainstorms/               # Requirements documents
│   ├── ideation/                  # Ideation artifacts
│   └── plans/                     # Implementation plans
│
├── requirements.txt
└── .gitignore                     # runs/ is git-ignored
```

> **Note:** See ADR-002 for the stable public API surface, `FeatureBundle` dataclass definition, the `research/` vs `runs/` split rationale, benchmark versioning rules, and objective Phase 0 success criteria.
