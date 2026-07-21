---
date: 2026-07-21
status: accepted
---

# ADR 002: Public API Stability, FeatureBundle, and Success Criteria

## Context

ADR-001 established the three-layer architecture (Official → Adapter → Research) and the core extension boundary (`get_intermediate_layers()` output). This ADR refines the framework's public API surface, data structures, and the objective success criteria for Phase 0 reproduction.

Several concerns motivate this ADR:
- Research code must know what it can depend on without breaking across versions.
- The raw `list[tuple[Tensor, Tensor]]` interface is brittle — adding fields later would cascade through every head.
- Generated artifacts (experiments) should not be mixed with source code.
- "Reproduced" is currently subjective. It needs objective pass conditions.

## Decisions

### 1. Public API Surface

The following APIs are guaranteed stable within a major version. Everything else is internal and may change without notice.

```python
# --- Layer 1: Adapter (stable) ---

class DA2Backbone(nn.Module):
    def __init__(self, variant: str)
    def forward(self, x: Tensor) -> Tensor
    def features(self, x: Tensor) -> FeatureBundle
    def load(self, path: str)
    @property
    def embed_dim(self) -> int
    @property
    def variant(self) -> str

class BaseHead(nn.Module):
    def forward(self, features: FeatureBundle) -> dict
    def compute_loss(self, preds: dict, batch: dict) -> Tensor
    def compute_metrics(self, preds: dict, batch: dict) -> dict

class FeatureBundle:
    stages: list[FeatureStage]  # always 4 stages
    @property
    def patch_tokens(self) -> list[Tensor]
    @property
    def cls_tokens(self) -> list[Tensor]
    @property
    def embed_dims(self) -> list[int]

class FeatureStage:
    patch_tokens: Tensor   # (B, N, D)
    cls_token: Tensor      # (B, D)
    stage_index: int       # 0-3, ordered shallow→deep
    embed_dim: int         # D for this stage

OFFICIAL_TRANSFORM: Compose  # single shared preprocessing pipeline

class ExperimentConfig:
    # Loaded from experiment.yaml
    model.variant: str
    head.type: str
    dataset.name: str
    training.seed: int
    benchmark_version: str

class BenchmarkProtocol: ...

# --- Config keys (stable) ---

model.variant         # vits | vitb | vitl
head.type             # relative | sef | temporal | uncertainty
dataset.name          # nyuv2 | kitti | hypersim | vkitti2
training.seed         # int
benchmark.version     # semver string
```

**Everything else is internal.** This includes:
- Internal methods of `DepthAnythingV2` beyond `get_intermediate_layers()`
- `DepthAnythingV2` internal modules (`DPTHead`, `DINOv2`, fusion blocks)
- The registry implementation (`_HEAD_REGISTRY`, `_DATASET_REGISTRY`)
- Checkpoint file formats
- TensorBoard event formats
- Any method not listed above

**Rationale:** A narrow public API means the framework can refactor internals without breaking research code. Adding to the public API is easy; removing from it is hard.

### 2. FeatureBundle

Replace the raw `list[tuple[Tensor, Tensor]]` with a named structure.

```python
@dataclass
class FeatureStage:
    patch_tokens: Tensor       # (B, N, D) — spatial tokens
    cls_token: Tensor          # (B, D) — class token
    stage_index: int           # 0 (shallowest) to 3 (deepest)
    embed_dim: int             # D for this variant

@dataclass
class FeatureBundle:
    stages: list[FeatureStage]  # always length 4

    @property
    def patch_tokens(self) -> list[Tensor]:
        return [s.patch_tokens for s in self.stages]

    @property
    def cls_tokens(self) -> list[Tensor]:
        return [s.cls_token for s in self.stages]

    @property
    def embed_dims(self) -> list[int]:
        return [s.embed_dim for s in self.stages]
```

**Why a dataclass instead of tuples:**
- Adding fields (e.g., `register_tokens`, `attention_maps`) later does not break existing heads
- Stage index makes ordering explicit — no positional assumptions
- Named access is self-documenting at the call site
- The `@property` shorthands keep common access patterns concise

**Usage in heads:**

```python
class SEFHead(BaseHead):
    def forward(self, features: FeatureBundle) -> dict:
        # All four stages available by index or iteration
        shallow = features.stages[0]
        deep = features.stages[3]

        # Quick access to all tokens
        patches = features.patch_tokens  # list of 4 tensors

        # Each stage carries its metadata
        assert shallow.stage_index == 0
        assert shallow.embed_dim == 384  # for vits
        ...
```

### 3. Separate `research/` from `runs/`

Source code and generated artifacts should not share a directory.

```
depthlab/
├── research/              # Source code (version controlled)
│   ├── heads/
│   ├── backbone.py
│   ├── transforms.py
│   ├── datasets.py
│   ├── metrics.py
│   ├── losses.py
│   ├── train.py
│   ├── eval.py
│   └── verify.py
│
├── runs/                  # Generated artifacts (git-ignored)
│   ├── EXP-001/
│   ├── EXP-002/
│   └── EXP-003/
│
├── benchmark/             # Source code (version controlled)
│   ├── protocol.py
│   └── strata/
│
├── configs/               # Source code (version controlled)
│   ├── variants.yaml
│   └── experiments/
│
├── golden/                # Test fixtures (version controlled)
│
└── docs/                  # Documentation (version controlled)
    ├── adr/
    ├── reproduction/
    ├── brainstorms/
    ├── ideation/
    └── plans/
```

- `research/` — all source code, including heads, training scripts, and evaluation
- `runs/` — experiment outputs, auto-created by `train.py`, git-ignored except `runs/.gitkeep`
- `benchmark/` — the frozen evaluation resource, has its own versioning

### 4. Version the Benchmark

The benchmark gets its own version, independent of the framework version.

```yaml
# benchmark/version.py
BENCHMARK_VERSION = "1.0.0"
```

Every experiment records which version it was evaluated against:

```yaml
# experiments/EXP-XXX/experiment.yaml
benchmark_version: 1.0.0
```

**Versioning rules:**
- Patch (1.0.0 → 1.0.1): sample corrections, documentation fixes. Scores should not change.
- Minor (1.0.0 → 1.1.0): new strata or sub-strata added. Scores may change.
- Major (1.0.0 → 2.0.0): protocol changes, sample removal, metric changes. Old results are not directly comparable.

The version is stored in `benchmark/__init__.py` and loaded by `BenchmarkProtocol` during evaluation.

### 5. Freeze the Reproduction Target

The reproduction report must record the following fields:

```yaml
# docs/reproduction/reproduction-report.yaml

official_commit: abc123def...
checkpoint_path: checkpoints/depth_anything_v2_vits.pth
checkpoint_sha256: a1b2c3d4...
dataset: NYUv2
dataset_version: 2021-01 (official split)
metric_implementation: depth_anything_v2_official/metric_depth/util/metric.py
metric_implementation_sha256: e5f6g7h8...
cuda_version: 12.6
pytorch_version: 2.12.1
gpu: RTX 4050 Laptop 6GB
driver: 610.62

results:
  abs_rel: 0.083  # must match published within 1%
  d1: 0.925
  rmse: 0.365

golden_mae: 1.2e-7  # verify.py tolerance against golden sample
```

Every field is mandatory. This report is the GO/NO-GO gate for Phase 1.

### 6. Phase 0 Success Criteria

| Check | Pass Condition |
|-------|---------------|
| Environment | Official model loads without modification on RTX 4050 6GB, CUDA 12.6 |
| Inference | Visual output of `run.py` on sample images matches official README examples |
| Metrics | Reported metrics within 1% of published values on NYUv2 Eigen split |
| Golden regression | `verify.py` reports MAE < 1e-6 against golden sample for all 10 images |
| Runtime | Single-image inference ≤ 1 second at 518×518 on RTX 4050 |
| VRAM | Peak VRAM ≤ 2GB for DA2-Small at 518×518 |
| No regressions | `verify.py --all` passes: checkpoint loads, inference runs, output shape (H×W), output range (non-negative), no NaNs, runtime bounded |

## Relationship to ADR-001

ADR-001 establishes the three-layer architecture and the extension boundary. ADR-002 refines the data structures within that boundary and defines what downstream code may depend on. Both ADRs together form the complete framework contract.

The key difference:

| Concern | ADR-001 | ADR-002 |
|---------|---------|---------|
| Layers | Official, Adapter, Research | — |
| Extension boundary | Feature maps | FeatureBundle dataclass |
| Model immutability | Never modify official | — |
| Public API | — | Explicit stable surface |
| Experiment organization | EXP-NNN folders | research/ vs runs/ |
| Benchmark versioning | — | Semantic versions |
| Phase 0 gates | Reproduction protocol | Objective success criteria table |

## Open Questions

- Should `FeatureBundle` support batching of multiple feature maps from a batch of images? Currently designed for single-batch usage. If multi-batch becomes necessary, the stage structure may need dimension annotations.
- The `research/` → `runs/` split assumes generated experiment artifacts don't need to be version-controlled. For reproducibility, the config snapshot and metrics.json should still be committed. Consider whether `runs/` should have a separate tracking mechanism.
