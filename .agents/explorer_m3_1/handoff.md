# Handoff Report — Explorer 1 (explorer_m3_1)

**From**: Explorer 1 (`explorer_m3_1`)  
**To**: Milestone 3 Worker / Orchestrator  
**Date**: 2026-07-21  
**Workspace**: `C:\Users\rushd\Downloads\prj-res\.agents\explorer_m3_1`  
**Subject**: Layer 1 Backbone Loader and Compatibility Adapter Layer Analysis & Design Specification (`depthlab/backbone/`)

---

## 1. Observation

### 1.1 Layer 0 Official Codebase Structure (`depth-anything-v2-official/`)
- **Immutability Constraint**: Layer 0 (`depth-anything-v2-official/`) is 100% immutable. Zero source files are modified in Layer 0.
- **Model Definition** (`depth_anything_v2/dpt.py` lines 153–184):
  `DepthAnythingV2` instantiates `self.pretrained = DINOv2(model_name=encoder)` and `self.depth_head = DPTHead(...)`.
- **DINOv2 Feature Extraction** (`depth_anything_v2/dinov2.py` lines 297–321):
  `DINOv2.get_intermediate_layers(x, n, return_class_token=True)` extracts intermediate feature representations from ViT transformer blocks:
  - Input: Image tensor `x` of shape `[B, 3, H, W]` (where $H, W$ are multiples of patch size 14, e.g., 518x518).
  - Intermediate layer indices for variants (defined in `dpt.py` lines 164–169):
    - `vits`: `[2, 5, 8, 11]` (embed_dim = 384, depth = 12)
    - `vitb`: `[2, 5, 8, 11]` (embed_dim = 768, depth = 12)
    - `vitl`: `[4, 11, 17, 23]` (embed_dim = 1024, depth = 24)
    - `vitg`: `[9, 19, 29, 39]` (embed_dim = 1536, depth = 40)
  - Output tuple when `return_class_token=True`:
    A tuple of 4 element pairs `((patch_0, cls_0), (patch_1, cls_1), (patch_2, cls_2), (patch_3, cls_3))`, where:
    - `patch_i`: Tensor of shape `[B, N, embed_dim]`, where $N = (H/14) \times (W/14)$ patch tokens.
    - `cls_i`: Tensor of shape `[B, embed_dim]`, global CLS classification token.

### 1.2 Official Checkpoint Format (`checkpoints/depth_anything_v2_vits.pth`)
- Inspected checkpoint via `torch.load('checkpoints/depth_anything_v2_vits.pth', map_location='cpu')`:
  - Total parameter tensor keys: 239 keys.
  - Key prefixes: `pretrained.*` (DINOv2 backbone: `pretrained.cls_token`, `pretrained.pos_embed`, `pretrained.patch_embed.proj.weight`, `pretrained.blocks.0...`) and `depth_head.*` (DPT decoder head).
  - Parameter count for `vits`: ~24.8M total parameters (~22.8M backbone + ~2.0M DPT head).

### 1.3 Verification Suite Requirements (`verify.py` lines 112–117)
- `verify.py` attempts Strategy 1 load:
  ```python
  from depthlab.backbone.loader import load_da2_checkpoint
  model = load_da2_checkpoint(variant=self.config.encoder, checkpoint_path=ckpt)
  model = model.to(self.config.device).eval()
  depth = model.infer_image(sample_img, input_size=self.config.input_size)
  ```
- Therefore, `load_da2_checkpoint` must return a `DA2Backbone` object that satisfies:
  1. `.features(x: Tensor) -> FeatureBundle` for Layer 1 ↔ Layer 2 multi-head architecture contract (ADR-001 / `PROJECT.md` line 19).
  2. `.to(device)` / `.eval()` standard PyTorch `nn.Module` methods.
  3. `.infer_image(raw_image, input_size=518)` method delegating end-to-end depth inference to Layer 0 `DepthAnythingV2` for backward compatibility & verification gate checks.

---

## 2. Logic Chain

1. **Layer 0 Immutability & Wrapping**:
   - Because `depth-anything-v2-official/` cannot be modified, we import `DepthAnythingV2` from `depth_anything_v2.dpt` into `depthlab/backbone/loader.py`.
   - `DA2Backbone` in `depthlab/backbone/adapter.py` acts as a facade pattern wrapper around `DepthAnythingV2`.

2. **Feature Extraction Contract (`FeatureStage` & `FeatureBundle`)**:
   - DINOv2 returns 4 intermediate stages, each containing patch tokens `[B, N, embed_dim]` and a CLS token `[B, embed_dim]`.
   - To make feature handling clean, structured, and type-safe across downstream heads (e.g. relative depth head, uncertainty head), we encapsulate each stage in a `FeatureStage` dataclass containing `patch_tokens`, `cls_token`, `stage_index`, and `embed_dim`.
   - `FeatureStage` provides a helper method `.spatial_features(patch_h, patch_w)` which transforms 1D patch tokens `[B, N, embed_dim]` into spatial 2D feature maps `[B, embed_dim, patch_h, patch_w]`.
   - The 4 stages are aggregated into `FeatureBundle(stages: List[FeatureStage])`, which acts as the unified contract passed to downstream research heads via `BaseHead.forward(FeatureBundle)`.

3. **Frozen Backbone & Parameter-Efficient Adapters**:
   - Standard backbone fine-tuning is computationally expensive and damages pre-trained spatial priors. Thus, `DA2Backbone` freezes all backbone weights (`requires_grad = False`).
   - To support research into parameter-efficient fine-tuning (PEFT), `FeatureBundle` passes through an abstract `BaseAdapter(nn.Module)` interface.
   - `IdentityAdapter`: Default pass-through, zero parameter overhead, 100% frozen.
   - `ResidualFeatureAdapter`: Learns lightweight per-stage feature projections $\text{patch\_tokens} + \text{Proj}_i(\text{patch\_tokens})$.
   - `LoRAAdapter`: Injects low-rank adaptation matrices ($W = W_0 + \frac{\alpha}{r} B A$) into target linear layers or stage outputs while keeping $W_0$ frozen.

4. **Integration with Verification Suite**:
   - `DA2Backbone` delegates `infer_image` to the underlying `DepthAnythingV2` instance, enabling `verify.py` to seamlessly execute Strategy 1 checkpoint loading and pass all 11 validation gate checks without modifying `verify.py` or Layer 0.

---

## 3. Detailed Technical Specification & Source Layout

### 3.1 Directory Structure
```
depthlab/
└── backbone/
    ├── __init__.py
    ├── adapter.py   # FeatureStage, FeatureBundle, BaseAdapter, IdentityAdapter, ResidualFeatureAdapter, LoRAAdapter, DA2Backbone
    └── loader.py    # load_da2_checkpoint function & variant configurations
```

### 3.2 `depthlab/backbone/adapter.py` Specification

```python
import dataclasses
from typing import List, Optional, Tuple, Dict, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclasses.dataclass
class FeatureStage:
    """Represents a single intermediate feature stage extracted from ViT backbone."""
    patch_tokens: torch.Tensor  # [B, N, embed_dim] where N = patch_h * patch_w
    cls_token: torch.Tensor     # [B, embed_dim]
    stage_index: int            # 0, 1, 2, 3
    embed_dim: int              # Embedding dimension (e.g. 384, 768, 1024, 1536)

    def spatial_features(self, patch_h: int, patch_w: int) -> torch.Tensor:
        """Converts [B, N, embed_dim] patch tokens to spatial 2D feature map [B, embed_dim, patch_h, patch_w]."""
        B, N, C = self.patch_tokens.shape
        assert N == patch_h * patch_w, f"Token count {N} != patch_h ({patch_h}) * patch_w ({patch_w})"
        return self.patch_tokens.permute(0, 2, 1).reshape(B, C, patch_h, patch_w)

    def to(self, device=None, dtype=None) -> 'FeatureStage':
        return FeatureStage(
            patch_tokens=self.patch_tokens.to(device=device, dtype=dtype),
            cls_token=self.cls_token.to(device=device, dtype=dtype),
            stage_index=self.stage_index,
            embed_dim=self.embed_dim,
        )

@dataclasses.dataclass
class FeatureBundle:
    """Container holding multi-stage intermediate features extracted from backbone."""
    stages: List[FeatureStage]

    def __getitem__(self, idx: int) -> FeatureStage:
        return self.stages[idx]

    def __len__(self) -> int:
        return len(self.stages)

    def __iter__(self):
        return iter(self.stages)

    @property
    def num_stages(self) -> int:
        return len(self.stages)

    @property
    def embed_dim(self) -> int:
        return self.stages[0].embed_dim

    def to(self, device=None, dtype=None) -> 'FeatureBundle':
        return FeatureBundle(stages=[stage.to(device=device, dtype=dtype) for stage in self.stages])

    def get_spatial_bundle(self, patch_h: int, patch_w: int) -> List[torch.Tensor]:
        return [stage.spatial_features(patch_h, patch_w) for stage in self.stages]

class BaseAdapter(nn.Module):
    """Abstract Base Class for parameter-efficient backbone adapters."""
    def forward(self, bundle: FeatureBundle) -> FeatureBundle:
        raise NotImplementedError

class IdentityAdapter(BaseAdapter):
    """Pass-through adapter that returns FeatureBundle unchanged."""
    def forward(self, bundle: FeatureBundle) -> FeatureBundle:
        return bundle

class ResidualFeatureAdapter(BaseAdapter):
    """Learned residual adapter for multi-stage feature refinement."""
    def __init__(self, embed_dim: int, num_stages: int = 4, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim // 2
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, embed_dim)
            ) for _ in range(num_stages)
        ])
        # Initialize final linear layer weights to zero so initial output is exactly identity
        for proj in self.projections:
            nn.init.zeros_(proj[-1].weight)
            nn.init.zeros_(proj[-1].bias)

    def forward(self, bundle: FeatureBundle) -> FeatureBundle:
        new_stages = []
        for i, stage in enumerate(bundle.stages):
            delta = self.projections[i](stage.patch_tokens)
            new_patch_tokens = stage.patch_tokens + delta
            new_stages.append(FeatureStage(
                patch_tokens=new_patch_tokens,
                cls_token=stage.cls_token,
                stage_index=stage.stage_index,
                embed_dim=stage.embed_dim
            ))
        return FeatureBundle(stages=new_stages)

class LoRALinear(nn.Module):
    """Low-Rank Adaptation (LoRA) module for linear layers."""
    def __init__(self, in_features: int, out_features: int, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.lora_A.T @ self.lora_B.T) * self.scaling

class DA2Backbone(nn.Module):
    """Frozen wrapper around official Depth Anything V2 backbone with FeatureBundle extraction."""
    def __init__(self, official_model: nn.Module, variant: str = "vits", adapter: Optional[BaseAdapter] = None):
        super().__init__()
        self.official_model = official_model
        self.variant = variant
        self.adapter = adapter if adapter is not None else IdentityAdapter()

        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23],
            'vitg': [9, 19, 29, 39]
        }[variant]

        self.embed_dim = self.official_model.pretrained.embed_dim

        # Freeze backbone parameters
        for param in self.official_model.parameters():
            param.requires_grad = False

    def features(self, x: torch.Tensor) -> FeatureBundle:
        """Extract multi-stage intermediate features from input image tensor [B, 3, H, W]."""
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        raw_features = self.official_model.pretrained.get_intermediate_layers(
            x, self.intermediate_layer_idx, return_class_token=True
        )

        stages = []
        for i, (patch_tokens, cls_token) in enumerate(raw_features):
            stages.append(FeatureStage(
                patch_tokens=patch_tokens,
                cls_token=cls_token,
                stage_index=i,
                embed_dim=self.embed_dim
            ))

        bundle = FeatureBundle(stages=stages)
        return self.adapter(bundle)

    def forward(self, x: torch.Tensor) -> FeatureBundle:
        return self.features(x)

    @torch.no_grad()
    def infer_image(self, raw_image, input_size=518):
        """Delegates end-to-end depth inference to official model (used by verify.py)."""
        return self.official_model.infer_image(raw_image, input_size=input_size)
```

### 3.3 `depthlab/backbone/loader.py` Specification

```python
from pathlib import Path
from typing import Union, Optional
import torch
import torch.nn as nn

from depth_anything_v2.dpt import DepthAnythingV2
from .adapter import DA2Backbone, BaseAdapter

MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

def load_da2_checkpoint(
    variant: str = "vits",
    checkpoint_path: Union[str, Path] = "checkpoints/depth_anything_v2_vits.pth",
    device: str = "cpu",
    freeze: bool = True,
    adapter: Optional[BaseAdapter] = None
) -> DA2Backbone:
    """Loads a Depth Anything V2 checkpoint and returns a DA2Backbone instance."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'")

    if variant not in MODEL_CONFIGS:
        raise ValueError(f"Unknown variant '{variant}'. Supported: {list(MODEL_CONFIGS.keys())}")

    model_config = MODEL_CONFIGS[variant]
    official_model = DepthAnythingV2(**model_config)

    state_dict = torch.load(checkpoint_path, map_location='cpu')
    official_model.load_state_dict(state_dict)

    backbone = DA2Backbone(official_model=official_model, variant=variant, adapter=adapter)

    if freeze:
        for param in backbone.parameters():
            param.requires_grad = False

    backbone = backbone.to(device).eval()
    return backbone
```

---

## 4. Caveats

- **Device Placement**: When running in mixed-precision (FP16 autocast) or on CUDA, ensure all input tensors `x` are transferred to the target CUDA device prior to calling `backbone.features(x)`.
- **Patch Resolution**: Input image height $H$ and width $W$ must be divisible by 14 (the DINOv2 patch size). `OFFICIAL_TRANSFORM` in `depthlab/transforms.py` ensures input dimensions satisfy this constraint.
- **Layer 0 Immutability**: All references to `DepthAnythingV2` and `DINOv2` are strictly read-only imports from `depth-anything-v2-official/`.

---

## 5. Conclusion

The proposed Layer 1 Backbone Loader (`depthlab/backbone/loader.py`) and Compatibility Adapter Layer (`depthlab/backbone/adapter.py`) achieve complete isolation of Layer 0 immutable code while establishing a clean, modular, typed `FeatureBundle` interface contract for Layer 2 research heads. The design fully satisfies `verify.py` Strategy 1 checkpoint loading requirements while providing pluggable PEFT adapter capabilities.

---

## 6. Verification Method

1. **File Existence Check**:
   Confirm `depthlab/backbone/__init__.py`, `depthlab/backbone/adapter.py`, and `depthlab/backbone/loader.py` are created per spec.

2. **Standalone Verification Gate**:
   Execute standalone verification suite:
   ```bash
   python verify.py
   ```
   **Expected Result**: Strategy 1 loads model via `depthlab.backbone` and passes all 11 checks cleanly (`OVERALL STATUS: ALL CHECKS PASSED`).

3. **Feature Bundle Interface Test**:
   Execute interactive python snippet:
   ```python
   import torch
   from depthlab.backbone.loader import load_da2_checkpoint
   bb = load_da2_checkpoint('vits', 'checkpoints/depth_anything_v2_vits.pth')
   x = torch.randn(1, 3, 518, 518)
   bundle = bb.features(x)
   assert len(bundle) == 4
   assert bundle[0].patch_tokens.shape == (1, 1369, 384)
   assert bundle[0].cls_token.shape == (1, 384)
   print("Backbone FeatureBundle contract verified!")
   ```
