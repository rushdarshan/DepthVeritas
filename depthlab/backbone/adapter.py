"""
depthlab/backbone/adapter.py — Feature representations, Adapters, and DA2Backbone wrapper.
"""

import dataclasses
from typing import List, Optional, Tuple, Dict, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclasses.dataclass
class FeatureStage:
    """Represents a single intermediate feature stage extracted from ViT backbone."""
    patch_tokens: torch.Tensor  # [B, N, embed_dim] where N = patch_h * patch_w
    cls_token: Optional[torch.Tensor]  # [B, embed_dim]
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
            cls_token=self.cls_token.to(device=device, dtype=dtype) if self.cls_token is not None else None,
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
        return self.stages[0].embed_dim if self.stages else 0

    def to(self, device=None, dtype=None) -> 'FeatureBundle':
        return FeatureBundle(stages=[stage.to(device=device, dtype=dtype) for stage in self.stages])

    def get_spatial_bundle(self, patch_h: int, patch_w: int) -> List[torch.Tensor]:
        return [stage.spatial_features(patch_h, patch_w) for stage in self.stages]

    def ablate_stage(self, stage_idx: int) -> 'FeatureBundle':
        """Returns a new FeatureBundle with the specified stage_idx zeroed out."""
        new_stages = []
        for stage in self.stages:
            if stage.stage_index == stage_idx:
                zero_patch = torch.zeros_like(stage.patch_tokens)
                zero_cls = torch.zeros_like(stage.cls_token) if stage.cls_token is not None else None
                new_stages.append(FeatureStage(
                    patch_tokens=zero_patch,
                    cls_token=zero_cls,
                    stage_index=stage.stage_index,
                    embed_dim=stage.embed_dim
                ))
            else:
                new_stages.append(stage)
        return FeatureBundle(stages=new_stages)


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
        hidden_dim = hidden_dim or max(embed_dim // 2, 64)
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, embed_dim)
            ) for _ in range(num_stages)
        ])
        # Initialize final linear layer weights to zero so initial output is identity
        for proj in self.projections:
            nn.init.zeros_(proj[-1].weight)
            nn.init.zeros_(proj[-1].bias)

    def forward(self, bundle: FeatureBundle) -> FeatureBundle:
        new_stages = []
        for i, stage in enumerate(bundle.stages):
            if i < len(self.projections):
                delta = self.projections[i](stage.patch_tokens)
                new_patch_tokens = stage.patch_tokens + delta
            else:
                new_patch_tokens = stage.patch_tokens

            new_stages.append(FeatureStage(
                patch_tokens=new_patch_tokens,
                cls_token=stage.cls_token,
                stage_index=stage.stage_index,
                embed_dim=stage.embed_dim
            ))
        return FeatureBundle(stages=new_stages)


class LoRALinear(nn.Module):
    """Low-Rank Adaptation (LoRA) module for linear projections."""
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


class LoRAAdapter(BaseAdapter):
    """LoRA feature adapter applied across FeatureBundle stages."""
    def __init__(self, embed_dim: int, num_stages: int = 4, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.lora_layers = nn.ModuleList([
            LoRALinear(embed_dim, embed_dim, rank=rank, alpha=alpha) for _ in range(num_stages)
        ])

    def forward(self, bundle: FeatureBundle) -> FeatureBundle:
        new_stages = []
        for i, stage in enumerate(bundle.stages):
            if i < len(self.lora_layers):
                delta = self.lora_layers[i](stage.patch_tokens)
                new_patch_tokens = stage.patch_tokens + delta
            else:
                new_patch_tokens = stage.patch_tokens

            new_stages.append(FeatureStage(
                patch_tokens=new_patch_tokens,
                cls_token=stage.cls_token,
                stage_index=stage.stage_index,
                embed_dim=stage.embed_dim
            ))
        return FeatureBundle(stages=new_stages)


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

        # Freeze official backbone parameters
        for param in self.official_model.parameters():
            param.requires_grad = False

    def features(self, x: torch.Tensor) -> FeatureBundle:
        """Extract multi-stage intermediate features from input image tensor [B, 3, H, W]."""
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
