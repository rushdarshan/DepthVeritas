"""Surface Existence Field depth head and uncertainty-aware loss interface."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from depthlab.heads import register_head
from depthlab.heads.base import BaseHead
from depthlab.losses.sef import SEFLoss


class SEFHead(nn.Module):
    """Predict a categorical depth distribution and image-conditioned bin centers."""

    def __init__(self, in_channels: int, n_bins: int = 96, hidden_dim: int = 256,
                 min_depth: float = 0.1, max_depth: float = 80.0) -> None:
        super().__init__()
        if n_bins < 1:
            raise ValueError("n_bins must be at least one")
        self.n_bins = n_bins
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.bin_widths = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(in_channels, hidden_dim),
            nn.GELU(), nn.Linear(hidden_dim, n_bins),
        )
        self.classifier = nn.Conv2d(in_channels, n_bins, kernel_size=1)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        if features.ndim != 4:
            raise ValueError("SEFHead expects features with shape [B, C, H, W]")
        widths = F.softmax(self.bin_widths(features), dim=-1)
        boundaries = torch.cat((torch.zeros_like(widths[:, :1]), widths.cumsum(dim=-1)), dim=-1)
        boundaries = self.min_depth + (self.max_depth - self.min_depth) * boundaries
        centers = (boundaries[:, :-1] + boundaries[:, 1:]) * 0.5
        logits = self.classifier(features)
        probabilities = F.softmax(logits, dim=1)
        depth = (probabilities * centers[:, :, None, None]).sum(dim=1)
        entropy_raw = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=1)
        entropy = entropy_raw / math.log(self.n_bins) if self.n_bins > 1 else torch.zeros_like(entropy_raw)
        return {
            "depth": depth,
            "entropy": entropy,
            "bin_probs": probabilities.permute(0, 2, 3, 1),
            "bin_centers": centers,
            "logits": logits.permute(0, 2, 3, 1),
        }


@register_head("sef")
class SEFDepthHead(BaseHead):
    """FeatureBundle adapter for :class:`SEFHead`.

    The final DINOv2 feature stage is reshaped to its patch grid. This retains a
    small, independent decoder that can be warmed up before backbone fine-tuning.
    """

    def __init__(self, encoder_variant: str = "vits", n_bins: int = 96,
                 hidden_dim: int = 256, min_depth: float = 0.1, max_depth: float = 80.0,
                 center_weight: float = 0.1, entropy_weight: float = 0.0, **_: Any) -> None:
        super().__init__()
        dims = {"vits": 384, "vitb": 768, "vitl": 1024, "vitg": 1536}
        if encoder_variant not in dims:
            raise ValueError(f"Unsupported encoder variant: {encoder_variant}")
        self.sef = SEFHead(dims[encoder_variant], n_bins, hidden_dim, min_depth, max_depth)
        self.loss_fn = SEFLoss(center_weight=center_weight, entropy_weight=entropy_weight)

    def forward(self, feature_bundle: Any) -> Dict[str, torch.Tensor]:
        tokens = feature_bundle.stages[-1].patch_tokens
        side = int(tokens.shape[1] ** 0.5)
        if side * side != tokens.shape[1]:
            raise ValueError("SEF requires a square patch grid")
        features = tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], side, side)
        return self.sef(features)

    def compute_loss(self, preds: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        target = batch["depth"]
        if target.shape[-2:] != preds["depth"].shape[-2:]:
            target = F.interpolate(target.unsqueeze(1), size=preds["depth"].shape[-2:], mode="nearest").squeeze(1)
            mask = batch.get("valid_mask")
            if mask is not None:
                batch = dict(batch)
                batch["valid_mask"] = F.interpolate(mask.float().unsqueeze(1), size=target.shape[-2:], mode="nearest").squeeze(1).bool()
        return self.loss_fn(preds, target, batch.get("valid_mask"))

    def compute_metrics(self, preds: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        from depthlab.metrics import compute_depth_metrics
        target = batch["depth"]
        prediction = preds["depth"]
        if target.shape[-2:] != prediction.shape[-2:]:
            target = F.interpolate(target.unsqueeze(1), size=prediction.shape[-2:], mode="nearest").squeeze(1)
        return compute_depth_metrics(prediction, target, align=False)
