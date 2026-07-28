"""Compact uncertainty head used by the refinement and calibration tracks."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from depthlab.heads import register_head
from depthlab.heads.base import BaseHead


@register_head("uncertainty")
class UncertaintyHead(BaseHead):
    """Predict a depth residual and aleatoric log-variance from final ViT tokens."""

    def __init__(self, encoder_variant: str = "vits", hidden_dim: int = 128, **_: Any) -> None:
        super().__init__()
        channels = {"vits": 384, "vitb": 768, "vitl": 1024, "vitg": 1536}[encoder_variant]
        self.decoder = nn.Sequential(nn.Conv2d(channels, hidden_dim, 1), nn.GELU(), nn.Conv2d(hidden_dim, 2, 1))

    def forward(self, feature_bundle: Any) -> Dict[str, torch.Tensor]:
        tokens = feature_bundle.stages[-1].patch_tokens
        side = int(tokens.shape[1] ** 0.5)
        output = self.decoder(tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], side, side))
        return {"depth": F.softplus(output[:, 0]), "log_variance": output[:, 1], "uncertainty": output[:, 1].exp()}

    def compute_loss(self, preds: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        target = batch["depth"]
        if target.shape[-2:] != preds["depth"].shape[-2:]:
            target = F.interpolate(target.unsqueeze(1), size=preds["depth"].shape[-2:], mode="nearest").squeeze(1)
        mask = batch.get("valid_mask", torch.isfinite(target) & (target > 0))
        error = (preds["depth"] - target).abs()
        return (error * (-preds["log_variance"]).exp() + preds["log_variance"])[mask].mean() if mask.any() else preds["depth"].sum() * 0

    def compute_metrics(self, preds: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        from depthlab.metrics import compute_depth_metrics
        target = batch["depth"]
        if target.shape[-2:] != preds["depth"].shape[-2:]:
            target = F.interpolate(target.unsqueeze(1), size=preds["depth"].shape[-2:], mode="nearest").squeeze(1)
        return compute_depth_metrics(preds["depth"], target)
