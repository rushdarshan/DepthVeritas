"""Sparse, confidence-ranked depth candidates decoded from DA2 patch tokens."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from depthlab.heads import register_head
from depthlab.heads.base import BaseHead


def select_sparse_candidates(
    candidates: torch.Tensor, max_candidates: int = 256, radius: float = 0.04,
    confidence_threshold: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select confidence-ranked, spatially separated candidates.

    Candidates are ``[B, N, 5]`` as normalized ``x, y, depth, confidence,
    variance``. The returned mask makes abstention explicit when fewer than K
    candidates survive suppression.
    """
    if candidates.ndim != 3 or candidates.shape[-1] != 5:
        raise ValueError("candidates must have shape [B, N, 5]")
    if max_candidates < 1 or radius < 0 or not 0 <= confidence_threshold <= 1:
        raise ValueError("max_candidates must be positive, radius non-negative, and confidence_threshold in [0, 1]")
    selected = candidates.new_zeros((candidates.shape[0], max_candidates, 5))
    valid = torch.zeros(candidates.shape[:1] + (max_candidates,), device=candidates.device, dtype=torch.bool)
    for batch_index in range(candidates.shape[0]):
        order = torch.argsort(candidates[batch_index, :, 3], descending=True, stable=True)
        kept: list[int] = []
        for index in order.tolist():
            if candidates[batch_index, index, 3] < confidence_threshold:
                break
            point = candidates[batch_index, index, :2]
            if not kept or torch.linalg.vector_norm(candidates[batch_index, kept, :2] - point, dim=1).ge(radius).all():
                kept.append(index)
            if len(kept) == max_candidates:
                break
        if kept:
            count = len(kept)
            selected[batch_index, :count] = candidates[batch_index, kept]
            valid[batch_index, :count] = True
    return selected, valid


@register_head("sparse_depth")
class SparseDepthHead(BaseHead):
    """Feature-grid sparse depth head with deterministic inference selection."""

    def __init__(self, encoder_variant: str = "vits", hidden_dim: int = 128, max_candidates: int = 256, confidence_threshold: float = 0.0, **_: Any) -> None:
        super().__init__()
        channels = {"vits": 384, "vitb": 768, "vitl": 1024, "vitg": 1536}[encoder_variant]
        self.max_candidates = max_candidates
        self.confidence_threshold = confidence_threshold
        self.decoder = nn.Sequential(nn.Conv2d(channels, hidden_dim, 1), nn.GELU(), nn.Conv2d(hidden_dim, 5, 1))

    @staticmethod
    def _grid_shape(token_count: int) -> Tuple[int, int]:
        side = int(token_count**0.5)
        if side * side != token_count:
            raise ValueError("SparseDepthHead needs a square token grid; pass resized square DA2 inputs")
        return side, side

    def forward(self, feature_bundle: Any) -> Dict[str, torch.Tensor]:
        tokens = feature_bundle.stages[-1].patch_tokens
        batch, count, channels = tokens.shape
        height, width = self._grid_shape(count)
        raw = self.decoder(tokens.transpose(1, 2).reshape(batch, channels, height, width))
        offsets = raw[:, :2].tanh().permute(0, 2, 3, 1) * 0.5
        yy, xx = torch.meshgrid(
            torch.arange(height, device=tokens.device, dtype=tokens.dtype),
            torch.arange(width, device=tokens.device, dtype=tokens.dtype), indexing="ij",
        )
        centers = torch.stack(((xx + 0.5) / width, (yy + 0.5) / height), dim=-1)
        positions = (centers.unsqueeze(0) + offsets / torch.tensor((width, height), device=tokens.device, dtype=tokens.dtype)).clamp(0, 1)
        values = raw[:, 2:].permute(0, 2, 3, 1)
        candidates = torch.cat((positions, F.softplus(values[..., :1]), values[..., 1:2].sigmoid(), values[..., 2:].exp()), dim=-1).reshape(batch, count, 5)
        selected, selection_mask = select_sparse_candidates(candidates, self.max_candidates, confidence_threshold=self.confidence_threshold)
        return {"candidates": candidates, "selected": selected, "selection_mask": selection_mask}

    def compute_loss(self, preds: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        target = batch["depth"]
        if target.ndim == 3:
            target = target.unsqueeze(1)
        candidate_depth = preds["candidates"][..., 2]
        target_cells = F.interpolate(target.float(), size=(int(candidate_depth.shape[1] ** 0.5),) * 2, mode="nearest").flatten(1)
        valid = torch.isfinite(target_cells) & (target_cells > 0)
        return (candidate_depth[valid] - target_cells[valid]).abs().mean() if valid.any() else candidate_depth.sum() * 0

    def compute_metrics(self, preds: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        selected = preds["selected"]
        valid = preds["selection_mask"]
        return {"selected_count": float(valid.sum(dim=1).float().mean().item()), "mean_confidence": float(selected[..., 3][valid].mean().item()) if valid.any() else 0.0}
