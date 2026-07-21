"""
depthlab/heads/relative_depth.py — DINOv2 DPT Relative Depth Head & Loss Suite.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure depth-anything-v2-official is importable
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
OFFICIAL_REPO_DIR = PROJECT_ROOT / "depth-anything-v2-official"
if OFFICIAL_REPO_DIR.exists() and str(OFFICIAL_REPO_DIR) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_REPO_DIR))

from depth_anything_v2.dpt import DPTHead
from depthlab.heads.base import BaseHead
from depthlab.heads import register_head


class SILogLoss(nn.Module):
    """
    Scale-Invariant Logarithmic (SILog) Loss.
    """
    def __init__(self, lambd: float = 0.5, eps: float = 1e-6):
        super().__init__()
        self.lambd = lambd
        self.eps = eps

    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor, 
        valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if valid_mask is None:
            valid_mask = (target > 0) & (~torch.isnan(target)) & (~torch.isinf(target))
        else:
            valid_mask = valid_mask.detach()

        if pred.ndim == 4 and pred.shape[1] == 1:
            pred = pred.squeeze(1)
        if target.ndim == 4 and target.shape[1] == 1:
            target = target.squeeze(1)
        if valid_mask.ndim == 4 and valid_mask.shape[1] == 1:
            valid_mask = valid_mask.squeeze(1)

        pred_val = torch.clamp(pred[valid_mask], min=self.eps)
        target_val = torch.clamp(target[valid_mask], min=self.eps)

        if pred_val.numel() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        diff_log = torch.log(target_val) - torch.log(pred_val)
        loss = torch.sqrt(
            torch.pow(diff_log, 2).mean() - self.lambd * torch.pow(diff_log.mean(), 2) + 1e-8
        )
        return loss


def compute_relative_depth_metrics(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    valid_mask: Optional[torch.Tensor] = None,
    align_scale: bool = True
) -> Dict[str, float]:
    """
    Computes standard depth evaluation metrics: AbsRel, RMSE, d1, d2, d3.
    Optionally performs median scale alignment for relative depth predictions.
    """
    if pred.ndim == 4 and pred.shape[1] == 1:
        pred = pred.squeeze(1)
    if target.ndim == 4 and target.shape[1] == 1:
        target = target.squeeze(1)

    if valid_mask is None:
        valid_mask = (target > 0) & (~torch.isnan(target)) & (~torch.isinf(target))
    elif valid_mask.ndim == 4 and valid_mask.shape[1] == 1:
        valid_mask = valid_mask.squeeze(1)

    pred_val = pred[valid_mask]
    target_val = target[valid_mask]

    if pred_val.numel() == 0:
        return {'abs_rel': 0.0, 'rmse': 0.0, 'd1': 0.0, 'd2': 0.0, 'd3': 0.0}

    pred_val = torch.clamp(pred_val, min=1e-3)
    target_val = torch.clamp(target_val, min=1e-3)

    if align_scale:
        pred_med = torch.median(pred_val)
        target_med = torch.median(target_val)
        scale = target_med / (pred_med + 1e-8)
        pred_val = torch.clamp(pred_val * scale, min=1e-3)

    thresh = torch.max((target_val / pred_val), (pred_val / target_val))
    d1 = (thresh < 1.25).float().mean().item()
    d2 = (thresh < 1.25 ** 2).float().mean().item()
    d3 = (thresh < 1.25 ** 3).float().mean().item()

    diff = pred_val - target_val
    abs_rel = (torch.abs(diff) / target_val).mean().item()
    rmse = torch.sqrt(torch.pow(diff, 2).mean()).item()

    return {
        'abs_rel': abs_rel,
        'rmse': rmse,
        'd1': d1,
        'd2': d2,
        'd3': d3,
    }


@register_head('relative_depth')
class RelativeDepthHead(BaseHead):
    """
    DPT Decoder Head registered for Relative Depth Estimation.
    """
    def __init__(
        self,
        encoder_variant: str = 'vits',
        in_channels: int = 384,
        features: int = 64,
        out_channels: Optional[List[int]] = None,
        use_bn: bool = False,
        use_clstoken: bool = False,
        lambd: float = 0.5,
    ):
        super().__init__()
        
        variant_configs = {
            'vits': {'in_channels': 384, 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'in_channels': 768, 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'in_channels': 1024, 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'in_channels': 1536, 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        
        if encoder_variant in variant_configs:
            cfg = variant_configs[encoder_variant]
            in_channels = cfg['in_channels']
            features = cfg['features']
            out_channels = cfg['out_channels']
        elif out_channels is None:
            out_channels = [48, 96, 192, 384]

        self.dpt = DPTHead(
            in_channels=in_channels,
            features=features,
            use_bn=use_bn,
            out_channels=out_channels,
            use_clstoken=use_clstoken
        )
        self.loss_fn = SILogLoss(lambd=lambd)

    def forward(self, feature_bundle: Any) -> Dict[str, torch.Tensor]:
        """
        Extracts intermediate feature tensors from FeatureBundle and passes through DPT decoder.
        """
        out_features = []
        for stage in feature_bundle.stages:
            if self.dpt.use_clstoken and stage.cls_token is not None:
                out_features.append((stage.patch_tokens, stage.cls_token))
            else:
                out_features.append((stage.patch_tokens,))

        tokens = feature_bundle.stages[0].patch_tokens
        num_patches = tokens.shape[1]
        patch_h = int(num_patches ** 0.5)
        patch_w = patch_h

        depth = self.dpt(out_features, patch_h, patch_w)
        depth = F.relu(depth)
        depth = depth.squeeze(1)  # Shape: [B, H, W]

        return {"depth": depth, "predicted_depth": depth}

    def compute_loss(
        self, 
        preds: Dict[str, torch.Tensor], 
        batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        pred_depth = preds.get("depth", preds.get("predicted_depth"))
        target_depth = batch["depth"]
        valid_mask = batch.get("valid_mask", None)
        return self.loss_fn(pred_depth, target_depth, valid_mask)

    def compute_metrics(
        self, 
        preds: Dict[str, torch.Tensor], 
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        pred_depth = preds.get("depth", preds.get("predicted_depth"))
        target_depth = batch["depth"]
        valid_mask = batch.get("valid_mask", None)
        return compute_relative_depth_metrics(pred_depth, target_depth, valid_mask, align_scale=True)
