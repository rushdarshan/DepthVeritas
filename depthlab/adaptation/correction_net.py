import torch
import torch.nn as nn
import torch.nn.functional as F


class CorrectionNet(nn.Module):
    """External correction network that predicts a log-depth residual from frozen DA2 features.

    Forward contract:
        features: [B, C, h, w] — spatial DA2 feature map at patch resolution
        base_depth: [B, 1, H, W] — zero-shot DA2 depth, must be positive finite

    Returns:
        corrected_depth: [B, 1, H, W]
        gate: [B, 1, H, W] — per-pixel confidence in [0, 1]
        residual: [B, 1, H, W] — tanh-bounded log-depth offset
    """

    def __init__(self, feature_dim: int = 384, hidden_dim: int = 64, max_params: int = 2_000_000):
        super().__init__()

        self.feature_proj = nn.Conv2d(feature_dim, hidden_dim, 1)
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim),
                nn.GroupNorm(8, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 1),
                nn.GroupNorm(8, hidden_dim),
            ) for _ in range(3)
        ])

        self.residual_head = nn.Conv2d(hidden_dim, 1, 1)
        self.gate_head = nn.Conv2d(hidden_dim, 1, 1)

        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.zeros_(self.gate_head.bias)

        n = sum(p.numel() for p in self.parameters())
        if n > max_params:
            raise ValueError(f"CorrectionNet has {n} params, exceeds limit {max_params}")

    def forward(
        self, features: torch.Tensor, base_depth: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not torch.isfinite(base_depth).all() or not (base_depth > 0).all():
            raise ValueError("base_depth must be positive and finite")
        B, _, H, W = base_depth.shape
        x = self.feature_proj(features)
        skip = x
        for block in self.res_blocks:
            x = block(x) + skip
            skip = x
        residual = self.residual_head(x)
        gate = self.gate_head(x).sigmoid()
        residual = residual.tanh()
        residual_up = F.interpolate(residual, (H, W), mode="bilinear", align_corners=False)
        gate_up = F.interpolate(gate, (H, W), mode="bilinear", align_corners=False)
        corrected = base_depth * torch.exp(gate_up * residual_up)
        return corrected, gate_up, residual_up
