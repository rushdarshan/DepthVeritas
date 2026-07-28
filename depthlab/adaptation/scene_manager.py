from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn

from depthlab.adaptation.correction_net import CorrectionNet


class SceneManager:
    """Manages scene identity, reset logic, and inference with adapted weights.

    Usage:
        manager = SceneManager(backbone, correction_net, device)
        result = manager.adapt(keyframes, val_frames)
        depth = manager.infer(image)
        manager.reset()
    """

    def __init__(
        self,
        backbone: nn.Module,
        correction_net: CorrectionNet,
        device: torch.device,
    ) -> None:
        self.backbone = backbone
        self.initial_state = copy.deepcopy(correction_net.state_dict())
        self.correction_net = correction_net.to(device)
        self.device = device
        self.adapted_state: Optional[dict] = None
        self.adapted_scale: float = 1.0

    def reset(self) -> None:
        """Restore initial adapter weights and clear adapted state."""
        self.correction_net.load_state_dict(self.initial_state)
        self.correction_net.to(self.device)
        self.adapted_state = None
        self.adapted_scale = 1.0

    def infer(self, image: torch.Tensor) -> torch.Tensor:
        """Run DA2 + adapted correction net on a single image.

        Returns base depth if no adaptation has been performed.
        """
        self.correction_net.eval()
        self.backbone.eval()
        with torch.no_grad():
            bundle = self.backbone.features(image.unsqueeze(0))
            H, W = image.shape[-2:]
            patch_h, patch_w = H // 14, W // 14
            features = bundle.stages[-1].spatial_features(patch_h, patch_w)
            base_depth = self.backbone(image.unsqueeze(0))

            if self.adapted_state is not None:
                corrected, _, _ = self.correction_net(features, base_depth)
                return corrected * self.adapted_scale
            return base_depth

    def load_adapted(self, state_dict: dict, scale: float) -> None:
        self.correction_net.load_state_dict(state_dict)
        self.adapted_state = state_dict
        self.adapted_scale = scale



