"""Composable promptable scale application. Model inference is supplied by callers."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch

from .scale_predictor import ScalePredictor


class PromptableMetricPipeline:
    def __init__(self, scale_predictor: ScalePredictor, inference: Callable[[np.ndarray], tuple[np.ndarray, torch.Tensor]]) -> None:
        self.scale_predictor = scale_predictor.eval()
        self.inference = inference

    @torch.no_grad()
    def compute_metric_depth(self, rgb: np.ndarray, click_x: Optional[int] = None,
                             click_y: Optional[int] = None) -> tuple[np.ndarray, float]:
        relative_depth, cls_token = self.inference(rgb)
        if click_x is None or click_y is None:
            return relative_depth, 1.0
        scale = float(self.scale_predictor(cls_token).reshape(-1)[0].cpu())
        return relative_depth * scale, scale
