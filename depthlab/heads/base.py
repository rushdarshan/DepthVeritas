"""
depthlab/heads/base.py — Abstract Base Class for DepthLab Heads.
"""

import abc
from typing import Dict, Any
import torch
import torch.nn as nn


class BaseHead(nn.Module, abc.ABC):
    """
    Abstract Base Class for all prediction heads in DepthLab (Layer 2).
    
    All head implementations must inherit from BaseHead and implement:
      1. forward(feature_bundle) -> Dict[str, torch.Tensor]
      2. compute_loss(preds, batch) -> torch.Tensor
      3. compute_metrics(preds, batch) -> Dict[str, float]
    """

    @abc.abstractmethod
    def forward(self, feature_bundle: Any) -> Dict[str, torch.Tensor]:
        """
        Forward pass converting backbone FeatureBundle into output predictions.
        
        Args:
            feature_bundle: FeatureBundle object containing multi-stage backbone features.
            
        Returns:
            Dict containing predicted tensors, e.g. {'depth': Tensor[B, H, W]}.
        """
        pass

    @abc.abstractmethod
    def compute_loss(
        self, 
        preds: Dict[str, torch.Tensor], 
        batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Computes scalar training loss.
        
        Args:
            preds: Output dict from forward pass.
            batch: Data dictionary containing ground truth ('depth', 'valid_mask', etc.).
            
        Returns:
            Scalar Loss Tensor.
        """
        pass

    @abc.abstractmethod
    def compute_metrics(
        self, 
        preds: Dict[str, torch.Tensor], 
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Computes evaluation metrics.
        
        Args:
            preds: Output dict from forward pass.
            batch: Data dictionary containing ground truth.
            
        Returns:
            Dict mapping metric names to scalar float values.
        """
        pass
