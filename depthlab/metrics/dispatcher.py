"""
depthlab/metrics/dispatcher.py — Metric Dispatcher and Feature Ablation Hook.
"""

from typing import Dict, List, Optional, Union, Any
import torch
from depthlab.metrics.depth_metrics import compute_depth_metrics


def apply_feature_ablation(
    feature_bundle: Any,
    ablate_layers: Optional[List[int]] = None
) -> Any:
    """
    Zeroes out feature representations for specified stage indices in FeatureBundle.

    Args:
        feature_bundle: Layer 1 FeatureBundle containing list of FeatureStage objects
        ablate_layers: List of stage indices (e.g. [0, 1]) to zero out

    Returns:
        A new FeatureBundle instance with specified stages ablated.
    """
    if not ablate_layers:
        return feature_bundle

    if hasattr(feature_bundle, "ablate_stage"):
        bundle = feature_bundle
        for idx in ablate_layers:
            bundle = bundle.ablate_stage(idx)
        return bundle

    from depthlab.backbone.adapter import FeatureBundle, FeatureStage
    new_stages = []
    for stage in feature_bundle.stages:
        if stage.stage_index in ablate_layers:
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


class MetricDispatcher:
    """
    Batched metric accumulator and reporter for evaluation loops.
    """
    def __init__(
        self,
        align_scale_shift: bool = True,
        min_depth: float = 1e-3,
        max_depth: float = 80.0
    ):
        self.align_scale_shift = align_scale_shift
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.reset()

    def reset(self):
        """Resets all accumulated metric statistics."""
        self.accumulated_metrics: Dict[str, float] = {}
        self.num_batches: int = 0

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Computes metrics for a batch and updates running totals.
        """
        metrics = compute_depth_metrics(
            pred=pred,
            target=target,
            mask=mask,
            align=self.align_scale_shift,
            min_depth=self.min_depth,
            max_depth=self.max_depth
        )

        for k, v in metrics.items():
            self.accumulated_metrics[k] = self.accumulated_metrics.get(k, 0.0) + v
        self.num_batches += 1
        return metrics

    def compute(self) -> Dict[str, float]:
        """
        Returns average metric values across all accumulated batches.
        """
        if self.num_batches == 0:
            return {}
        return {k: v / self.num_batches for k, v in self.accumulated_metrics.items()}

    def summary_table(self) -> str:
        """Generates a formatted text table summarizing evaluated metrics."""
        results = self.compute()
        if not results:
            return "No metrics accumulated."

        lines = [
            "+------------+------------+",
            "| Metric     | Value      |",
            "+------------+------------+",
        ]
        for k, v in results.items():
            lines.append(f"| {k:<10} | {v:10.4f} |")
        lines.append("+------------+------------+")
        return "\n".join(lines)
