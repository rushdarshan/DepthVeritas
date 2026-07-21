"""
depthlab/metrics/__init__.py — Metrics Suite & Dispatcher.
"""

from .depth_metrics import align_depth_scale_shift, compute_depth_metrics
from .dispatcher import MetricDispatcher, apply_feature_ablation

__all__ = [
    "align_depth_scale_shift",
    "compute_depth_metrics",
    "MetricDispatcher",
    "apply_feature_ablation",
]
