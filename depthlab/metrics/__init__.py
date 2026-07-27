"""
depthlab/metrics/__init__.py — Metrics Suite & Dispatcher.
"""

from .depth_metrics import align_depth_scale_shift, compute_depth_metrics
from .dispatcher import MetricDispatcher, apply_feature_ablation
from .calibration import area_under_risk_coverage, expected_calibration_error, negative_log_likelihood, uncertainty_metrics
from .temporal import flow_warp, temporal_metrics

__all__ = [
    "align_depth_scale_shift",
    "compute_depth_metrics",
    "MetricDispatcher",
    "apply_feature_ablation",
    "area_under_risk_coverage",
    "expected_calibration_error",
    "negative_log_likelihood",
    "uncertainty_metrics",
    "flow_warp",
    "temporal_metrics",
]
