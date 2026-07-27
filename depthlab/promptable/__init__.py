"""Promptable metric-depth scale prediction from a reference click."""

from .feature_extractor import get_click_features
from .scale_predictor import ScalePredictor

__all__ = ["get_click_features", "ScalePredictor"]
