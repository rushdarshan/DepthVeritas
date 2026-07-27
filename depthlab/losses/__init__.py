"""Training losses shared across DepthLab research heads."""

from .sef import SEFLoss
from .temporal import PhotometricConsistencyLoss, TemporalConsistencyLoss

__all__ = ["SEFLoss", "PhotometricConsistencyLoss", "TemporalConsistencyLoss"]
