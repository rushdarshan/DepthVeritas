"""
Layer 1 Backbone Loader and Compatibility Adapter Layer.
"""

from .adapter import (
    FeatureStage,
    FeatureBundle,
    BaseAdapter,
    IdentityAdapter,
    ResidualFeatureAdapter,
    LoRAAdapter,
    DA2Backbone,
)


def load_da2_checkpoint(*args, **kwargs):
    """Lazily import the optional upstream DA2 loader on first model use."""
    from .loader import load_da2_checkpoint as _load_da2_checkpoint

    return _load_da2_checkpoint(*args, **kwargs)

__all__ = [
    "FeatureStage",
    "FeatureBundle",
    "BaseAdapter",
    "IdentityAdapter",
    "ResidualFeatureAdapter",
    "LoRAAdapter",
    "DA2Backbone",
    "load_da2_checkpoint",
]
