"""
depthlab/heads/__init__.py — Head Registry & Dispatcher.
"""

from typing import Dict, Type, List, Any
from depthlab.heads.base import BaseHead

_HEAD_REGISTRY: Dict[str, Type[BaseHead]] = {}


def register_head(name: str):
    """
    Decorator to register a prediction head implementation in DepthLab.
    
    Usage:
        @register_head('relative_depth')
        class RelativeDepthHead(BaseHead):
            ...
    """
    def decorator(cls: Type[BaseHead]):
        if name in _HEAD_REGISTRY:
            # Allow re-registration of the exact same class (e.g. module reload)
            if _HEAD_REGISTRY[name] is cls:
                return cls
            raise ValueError(f"Head name '{name}' is already registered to {_HEAD_REGISTRY[name]}.")
        if not issubclass(cls, BaseHead):
            raise TypeError(f"Class '{cls.__name__}' must inherit from BaseHead to be registered.")
        _HEAD_REGISTRY[name] = cls
        return cls
    return decorator


def get_head(name: str, **kwargs: Any) -> BaseHead:
    """
    Factory function to instantiate a registered head by name.
    """
    if name not in _HEAD_REGISTRY:
        available = list_heads()
        raise KeyError(f"Head '{name}' not found in registry. Available heads: {available}")
    return _HEAD_REGISTRY[name](**kwargs)


def list_heads() -> List[str]:
    """
    Returns list of all registered head names.
    """
    return sorted(list(_HEAD_REGISTRY.keys()))


# Import head submodules to trigger registration
from depthlab.heads import relative_depth

__all__ = [
    "BaseHead",
    "register_head",
    "get_head",
    "list_heads",
]
