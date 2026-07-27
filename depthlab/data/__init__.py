"""Dataset registry and built-in lightweight datasets for DepthLab."""

from __future__ import annotations

from typing import Any, Dict, List, Type

from torch.utils.data import Dataset

_DATASET_REGISTRY: Dict[str, Type[Dataset]] = {}


def register_dataset(name: str):
    """Register a dataset class under a config-facing name."""
    def decorator(cls: Type[Dataset]) -> Type[Dataset]:
        if name in _DATASET_REGISTRY and _DATASET_REGISTRY[name] is not cls:
            raise ValueError(f"Dataset '{name}' is already registered.")
        _DATASET_REGISTRY[name] = cls
        return cls
    return decorator


def get_dataset(name: str, **kwargs: Any) -> Dataset:
    """Instantiate a registered dataset by name."""
    if name not in _DATASET_REGISTRY:
        raise KeyError(f"Dataset '{name}' not found. Available: {list_datasets()}")
    return _DATASET_REGISTRY[name](**kwargs)


def list_datasets() -> List[str]:
    return sorted(_DATASET_REGISTRY)


from depthlab.data import datasets, temporal  # noqa: E402,F401

__all__ = ["get_dataset", "list_datasets", "register_dataset"]
