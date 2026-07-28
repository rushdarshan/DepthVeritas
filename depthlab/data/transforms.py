"""Shared image preprocessing helpers.

`OFFICIAL_TRANSFORM` is exposed as the single import location required by the
architecture docs. When the official DA2 transform utilities are unavailable,
callers can still use `normalize_image_tensor` for tensor datasets.
"""

from __future__ import annotations

import torch

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

OFFICIAL_TRANSFORM = None


def normalize_image_tensor(image: torch.Tensor) -> torch.Tensor:
    """Normalize a CHW RGB tensor in [0, 1] using ImageNet statistics."""
    mean = IMAGENET_MEAN.to(device=image.device, dtype=image.dtype)
    std = IMAGENET_STD.to(device=image.device, dtype=image.dtype)
    return (image - mean) / std
