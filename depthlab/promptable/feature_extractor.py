"""O(1) extraction of click-local DINOv2 tokens from cached model outputs."""

from __future__ import annotations

from typing import Dict

import torch


def get_click_features(tokens: torch.Tensor, x: int, y: int, image_width: int,
                       image_height: int, patch_size: int = 14, include_patch: bool = False) -> Dict[str, torch.Tensor]:
    """Return CLS and the clicked patch feature from [B, 1+N, C] transformer tokens."""
    if tokens.ndim != 3 or tokens.shape[1] < 2:
        raise ValueError("tokens must have shape [B, 1 + patch_count, C]")
    grid_width = max(image_width // patch_size, 1)
    grid_height = max(image_height // patch_size, 1)
    col = min(max(x // patch_size, 0), grid_width - 1)
    row = min(max(y // patch_size, 0), grid_height - 1)
    index = min(1 + row * grid_width + col, tokens.shape[1] - 1)
    cls_token, patch_token = tokens[:, 0], tokens[:, index]
    combined = torch.cat((cls_token, patch_token), dim=-1) if include_patch else cls_token
    return {"features": combined, "cls_token": cls_token, "patch_token": patch_token, "row": torch.tensor(row), "col": torch.tensor(col)}
