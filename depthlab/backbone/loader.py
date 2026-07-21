"""
depthlab/backbone/loader.py — Checkpoint loader and DA2Backbone factory.
"""

import sys
from pathlib import Path
from typing import Union, Optional
import torch
import torch.nn as nn

# Ensure depth-anything-v2-official is importable
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
OFFICIAL_REPO_DIR = PROJECT_ROOT / "depth-anything-v2-official"
if OFFICIAL_REPO_DIR.exists() and str(OFFICIAL_REPO_DIR) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_REPO_DIR))

from depth_anything_v2.dpt import DepthAnythingV2
from .adapter import DA2Backbone, BaseAdapter


MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}


def load_da2_checkpoint(
    variant: str = "vits",
    checkpoint_path: Union[str, Path] = "checkpoints/depth_anything_v2_vits.pth",
    device: str = "cpu",
    freeze: bool = True,
    adapter: Optional[BaseAdapter] = None
) -> DA2Backbone:
    """Loads a Depth Anything V2 checkpoint and returns a DA2Backbone instance."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'")

    if variant not in MODEL_CONFIGS:
        raise ValueError(f"Unknown variant '{variant}'. Supported: {list(MODEL_CONFIGS.keys())}")

    model_config = MODEL_CONFIGS[variant]
    official_model = DepthAnythingV2(**model_config)

    state_dict = torch.load(checkpoint_path, map_location='cpu')
    official_model.load_state_dict(state_dict)

    backbone = DA2Backbone(official_model=official_model, variant=variant, adapter=adapter)

    if freeze:
        for param in backbone.parameters():
            param.requires_grad = False

    backbone = backbone.to(device).eval()
    return backbone
