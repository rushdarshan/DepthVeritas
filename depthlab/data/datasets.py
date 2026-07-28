"""Built-in dataset implementations.

These loaders keep the framework bootable without bundling large datasets. The
synthetic dataset is intended for smoke tests; file-list datasets support real
RGB/depth pairs through manifests.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from depthlab.data import register_dataset


def _resize_chw(tensor: torch.Tensor, size: int | tuple[int, int]) -> torch.Tensor:
    if isinstance(size, int):
        out_size = (size, size)
    else:
        out_size = size
    return F.interpolate(tensor.unsqueeze(0), size=out_size, mode="bilinear", align_corners=False).squeeze(0)


@register_dataset("synthetic")
class SyntheticDepthDataset(Dataset):
    """Deterministic synthetic RGB/depth samples for import and smoke checks."""

    def __init__(self, split: str = "train", num_samples: int = 16, image_size: int = 518, seed: int = 42, **_: Any):
        self.split = split
        self.num_samples = int(num_samples)
        self.image_size = int(image_size)
        self.seed = int(seed) + (0 if split == "train" else 10_000)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        gen = torch.Generator().manual_seed(self.seed + idx)
        h = w = self.image_size
        image = torch.rand((3, h, w), generator=gen)
        y = torch.linspace(0.5, 8.0, h).view(h, 1).expand(h, w)
        x = torch.linspace(0.0, 1.0, w).view(1, w).expand(h, w)
        depth = y + 0.25 * torch.sin(6.28318 * x)
        valid_mask = torch.isfinite(depth) & (depth > 0)
        return {"image": image, "depth": depth.float(), "valid_mask": valid_mask}


@register_dataset("file_list")
class FileListDepthDataset(Dataset):
    """Load RGB/depth pairs from a CSV manifest.

    Manifest columns: `split,image_path,depth_path`. Paths may be absolute or
    relative to the manifest directory. Depth files may be `.npy` arrays.
    """

    def __init__(self, manifest: str | Path, split: str = "train", image_size: Optional[int] = None, **_: Any):
        self.manifest = Path(manifest)
        self.split = split
        self.image_size = image_size
        self.rows = self._read_rows()

    def _resolve(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.manifest.parent / path

    def _read_rows(self) -> List[Dict[str, str]]:
        with open(self.manifest, "r", encoding="utf-8", newline="") as f:
            rows = [row for row in csv.DictReader(f) if row.get("split", self.split) == self.split]
        if not rows:
            raise ValueError(f"No rows for split '{self.split}' in {self.manifest}")
        return rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]
        image = self._load_image(self._resolve(row["image_path"]))
        depth = self._load_depth(self._resolve(row["depth_path"]))
        if self.image_size:
            image = _resize_chw(image, int(self.image_size))
            depth = _resize_chw(depth.unsqueeze(0), int(self.image_size)).squeeze(0)
        valid_mask = torch.isfinite(depth) & (depth > 0)
        return {"image": image, "depth": depth.float(), "valid_mask": valid_mask}

    @staticmethod
    def _load_depth(path: Path) -> torch.Tensor:
        if path.suffix.lower() != ".npy":
            raise ValueError(f"Only .npy depth maps are currently supported: {path}")
        return torch.from_numpy(np.load(path)).float()

    @staticmethod
    def _load_image(path: Path) -> torch.Tensor:
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("Pillow is required for file_list image loading.") from exc
        arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()
