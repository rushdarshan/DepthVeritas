"""Manifest-driven video clips for reprojection and temporal consistency training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from depthlab.data.datasets import FileListDepthDataset
from depthlab.data import register_dataset


@register_dataset("video_manifest")
class VideoManifestDataset(Dataset):
    """Load paired RGB frames, intrinsics, optional poses/flow, and optional GT depth."""
    def __init__(self, manifest: str | Path, split: str = "train", **_: Any) -> None:
        self.manifest = Path(manifest)
        entries: List[Dict[str, Any]] = json.loads(self.manifest.read_text(encoding="utf-8"))
        self.entries = [entry for entry in entries if entry.get("split", split) == split]
        if not self.entries:
            raise ValueError(f"No {split} video entries in {self.manifest}")

    def __len__(self) -> int:
        return len(self.entries)

    def _image(self, value: str) -> torch.Tensor:
        return FileListDepthDataset._load_image(self.manifest.parent / value)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.entries[index]
        result: Dict[str, torch.Tensor] = {"image": self._image(row["image_path"]), "next_image": self._image(row["next_image_path"]),
                                           "intrinsics": torch.tensor(row["intrinsics"], dtype=torch.float32),
                                           "transform": torch.tensor(row.get("transform", np.eye(4)), dtype=torch.float32)}
        for key in ("depth_path", "next_depth_path", "flow_path"):
            if key in row:
                result[key.removesuffix("_path")] = torch.from_numpy(np.load(self.manifest.parent / row[key])).float()
        return result
