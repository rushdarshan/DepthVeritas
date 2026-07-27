"""Precomputed-token dataset for CPU-fast promptable-scale training."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset


class PromptableScaleDataset(Dataset):
    def __init__(self, manifest_path: str | Path) -> None:
        with Path(manifest_path).open(encoding="utf-8", newline="") as handle:
            self.samples = list(csv.DictReader(handle))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        features = torch.from_numpy(np.load(sample["feature_path"]).astype("float32"))
        return {"features": features, "scale": torch.tensor(float(sample["s_star"])), "category": sample.get("category", "unknown")}
