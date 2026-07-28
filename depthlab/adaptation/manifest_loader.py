from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset


class AdaptationManifest(Dataset):
    """Load scene adaptation frames with known poses and intrinsics.

    Manifest JSON schema per entry:
        image_path: str — path to RGB frame
        intrinsics: list[float] — 3x3 intrinsic matrix (row-major)
        transform: list[float] — 4x4 T_target_from_source pose (row-major)
        split: str — "adapt" or "test"

    Pose convention: T_target_from_source maps a point in source frame
    to target frame: x_target = T @ x_source.

    Missing intrinsics is always an error for v1 — no silent focal fallback.
    """

    def __init__(
        self,
        manifest: str | Path,
        split: str = "adapt",
        min_baseline: float = 0.05,
        check_baseline: bool | None = None,
        **_: Any,
    ) -> None:
        self.manifest = Path(manifest)
        self.split = split
        self.min_baseline = min_baseline
        self.check_baseline = min_baseline > 0 if check_baseline is None else check_baseline
        entries: List[Dict[str, Any]] = json.loads(self.manifest.read_text(encoding="utf-8"))
        self.entries = [e for e in entries if e.get("split", split) == split]
        if not self.entries:
            raise ValueError(f"No {split} entries in {self.manifest}")

    def __len__(self) -> int:
        return len(self.entries)

    @staticmethod
    def _load_image(path: Path) -> torch.Tensor:
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("Pillow is required for image loading") from exc
        arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    def _check_baseline(self, pose: torch.Tensor) -> float:
        trans = pose[:3, 3]
        return float(trans.norm(p=2))

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.entries[index]
        if "intrinsics" not in row:
            raise ValueError("Intrinsics required for v1 — no silent fallback")
        if "transform" not in row:
            raise ValueError("Pose (transform) required for v1")

        image = self._load_image(self.manifest.parent / row["image_path"])
        intrinsics = torch.tensor(row["intrinsics"], dtype=torch.float32).reshape(3, 3)
        pose = torch.tensor(row["transform"], dtype=torch.float32).reshape(4, 4)

        baseline = self._check_baseline(pose)
        if self.check_baseline and baseline < self.min_baseline:
            raise ValueError(
                f"Translation baseline {baseline:.4f} < min_baseline {self.min_baseline}. "
                "Tiny-baseline or pure-rotation pairs cannot constrain depth."
            )

        result: Dict[str, torch.Tensor] = {
            "image": image,
            "intrinsics": intrinsics,
            "transform": pose,
            "baseline": torch.tensor(baseline),
        }
        if "depth_path" in row:
            dp = self.manifest.parent / row["depth_path"]
            result["depth"] = torch.from_numpy(np.load(dp)).float()
        return result


def select_keyframes(
    dataset: AdaptationManifest,
    max_frames: int = 5,
) -> List[int]:
    """Select up to max_frames by highest translation baseline."""
    baselines = []
    for i in range(len(dataset)):
        row = dataset.entries[i]
        pose = torch.tensor(row["transform"], dtype=torch.float32).reshape(4, 4)
        bl = pose[:3, 3].norm(p=2).item()
        baselines.append((bl, i))
    baselines.sort(reverse=True)
    return [idx for _, idx in baselines[:max_frames]]
