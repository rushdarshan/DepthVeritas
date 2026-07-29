"""Run a deterministic, explicitly local NYU perturbation study for DA2."""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from depthlab.backbone.loader import load_da2_checkpoint
from depthlab.data import get_dataset
from depthlab.data.transforms import normalize_image_tensor
from depthlab.evidence import capability_matrix, write_evidence_bundle
from depthlab.metrics import compute_depth_metrics


def _resize_depth(depth: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(depth.unsqueeze(0).unsqueeze(0), size=size, mode="bilinear", align_corners=False)[0, 0]


def apply_perturbation(name: str, image: torch.Tensor, depth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RGB-only corruptions except crop, which preserves RGB/depth alignment."""
    if name == "baseline":
        return image, depth
    if name == "darken":
        return (image * 0.35).clamp(0, 1), depth
    if name == "blur":
        return F.avg_pool2d(image.unsqueeze(0), kernel_size=7, stride=1, padding=3)[0], depth
    if name == "jpeg":
        array = (image.permute(1, 2, 0).numpy() * 255).round().astype("uint8")
        buffer = io.BytesIO()
        Image.fromarray(array).save(buffer, format="JPEG", quality=25)
        restored = torch.from_numpy(np.asarray(Image.open(io.BytesIO(buffer.getvalue())).convert("RGB"), dtype=np.float32) / 255).permute(2, 0, 1)
        return restored, depth
    if name == "center_crop":
        height, width = depth.shape
        crop_height, crop_width = int(height * 0.75), int(width * 0.75)
        top, left = (height - crop_height) // 2, (width - crop_width) // 2
        cropped_image = image[:, top : top + crop_height, left : left + crop_width]
        cropped_depth = depth[top : top + crop_height, left : left + crop_width]
        return F.interpolate(cropped_image.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False)[0], _resize_depth(cropped_depth, (height, width))
    raise ValueError(f"Unknown perturbation: {name}")


def _mean(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: round(sum(row[key] for row in rows) / len(rows), 6) for key in rows[0]} if rows else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Local NYU perturbation study; not a public benchmark")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "depth_manifest.csv")
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "nyu_failure_studio.json")
    args = parser.parse_args()
    if args.limit < 1:
        raise ValueError("limit must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_da2_checkpoint("vits", ROOT / "checkpoints" / "depth_anything_v2_vits.pth", str(device)).eval()
    dataset = get_dataset("file_list", manifest=args.manifest, split=args.split, image_size=392)
    perturbations = ("baseline", "darken", "blur", "jpeg", "center_crop")
    results: dict[str, list[dict[str, float]]] = {name: [] for name in perturbations}
    with torch.inference_mode():
        for index in range(min(args.limit, len(dataset))):
            sample = dataset[index]
            for name in perturbations:
                image, target = apply_perturbation(name, sample["image"], sample["depth"])
                prediction = model(normalize_image_tensor(image.unsqueeze(0).to(device)))[0].cpu()
                results[name].append(compute_depth_metrics(prediction, target, align=True))
    report = {
        "evidence_tier": "measured",
        "scope": "Local NYU RGB-D perturbation study only; not the six-stratum failure-aware benchmark.",
        "dataset": {"manifest": str(args.manifest), "split": args.split, "samples": min(args.limit, len(dataset))},
        "perturbations": {name: _mean(rows) for name, rows in results.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_evidence_bundle(args.output.with_name(args.output.stem + "_evidence.json"), command=f"python scripts/nyu_failure_studio.py --split {args.split} --limit {args.limit}", tier="measured", inputs=[args.manifest, ROOT / "checkpoints" / "depth_anything_v2_vits.pth"], outputs=[args.output], capabilities=capability_matrix(ROOT), extra={"scope": report["scope"]})
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
