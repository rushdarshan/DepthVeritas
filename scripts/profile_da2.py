"""Measure the safe DA2 inference envelope on the current GPU."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from depthlab.backbone.loader import load_da2_checkpoint
from depthlab.evidence import capability_matrix, write_evidence_bundle


def profile(model: torch.nn.Module, device: torch.device, size: int, repeats: int) -> dict[str, float | int | str]:
    if size % 14:
        return {"size": size, "status": "rejected", "reason": "DA2 input size must be divisible by patch size 14"}
    image = torch.rand(1, 3, size, size, device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        with torch.inference_mode():
            for _ in range(2):
                model(image)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            durations = []
            for _ in range(repeats):
                started = time.perf_counter()
                model(image)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                durations.append((time.perf_counter() - started) * 1000)
        return {"size": size, "status": "measured", "median_latency_ms": round(statistics.median(durations), 3), "peak_vram_mb": round(torch.cuda.max_memory_allocated(device) / 1024**2, 2) if device.type == "cuda" else 0.0}
    except torch.cuda.OutOfMemoryError:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return {"size": size, "status": "oom"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile a bounded DA2 GPU inference envelope")
    parser.add_argument("--sizes", type=int, nargs="+", default=[256, 392, 518])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "da2_operating_envelope.json")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_da2_checkpoint("vits", ROOT / "checkpoints" / "depth_anything_v2_vits.pth", str(device)).eval()
    results = [profile(model, device, size, args.repeats) for size in args.sizes]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"device": str(device), "results": results}, indent=2), encoding="utf-8")
    write_evidence_bundle(args.output.with_name(args.output.stem + "_evidence.json"), command=f"python scripts/profile_da2.py --sizes {' '.join(map(str, args.sizes))} --repeats {args.repeats}", tier="measured", inputs=[ROOT / "checkpoints" / "depth_anything_v2_vits.pth"], outputs=[args.output], capabilities=capability_matrix(ROOT))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
