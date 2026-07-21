#!/usr/bin/env python3
"""
download_checkpoints.py — Utility to download Depth Anything V2 model checkpoints.

Downloads official pretrained weights from HuggingFace into `checkpoints/`.
Default variant: `depth_anything_v2_vits.pth` (Depth Anything V2 Small).
"""

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

CHECKPOINT_URLS = {
    "vits": "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth",
    "vitb": "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth",
    "vitl": "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth",
}

EXPECTED_FILE_SIZES = {
    "vits": 99000000,  # ~99 MB
}


def compute_sha256(filepath: Path) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def download_checkpoint(variant: str = "vits", force: bool = False) -> Path:
    if variant not in CHECKPOINT_URLS:
        raise ValueError(f"Unknown variant '{variant}'. Available: {list(CHECKPOINT_URLS.keys())}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    target_file = CHECKPOINT_DIR / f"depth_anything_v2_{variant}.pth"
    url = CHECKPOINT_URLS[variant]

    if target_file.exists() and not force:
        size = target_file.stat().st_size
        print(f"[CHECKPOINT] Found existing checkpoint '{target_file}' ({size / (1024*1024):.2f} MB).")
        sha256_hash = compute_sha256(target_file)
        print(f"[CHECKPOINT] SHA256: {sha256_hash}")
        return target_file

    print(f"[CHECKPOINT] Downloading {variant} checkpoint from {url}...")
    print(f"[CHECKPOINT] Saving to {target_file}")

    def progress_callback(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100.0, downloaded * 100.0 / total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\rDownloading: {percent:5.1f}% ({mb_downloaded:6.2f} / {mb_total:6.2f} MB)")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, target_file, reporthook=progress_callback)
        print("\n[CHECKPOINT] Download completed successfully.")
    except Exception as e:
        if target_file.exists():
            target_file.unlink()
        print(f"\n[CHECKPOINT] Error downloading checkpoint: {e}", file=sys.stderr)
        raise

    size = target_file.stat().st_size
    sha256_hash = compute_sha256(target_file)
    print(f"[CHECKPOINT] Downloaded file size: {size / (1024*1024):.2f} MB")
    print(f"[CHECKPOINT] SHA256: {sha256_hash}")
    return target_file


def main():
    parser = argparse.ArgumentParser(description="Download Depth Anything V2 Checkpoints")
    parser.add_argument("--variant", type=str, default="vits", choices=["vits", "vitb", "vitl"], help="Model variant")
    parser.add_argument("--force", action="store_true", help="Force redownload even if file exists")
    args = parser.parse_args()

    download_checkpoint(variant=args.variant, force=args.force)


if __name__ == "__main__":
    main()
