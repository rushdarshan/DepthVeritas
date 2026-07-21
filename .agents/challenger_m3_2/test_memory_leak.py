"""
Memory Leak Verification Test Suite
Tracks Process RAM and GPU VRAM over 10 consecutive training epochs using synthetic data.
"""

import sys
import os
import gc
import psutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from depthlab.trainer import Trainer


def run_memory_leak_test(num_epochs: int = 10):
    print(f"=== Memory Leak Verification ({num_epochs} Epochs) ===")

    # Config for synthetic dataset training
    config = {
        "experiment": {
            "name": "mem_leak_test",
            "output_dir": "outputs/mem_leak_test",
            "seed": 42
        },
        "model": {
            "backbone": {
                "variant": "vits",
                "checkpoint_path": "checkpoints/depth_anything_v2_vits.pth",
                "frozen": True
            },
            "head": {
                "name": "relative_depth",
                "features": 64,
                "out_channels": [48, 96, 192, 384]
            }
        },
        "dataset": {
            "name": "synthetic",
            "num_samples": 32,
            "image_size": [140, 140],
            "batch_size": 4,
            "num_workers": 0
        },
        "training": {
            "epochs": num_epochs,
            "learning_rate": 1e-4,
            "weight_decay": 1e-2,
            "amp": True,
            "clip_grad_norm": 1.0,
            "save_interval": 100
        },
        "eval": {
            "align_scale_shift": True,
            "min_depth": 1e-3,
            "max_depth": 80.0
        }
    }

    process = psutil.Process(os.getpid())

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    trainer = Trainer(config)

    ram_history = []
    vram_alloc_history = []
    vram_peak_history = []

    print(f"{'Epoch':<6} | {'RAM (MB)':<12} | {'VRAM Alloc (MB)':<16} | {'VRAM Peak (MB)':<16}")
    print("-" * 58)

    # Initial baseline
    gc.collect()
    ram_base = process.memory_info().rss / (1024 ** 2)
    vram_alloc_base = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
    vram_peak_base = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0

    print(f"{'Base':<6} | {ram_base:<12.2f} | {vram_alloc_base:<16.2f} | {vram_peak_base:<16.2f}")

    for epoch in range(1, num_epochs + 1):
        trainer.train_epoch(epoch)
        trainer.evaluate(epoch)
        trainer.scheduler.step()

        gc.collect()
        ram = process.memory_info().rss / (1024 ** 2)
        vram_alloc = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
        vram_peak = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0

        ram_history.append(ram)
        vram_alloc_history.append(vram_alloc)
        vram_peak_history.append(vram_peak)

        print(f"{epoch:<6} | {ram:<12.2f} | {vram_alloc:<16.2f} | {vram_peak:<16.2f}")

    # Analysis
    ram_diff_total = ram_history[-1] - ram_history[0]
    ram_diff_late = ram_history[-1] - ram_history[2]  # Epoch 3 to 10 (steady state)
    vram_diff_total = vram_alloc_history[-1] - vram_alloc_history[0]

    print("\n--- Memory Leak Analysis ---")
    print(f"RAM Growth (Epoch 1 -> 10): {ram_diff_total:+.2f} MB")
    print(f"RAM Growth (Epoch 3 -> 10 steady-state): {ram_diff_late:+.2f} MB")
    if torch.cuda.is_available():
        print(f"VRAM Alloc Growth (Epoch 1 -> 10): {vram_diff_total:+.2f} MB")

    # Assertions
    success = True

    # Steady state RAM growth should not exceed 50 MB over 7 epochs
    if ram_diff_late > 50.0:
        print(f"FAIL: Significant RAM growth detected in steady state ({ram_diff_late:.2f} MB > 50.0 MB)")
        success = False

    if torch.cuda.is_available() and vram_diff_total > 10.0:
        print(f"FAIL: Significant VRAM growth detected ({vram_diff_total:.2f} MB > 10.0 MB)")
        success = False

    if success:
        print("SUCCESS: No memory leaks detected over 10 training epochs.")
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(run_memory_leak_test(10))
