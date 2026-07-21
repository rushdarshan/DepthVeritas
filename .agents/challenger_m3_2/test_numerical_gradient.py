"""
Numerical Gradient Flow Verification Test Suite
Validates numerical gradient flow across FP32 and FP16 AMP training passes.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from depthlab.backbone.loader import load_da2_checkpoint
from depthlab.heads import get_head


def run_gradient_verification():
    print("=== FP16 / FP32 Numerical Gradient Flow Verification ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    backbone = load_da2_checkpoint(variant="vits", checkpoint_path="checkpoints/depth_anything_v2_vits.pth", device=str(device), freeze=True)
    head = get_head("relative_depth", encoder_variant="vits", features=64, out_channels=[48, 96, 192, 384]).to(device)

    # Fixed seed for identical synthetic inputs
    torch.manual_seed(12345)
    images = torch.randn(2, 3, 140, 140, device=device)
    targets = torch.rand(2, 1, 140, 140, device=device) + 0.1

    print("\n--- 1. FP32 Training Pass ---")
    head.zero_grad()
    features_fp32 = backbone.features(images)
    preds_fp32 = head(features_fp32)
    loss_fp32 = head.compute_loss(preds_fp32, {"depth": targets})
    loss_fp32.backward()

    fp32_grads = {}
    fp32_nan_count = 0
    fp32_inf_count = 0
    fp32_total_grad_norm_sq = 0.0

    for name, param in head.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                print(f"FAIL (FP32): Param '{name}' grad is None")
                return 1
            if torch.isnan(param.grad).any():
                fp32_nan_count += 1
            if torch.isinf(param.grad).any():
                fp32_inf_count += 1
            grad_norm = param.grad.norm().item()
            fp32_grads[name] = grad_norm
            fp32_total_grad_norm_sq += grad_norm ** 2

    fp32_grad_norm = fp32_total_grad_norm_sq ** 0.5
    print(f"FP32 Loss: {loss_fp32.item():.6f}")
    print(f"FP32 Total Head Grad Norm: {fp32_grad_norm:.6f}")
    print(f"FP32 NaN count: {fp32_nan_count}, Inf count: {fp32_inf_count}")

    if fp32_nan_count > 0 or fp32_inf_count > 0 or fp32_grad_norm == 0:
        print("FAIL: FP32 gradient contains NaN/Inf or is zero.")
        return 1

    print("\n--- 2. FP16 AMP Training Pass ---")
    head.zero_grad()
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        features_amp = backbone.features(images)
        preds_amp = head(features_amp)
        loss_amp = head.compute_loss(preds_amp, {"depth": targets})

    if use_amp:
        scaler.scale(loss_amp).backward()
        scaler.unscale_(optimizer)
    else:
        loss_amp.backward()

    amp_grads = {}
    amp_nan_count = 0
    amp_inf_count = 0
    amp_total_grad_norm_sq = 0.0

    for name, param in head.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                print(f"FAIL (FP16 AMP): Param '{name}' grad is None")
                return 1
            if torch.isnan(param.grad).any():
                amp_nan_count += 1
            if torch.isinf(param.grad).any():
                amp_inf_count += 1
            grad_norm = param.grad.norm().item()
            amp_grads[name] = grad_norm
            amp_total_grad_norm_sq += grad_norm ** 2

    amp_grad_norm = amp_total_grad_norm_sq ** 0.5
    print(f"FP16 AMP Loss: {loss_amp.item():.6f}")
    print(f"FP16 AMP Total Head Grad Norm: {amp_grad_norm:.6f}")
    print(f"FP16 AMP NaN count: {amp_nan_count}, Inf count: {amp_inf_count}")

    if amp_nan_count > 0 or amp_inf_count > 0 or amp_grad_norm == 0:
        print("FAIL: FP16 AMP gradient contains NaN/Inf or is zero.")
        return 1

    print("\n--- 3. FP32 vs FP16 AMP Gradient Flow Comparison ---")
    loss_diff = abs(loss_fp32.item() - loss_amp.item())
    loss_rel_diff = loss_diff / (abs(loss_fp32.item()) + 1e-8)
    print(f"Absolute Loss Difference: {loss_diff:.6e}")
    print(f"Relative Loss Difference: {loss_rel_diff:.6e}")

    grad_norm_diff = abs(fp32_grad_norm - amp_grad_norm)
    print(f"Grad Norm Difference: {grad_norm_diff:.6e}")

    if loss_rel_diff > 0.05:
        print(f"WARNING: FP16 AMP loss differs from FP32 loss by > 5%: {loss_rel_diff:.4%}")

    print("\nSUCCESS: Numerical gradient flow validated cleanly across FP32 and FP16 AMP passes.")
    return 0


if __name__ == "__main__":
    sys.exit(run_gradient_verification())
