from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from depthlab.adaptation.correction_net import CorrectionNet
from depthlab.geometry import (
    reproject_depth,
    projected_coords_in_bounds,
    positive_z_mask,
    occlusion_mask,
    sample_target_depth,
    minimum_reprojection_mask,
)
from depthlab.losses.temporal import PhotometricConsistencyLoss, TemporalConsistencyLoss


@torch.no_grad()
def _extract_features_and_depth(
    backbone: nn.Module,
    images: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract frozen DA2 spatial features [B,384,h,w] and base depth [B,1,H,W]."""
    bundle = backbone.features(images)
    H, W = images.shape[-2:]
    patch_h = int(H // 14)  # DA2 ViT patch size
    patch_w = int(W // 14)
    features_4 = bundle.stages[-1].spatial_features(patch_h, patch_w)
    base_depth = backbone(images)
    return features_4, base_depth


def adapt_scene(
    backbone: nn.Module,
    correction_net: CorrectionNet,
    keyframes: list[dict],
    val_frames: list[dict],
    device: torch.device,
    num_steps: int = 100,
    learning_rate: float = 1e-3,
    patience: int = 15,
    photometric_weight: float = 1.0,
    temporal_weight: float = 0.5,
    output_dir: Optional[Path] = None,
) -> dict:
    """Run self-supervised scene adaptation.

    Args:
        backbone: Frozen DA2 backbone (inference mode).
        correction_net: CorrectionNet on CPU (will be moved to device).
        keyframes: List of dicts with 'features', 'base_depth', 'image', 'intrinsics', 'transform'.
        val_frames: Held-out validation frames (same schema) for best-checkpoint selection.
        device: Target GPU device.
        num_steps: Max optimization steps.
        learning_rate: Adam learning rate.
        patience: Early stopping patience.
        photometric_weight: Weight for photometric loss.
        temporal_weight: Weight for temporal depth consistency loss.
        output_dir: Optional path to save adapted weights.

    Returns:
        dict with 'adapted_state_dict', 'best_scale', 'step_metrics'.
    """
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    net = copy.deepcopy(correction_net).to(device).train()
    scale = nn.Parameter(torch.ones(1, device=device))
    optimizer = torch.optim.Adam(list(net.parameters()) + [scale], lr=learning_rate)

    photo_loss_fn = PhotometricConsistencyLoss()
    temp_loss_fn = TemporalConsistencyLoss()

    def _build_batch(kf: dict) -> tuple:
        return (
            kf["features"].unsqueeze(0).to(device),
            kf["base_depth"].unsqueeze(0).to(device),
            kf["image"].unsqueeze(0).to(device),
            kf["intrinsics"].unsqueeze(0).to(device),
            kf["transform"].unsqueeze(0).to(device),
        )

    def _val_loss() -> torch.Tensor:
        net.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for vf in val_frames:
                feat, bd, img, K, T = _build_batch(vf)
                corrected, _, _ = net(feat, bd)
                scaled_depth = corrected * scale.clamp_min(0.1)
                _, z = reproject_depth(scaled_depth.squeeze(1), K, T)
                in_bounds = projected_coords_in_bounds(
                    torch.zeros(1, 2, *scaled_depth.shape[-2:], device=device), *scaled_depth.shape[-2:]
                )
                if in_bounds.any():
                    total += (z[in_bounds[:, 0]]).abs().mean()
                    count += 1
        net.train()
        if count == 0:
            return torch.tensor(0.0, device=device)
        return total / count

    best_loss = float("inf")
    best_state = copy.deepcopy(net.state_dict())
    best_scale = 1.0
    steps_no_improve = 0
    step_metrics: list[dict] = []

    for step in range(num_steps):
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=device)
        n_pairs = 0

        for kf in keyframes:
            feat, bd, img, K, T = _build_batch(kf)
            corrected, gate, residual = net(feat, bd)
            scaled_depth = corrected * scale.clamp_min(0.1)
            pixels, z = reproject_depth(scaled_depth.squeeze(1), K, T)
            H, W = scaled_depth.shape[-2:]
            in_bounds = projected_coords_in_bounds(pixels, H, W)
            z_pos = positive_z_mask(z)
            valid_mask = in_bounds & z_pos

            # ponytail: occlusion mask omitted — no valid target-depth estimate
            # during self-supervised adaptation. Add when a geometry-validity
            # module with reliable target depth is available.

            if not valid_mask.any():
                continue

            photo_loss = photo_loss_fn(img, img, pixels, mask=valid_mask)
            temp_loss = temp_loss_fn(scaled_depth, z.unsqueeze(1), pixels, mask=valid_mask)
            pair_loss = photometric_weight * photo_loss + temporal_weight * temp_loss

            if not torch.isfinite(pair_loss):
                continue

            total_loss = total_loss + pair_loss
            n_pairs += 1

        if n_pairs == 0:
            step_metrics.append({"step": step, "loss": float("nan")})
            continue

        total_loss = total_loss / n_pairs
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()

        vloss = _val_loss()
        step_metrics.append({"step": step, "train_loss": total_loss.item(), "val_loss": vloss.item()})

        if vloss < best_loss:
            best_loss = vloss.item()
            best_state = copy.deepcopy(net.state_dict())
            best_scale = scale.item()
            steps_no_improve = 0
        else:
            steps_no_improve += 1
            if steps_no_improve >= patience:
                break

    net.load_state_dict(best_state)
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, output_dir / "adapted_state.pt")
        torch.save({"scale": best_scale}, output_dir / "adapted_scale.pt")

    return {
        "adapted_state_dict": best_state,
        "best_scale": best_scale,
        "step_metrics": step_metrics,
    }
