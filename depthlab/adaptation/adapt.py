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
    sample_target_depth,
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
    patch_h = int(H // 14)
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
    """Run self-supervised scene adaptation on paired source/target frames.

    Each keyframe dict must have:
        features_src, base_depth_src, image_src     — source frame tensors
        features_tgt, base_depth_tgt, image_tgt     — target frame tensors
        intrinsics, transform                        — shared intrinsics + T_target_from_source

    Args:
        backbone: Frozen DA2 backbone (inference mode).
        correction_net: CorrectionNet (will be deep-copied to device).
        keyframes: Paired frames for training.
        val_frames: Held-out paired frames for best-checkpoint selection.
        device: Target device.
        num_steps: Max optimization steps.
        learning_rate: Adam learning rate.
        patience: Early stopping patience.
        photometric_weight: Weight for photometric loss (target vs warped source).
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
            kf["features_src"].unsqueeze(0).to(device),
            kf["base_depth_src"].unsqueeze(0).to(device),
            kf["image_src"].unsqueeze(0).to(device),
            kf["features_tgt"].unsqueeze(0).to(device),
            kf["base_depth_tgt"].unsqueeze(0).to(device),
            kf["image_tgt"].unsqueeze(0).to(device),
            kf["intrinsics"].unsqueeze(0).to(device),
            kf["transform"].unsqueeze(0).to(device),
        )

    def _paired_loss(
        feat_src: torch.Tensor, bd_src: torch.Tensor, img_src: torch.Tensor,
        feat_tgt: torch.Tensor, bd_tgt: torch.Tensor, img_tgt: torch.Tensor,
        K: torch.Tensor, T: torch.Tensor,
    ) -> torch.Tensor:
        corrected_src, _, _ = net(feat_src, bd_src)
        scaled_src = corrected_src * scale.clamp_min(0.1)
        pixels, z = reproject_depth(scaled_src.squeeze(1), K, T)
        H, W = scaled_src.shape[-2:]
        in_bounds = projected_coords_in_bounds(pixels, H, W)
        z_pos = positive_z_mask(z)
        valid = in_bounds & z_pos
        if not valid.any():
            return torch.tensor(float("nan"), device=device)
        photo = photo_loss_fn(img_tgt, img_src, pixels, mask=valid)

        # ponytail: use target base depth as geometric anchor
        depth_tgt = bd_tgt * scale.clamp_min(0.1)
        temp = temp_loss_fn(depth_tgt, z.unsqueeze(1), pixels, mask=valid)
        return photometric_weight * photo + temporal_weight * temp

    def _val_loss() -> torch.Tensor:
        net.eval()
        losses = []
        with torch.no_grad():
            for vf in val_frames:
                args = tuple(x.unsqueeze(0).to(device) for x in (
                    vf["features_src"], vf["base_depth_src"], vf["image_src"],
                    vf["features_tgt"], vf["base_depth_tgt"], vf["image_tgt"],
                    vf["intrinsics"], vf["transform"],
                ))
                loss = _paired_loss(*args)
                if torch.isfinite(loss):
                    losses.append(loss)
        net.train()
        if not losses:
            return torch.tensor(float("inf"), device=device)
        return torch.stack(losses).mean()

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
            args = tuple(x.unsqueeze(0).to(device) for x in (
                kf["features_src"], kf["base_depth_src"], kf["image_src"],
                kf["features_tgt"], kf["base_depth_tgt"], kf["image_tgt"],
                kf["intrinsics"], kf["transform"],
            ))
            pair_loss = _paired_loss(*args)
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
        step_metrics.append({"step": step, "train_loss": total_loss.item(), "val_loss": vloss.item() if torch.isfinite(vloss) else float("inf")})

        v = vloss.item() if torch.isfinite(vloss) else float("inf")
        if v < best_loss:
            best_loss = v
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
