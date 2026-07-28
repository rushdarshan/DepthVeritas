#!/usr/bin/env python3
"""Evaluate self-adapting depth: zero-shot AbsRel vs adapted AbsRel + wall time + peak VRAM."""

import argparse
import json
import time
from pathlib import Path

import torch

from depthlab.adaptation.correction_net import CorrectionNet
from depthlab.adaptation.manifest_loader import AdaptationManifest, select_keyframes
from depthlab.adaptation.adapt import _extract_features_and_depth, adapt_scene
from depthlab.adaptation.scene_manager import SceneManager
from depthlab.metrics.depth_metrics import compute_depth_metrics


def _prepare_paired(
    backbone: torch.nn.Module, item: dict, device: torch.device,
) -> dict:
    """Extract frozen features and depth for a paired manifest entry, return keyframe dict."""
    src_img = item["source_image"].to(device)
    tgt_img = item["target_image"].to(device)
    feats_src, bd_src = _extract_features_and_depth(backbone, src_img.unsqueeze(0))
    feats_tgt, bd_tgt = _extract_features_and_depth(backbone, tgt_img.unsqueeze(0))
    return {
        "features_src": feats_src[0], "base_depth_src": bd_src[0], "image_src": src_img,
        "features_tgt": feats_tgt[0], "base_depth_tgt": bd_tgt[0], "image_tgt": tgt_img,
        "intrinsics": item["intrinsics"], "transform": item["transform"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--backbone-checkpoint", type=Path, required=True)
    parser.add_argument("--encoder", default="vits")
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    from depthlab.backbone.loader import load_da2_checkpoint
    backbone = load_da2_checkpoint(variant=args.encoder, checkpoint_path=args.backbone_checkpoint, device=args.device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    device = torch.device(args.device)
    adapt_set = AdaptationManifest(args.manifest, split="adapt")
    test_set = AdaptationManifest(args.manifest, split="test", check_baseline=False)
    kf_indices = select_keyframes(adapt_set, max_frames=5)

    keyframes = [_prepare_paired(backbone, adapt_set[i], device) for i in kf_indices]
    val_frames = [_prepare_paired(backbone, test_set[i], device) for i in range(min(2, len(test_set)))]

    correction_net = CorrectionNet().to("cpu")

    if torch.cuda.is_available() and args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    t0 = time.perf_counter()
    result = adapt_scene(backbone, correction_net, keyframes, val_frames,
                         device=device, num_steps=args.num_steps,
                         output_dir=args.output)
    t_adapt = time.perf_counter() - t0

    peak_vram = 0.0
    if torch.cuda.is_available() and args.device == "cuda":
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)

    sm = SceneManager(backbone, correction_net, device)
    sm.initial_state = correction_net.state_dict()
    sm.load_adapted(result["adapted_state_dict"], result["best_scale"])

    abs_rel_before = []
    abs_rel_after = []
    for idx in range(len(test_set)):
        item = test_set[idx]
        img = item["source_image"].to(device)
        gt = item.get("depth")
        if gt is None:
            continue
        gt = gt.to(device)

        bundle = backbone.features(img.unsqueeze(0))
        H, W = img.shape[-2:]
        patch_h, patch_w = H // 14, W // 14
        features = bundle.stages[-1].spatial_features(patch_h, patch_w)
        base_depth = backbone(img.unsqueeze(0))

        m0 = compute_depth_metrics(base_depth, gt.unsqueeze(0), align=False)
        abs_rel_before.append(m0["abs_rel"])

        corrected = sm.infer(img)
        m1 = compute_depth_metrics(corrected, gt.unsqueeze(0).to(corrected.device), align=False)
        abs_rel_after.append(m1["abs_rel"])

    report = {
        "zero_shot_abs_rel": float(torch.tensor(abs_rel_before).mean()) if abs_rel_before else None,
        "adapted_abs_rel": float(torch.tensor(abs_rel_after).mean()) if abs_rel_after else None,
        "improvement_pct": None,
        "adapt_time_s": t_adapt,
        "peak_vram_gb": peak_vram,
        "num_steps": result.get("steps", args.num_steps),
    }
    if report["zero_shot_abs_rel"] and report["zero_shot_abs_rel"] > 0:
        report["improvement_pct"] = (
            (report["zero_shot_abs_rel"] - report["adapted_abs_rel"])
            / report["zero_shot_abs_rel"]
            * 100
        )

    print(json.dumps(report, indent=2))
    if args.output:
        (args.output / "eval_results.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
