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

    adapt_set = AdaptationManifest(args.manifest, split="adapt")
    test_set = AdaptationManifest(args.manifest, split="test")
    kf_indices = select_keyframes(adapt_set, max_frames=5)

    keyframes = []
    for idx in kf_indices:
        item = adapt_set[idx]
        image = item["image"].to(args.device)
        feats, bd = _extract_features_and_depth(backbone, image)
        keyframes.append({"features": feats[0], "base_depth": bd[0], "image": image[0],
                          "intrinsics": item["intrinsics"], "transform": item["transform"]})

    val_frames = []
    for idx in range(min(2, len(test_set))):
        item = test_set[idx]
        image = item["image"].to(args.device)
        feats, bd = _extract_features_and_depth(backbone, image)
        val_frames.append({"features": feats[0], "base_depth": bd[0], "image": image[0],
                           "intrinsics": item["intrinsics"], "transform": item["transform"]})

    correction_net = CorrectionNet().to("cpu")

    if torch.cuda.is_available() and args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    t0 = time.perf_counter()
    result = adapt_scene(backbone, correction_net, keyframes, val_frames,
                         device=torch.device(args.device), num_steps=args.num_steps,
                         output_dir=args.output)
    t_adapt = time.perf_counter() - t0

    peak_vram = 0.0
    if torch.cuda.is_available() and args.device == "cuda":
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)

    sm = SceneManager(backbone, correction_net, torch.device(args.device))
    sm.initial_state = correction_net.state_dict()
    sm.load_adapted(result["adapted_state_dict"], result["best_scale"])

    abs_rel_before = []
    abs_rel_after = []
    for idx in range(len(test_set)):
        item = test_set[idx]
        image = item["image"].to(args.device)
        gt = item.get("depth")
        if gt is None:
            continue
        gt = gt.to(args.device)

        bundle = backbone.features(image.unsqueeze(0))
        H, W = image.shape[-2:]
        patch_h, patch_w = H // 14, W // 14
        features = bundle.stages[-1].spatial_features(patch_h, patch_w)
        base_depth = backbone(image.unsqueeze(0))

        m0 = compute_depth_metrics(base_depth, gt.unsqueeze(0), align=False)
        abs_rel_before.append(m0["abs_rel"])

        corrected = sm.infer(image)
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
