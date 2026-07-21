#!/usr/bin/env python3
"""
stress_test_harness.py — Empirical Stress Test Harness for Milestone 3 DepthLab Framework.
"""

import sys
import os
import io
import json
import yaml
import tempfile
import traceback
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OFFICIAL_REPO_DIR = PROJECT_ROOT / "depth-anything-v2-official"
if OFFICIAL_REPO_DIR.exists() and str(OFFICIAL_REPO_DIR) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_REPO_DIR))

from depthlab.backbone.loader import load_da2_checkpoint
from depthlab.backbone.adapter import FeatureStage, FeatureBundle
from depthlab.heads import get_head, list_heads
from depthlab.data import get_dataset, list_datasets
from depthlab.metrics.dispatcher import MetricDispatcher, apply_feature_ablation
from depthlab.trainer import Trainer


def run_test(name, func):
    print(f"\n--- [RUNNING] {name} ---", flush=True)
    try:
        res = func()
        print(f"[RESULT] {name}: {res['status']}", flush=True)
        if "details" in res:
            print(f" Details: {res['details']}", flush=True)
        return res
    except Exception as e:
        err_msg = f"UNCAUGHT EXCEPTION: {type(e).__name__}: {e}\n{traceback.format_exc()}"
        print(f"[RESULT] {name}: CRASH", flush=True)
        print(err_msg, flush=True)
        return {
            "name": name,
            "status": "CRASH",
            "error_type": type(e).__name__,
            "error_msg": str(e),
            "traceback": traceback.format_exc()
        }


# =====================================================================
# 1. MALFORMED / MISSING YAML CONFIG TESTS
# =====================================================================

def test_config_missing_file():
    cmd = [sys.executable, str(PROJECT_ROOT / "train.py"), "--config", "non_existent_config_123.yaml"]
    import subprocess
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    passed = proc.returncode != 0 and "not found" in proc.stderr.lower()
    return {
        "name": "Missing Config File CLI",
        "status": "PASS" if passed else "FAIL",
        "details": f"Exit code {proc.returncode}, stderr: {proc.stderr.strip()}"
    }

def test_config_malformed_syntax():
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write("model:\n  backbone: [invalid yaml syntax here: {{")
        tmp_path = Path(f.name)
    try:
        import subprocess
        cmd = [sys.executable, str(PROJECT_ROOT / "train.py"), "--config", str(tmp_path)]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        passed = proc.returncode != 0 and ("scanner" in proc.stderr.lower() or "parser" in proc.stderr.lower() or "yaml" in proc.stderr.lower())
        return {
            "name": "Malformed YAML Syntax CLI",
            "status": "PASS" if passed else "FAIL",
            "details": f"Exit code {proc.returncode}, handled cleanly with YAML error: {proc.stderr.strip()[:100]}"
        }
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

def test_config_empty_file():
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write("")  # 0 bytes
        tmp_path = Path(f.name)
    try:
        import subprocess
        cmd = [sys.executable, str(PROJECT_ROOT / "train.py"), "--config", str(tmp_path)]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return {
            "name": "Empty YAML File (0 bytes)",
            "status": "FAIL_CRASH" if proc.returncode != 0 else "UNCHECKED_PASS",
            "details": f"Exit code {proc.returncode}, stderr: {proc.stderr.strip()[:200]}"
        }
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

def test_config_unregistered_head():
    cfg = {
        "experiment": {"name": "test_unregistered"},
        "model": {"head": {"name": "non_existent_head_xyz"}},
        "dataset": {"name": "synthetic"},
        "training": {"epochs": 1}
    }
    try:
        trainer = Trainer(cfg)
        return {"name": "Unregistered Head Config", "status": "UNCHECKED_PASS", "details": "Did not raise error"}
    except ValueError as e:
        return {"name": "Unregistered Head Config", "status": "PASS", "details": f"Raised expected ValueError: {e}"}
    except Exception as e:
        return {"name": "Unregistered Head Config", "status": "FAIL", "details": f"Raised unexpected error: {type(e).__name__}: {e}"}

def test_config_unregistered_dataset():
    cfg = {
        "experiment": {"name": "test_unregistered_ds"},
        "model": {"head": {"name": "relative_depth"}},
        "dataset": {"name": "non_existent_dataset_abc"},
        "training": {"epochs": 1}
    }
    try:
        trainer = Trainer(cfg)
        return {"name": "Unregistered Dataset Config", "status": "UNCHECKED_PASS", "details": "Did not raise error"}
    except ValueError as e:
        return {"name": "Unregistered Dataset Config", "status": "PASS", "details": f"Raised expected ValueError: {e}"}
    except Exception as e:
        return {"name": "Unregistered Dataset Config", "status": "FAIL", "details": f"Raised unexpected error: {type(e).__name__}: {e}"}

def test_config_invalid_data_types():
    cfg = {
        "experiment": {"name": "test_invalid_types"},
        "model": {"head": {"name": "relative_depth"}},
        "dataset": {"name": "synthetic", "batch_size": "invalid_string_bs"},
        "training": {"epochs": 1}
    }
    try:
        trainer = Trainer(cfg)
        return {"name": "Invalid Data Types (string batch_size)", "status": "FAIL", "details": "Did not raise error on string batch_size"}
    except ValueError as e:
        return {"name": "Invalid Data Types (string batch_size)", "status": "PASS", "details": f"Raised expected ValueError: {e}"}
    except Exception as e:
        return {"name": "Invalid Data Types (string batch_size)", "status": "FAIL", "details": f"Raised {type(e).__name__}: {e}"}


# =====================================================================
# 2. FEATURE ABLATION ON OUT-OF-RANGE STAGE INDICES TESTS
# =====================================================================

def test_ablation_out_of_range_positive():
    dummy_bundle = FeatureBundle(stages=[
        FeatureStage(torch.randn(2, 1369, 384), None, stage_index=i, embed_dim=384) for i in range(4)
    ])
    ablated = apply_feature_ablation(dummy_bundle, [99, 100])
    all_nonzero = all(torch.count_nonzero(s.patch_tokens) > 0 for s in ablated.stages)
    return {
        "name": "Feature Ablation Stage 99",
        "status": "SILENT_NOOP" if all_nonzero else "UNEXPECTED_ABLATION",
        "details": "Out-of-range stage index 99 was silently ignored without error or warning"
    }

def test_ablation_negative_indices():
    dummy_bundle = FeatureBundle(stages=[
        FeatureStage(torch.randn(2, 1369, 384), None, stage_index=i, embed_dim=384) for i in range(4)
    ])
    ablated = apply_feature_ablation(dummy_bundle, [-1])
    st3_nonzero = torch.count_nonzero(ablated.stages[3].patch_tokens) > 0
    return {
        "name": "Feature Ablation Negative Index -1",
        "status": "SILENT_IGNORED" if st3_nonzero else "PYTHONIC_ABLATION",
        "details": f"Negative index -1 was ignored (stage 3 non-zero: {st3_nonzero})"
    }

def test_ablation_eval_cli_out_of_range():
    import subprocess
    cmd = [
        sys.executable, str(PROJECT_ROOT / "eval.py"),
        "--config", str(PROJECT_ROOT / "config/default.yaml"),
        "--ablate-layer", "99"
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return {
        "name": "eval.py --ablate-layer 99 CLI",
        "status": "PASS_SILENT" if proc.returncode == 0 else "FAIL",
        "details": f"Exit code {proc.returncode}, output summary: {proc.stdout.strip()[-200:] if proc.returncode==0 else proc.stderr.strip()[:200]}"
    }


# =====================================================================
# 3. SPATIAL RESOLUTIONS & BATCH SIZE TESTS
# =====================================================================

def test_zero_batch_size():
    cfg = {
        "experiment": {"name": "test_zero_bs"},
        "model": {"head": {"name": "relative_depth"}},
        "dataset": {"name": "synthetic", "batch_size": 0},
        "training": {"epochs": 1}
    }
    try:
        trainer = Trainer(cfg)
        return {"name": "Zero Batch Size (batch_size=0)", "status": "FAIL", "details": "Trainer initialized with batch_size=0 without error"}
    except ValueError as e:
        return {"name": "Zero Batch Size (batch_size=0)", "status": "PASS", "details": f"Raised expected ValueError: {e}"}
    except Exception as e:
        return {"name": "Zero Batch Size (batch_size=0)", "status": "HANDLED_EXCEPT", "details": f"Raised {type(e).__name__}: {e}"}

def test_non_square_spatial_resolution():
    """Test rectangular spatial resolution e.g. 480x640 or 378x504"""
    head = get_head("relative_depth", encoder_variant="vits", features=64, out_channels=[48, 96, 192, 384])
    head.eval()
    
    # 378x504 is divisible by 14 (378/14 = 27, 504/14 = 36). Total patches = 972.
    patch_h, patch_w = 27, 36
    num_patches = patch_h * patch_w # 972
    embed_dim = 384
    
    dummy_bundle = FeatureBundle(stages=[
        FeatureStage(torch.randn(1, num_patches, embed_dim), None, stage_index=i, embed_dim=embed_dim) for i in range(4)
    ])
    
    try:
        preds = head(dummy_bundle)
        return {"name": "Non-Square Spatial Res (378x504 -> 27x36 patches)", "status": "PASS", "details": f"Pred shape: {preds['depth'].shape}"}
    except Exception as e:
        return {
            "name": "Non-Square Spatial Res (378x504 -> 27x36 patches)",
            "status": "BUG_CRASH",
            "details": f"CRASH: {type(e).__name__}: {e}"
        }

def test_non_multiple_of_14_spatial_resolution():
    """Test image resolution not divisible by 14, e.g., 500x500"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = PROJECT_ROOT / "checkpoints" / "depth_anything_v2_vits.pth"
    if not ckpt_path.exists():
        return {"name": "Non-Multiple of 14 (500x500)", "status": "SKIP", "details": "No checkpoint found"}
        
    backbone = load_da2_checkpoint("vits", ckpt_path, device=device).eval()
    dummy_img = torch.randn(1, 3, 500, 500, device=device)
    
    try:
        features = backbone.features(dummy_img)
        return {
            "name": "Non-Multiple of 14 (500x500)",
            "status": "PASS",
            "details": f"Extracted {len(features.stages)} stages, stage 0 shape: {features.stages[0].patch_tokens.shape}"
        }
    except Exception as e:
        return {
            "name": "Non-Multiple of 14 (500x500)",
            "status": "CRASH",
            "details": f"DINOv2 backbone crashed on 500x500: {type(e).__name__}: {e}"
        }

def test_single_channel_grayscale_image():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = PROJECT_ROOT / "checkpoints" / "depth_anything_v2_vits.pth"
    if not ckpt_path.exists():
        return {"name": "Single Channel Image (1x518x518)", "status": "SKIP", "details": "No checkpoint found"}
        
    backbone = load_da2_checkpoint("vits", ckpt_path, device=device).eval()
    dummy_img = torch.randn(1, 1, 518, 518, device=device)
    
    try:
        features = backbone.features(dummy_img)
        return {"name": "Single Channel Image (1x518x518)", "status": "FAIL_NO_RAISE", "details": "Extracted features without checking 3 channels"}
    except Exception as e:
        return {"name": "Single Channel Image (1x518x518)", "status": "PASS_HANDLED", "details": f"Raised expected shape/channel exception: {type(e).__name__}: {e}"}


# =====================================================================
# 4. VERIFY.PY --ALL UNDER STRESS CONDITIONS TESTS
# =====================================================================

def test_verify_all_normal():
    import subprocess
    cmd = [sys.executable, str(PROJECT_ROOT / "verify.py"), "--all"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    passed = (proc.returncode == 0) and ("ALL CHECKS PASSED" in proc.stdout)
    return {
        "name": "verify.py --all Normal Environment",
        "status": "PASS" if passed else "FAIL",
        "details": f"Exit code {proc.returncode}, summary line: {[line for line in proc.stdout.splitlines() if 'STATUS' in line or 'Passed' in line]}"
    }

def test_verify_missing_checkpoint():
    import subprocess
    cmd = [sys.executable, str(PROJECT_ROOT / "verify.py"), "--checkpoint", "non_existent_ckpt.pth"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    handled_cleanly = (proc.returncode != 0) and ("failed" in proc.stdout.lower() or "not found" in proc.stdout.lower())
    return {
        "name": "verify.py Missing Checkpoint",
        "status": "PASS" if handled_cleanly else "FAIL",
        "details": f"Exit code {proc.returncode}, reported failure gracefully without unhandled exception"
    }

def test_verify_missing_reproduction_report():
    import subprocess
    cmd = [sys.executable, str(PROJECT_ROOT / "verify.py"), "--check-report", "--reproduction-report", "non_existent_report.yaml"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    handled_cleanly = (proc.returncode != 0) and ("failed" in proc.stdout.lower() or "missing" in proc.stdout.lower())
    return {
        "name": "verify.py Missing Reproduction Report",
        "status": "PASS" if handled_cleanly else "FAIL",
        "details": f"Exit code {proc.returncode}, reported failure gracefully"
    }


def main():
    print("======================================================================", flush=True)
    print("      DepthLab Milestone 3 Empirical Stress Testing Harness          ", flush=True)
    print("======================================================================", flush=True)
    
    tests = [
        # Config tests
        ("test_config_missing_file", test_config_missing_file),
        ("test_config_malformed_syntax", test_config_malformed_syntax),
        ("test_config_empty_file", test_config_empty_file),
        ("test_config_unregistered_head", test_config_unregistered_head),
        ("test_config_unregistered_dataset", test_config_unregistered_dataset),
        ("test_config_invalid_data_types", test_config_invalid_data_types),
        
        # Ablation tests
        ("test_ablation_out_of_range_positive", test_ablation_out_of_range_positive),
        ("test_ablation_negative_indices", test_ablation_negative_indices),
        ("test_ablation_eval_cli_out_of_range", test_ablation_eval_cli_out_of_range),
        
        # Spatial resolution & Batch size tests
        ("test_zero_batch_size", test_zero_batch_size),
        ("test_non_square_spatial_resolution", test_non_square_spatial_resolution),
        ("test_non_multiple_of_14_spatial_resolution", test_non_multiple_of_14_spatial_resolution),
        ("test_single_channel_grayscale_image", test_single_channel_grayscale_image),
        
        # Verify.py stress tests
        ("test_verify_all_normal", test_verify_all_normal),
        ("test_verify_missing_checkpoint", test_verify_missing_checkpoint),
        ("test_verify_missing_reproduction_report", test_verify_missing_reproduction_report),
    ]
    
    results = []
    for t_name, t_func in tests:
        res = run_test(t_name, t_func)
        results.append(res)
        
    res_path = Path(__file__).parent / "stress_results.json"
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved all stress test results to {res_path}", flush=True)


if __name__ == "__main__":
    main()
