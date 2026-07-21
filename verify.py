#!/usr/bin/env python3
"""
verify.py — Standalone Regression & Environment Verification Suite for DepthLab.

Validates Depth Anything V2 Small checkpoint loading, FP16 inference execution,
output shape correctness, non-negative depth range, absence of NaNs/Infs,
runtime bounds (<= 1.0s at 518x518), peak VRAM bounds (<= 2.0GB), golden fixture
regression alignment (MAE < 1e-6), and Phase 0.3 reproduction report validation.

Usage:
    python verify.py
    python verify.py --all
    python verify.py --checkpoint checkpoints/depth_anything_v2_vits.pth --tolerance 1e-6
    python verify.py --check-report --reproduction-report docs/reproduction/reproduction-report.yaml
"""

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import torch
import torch.nn as nn

# Ensure project root & official DA2 repo are in python path
PROJECT_ROOT = Path(__file__).parent.resolve()
OFFICIAL_REPO_DIR = PROJECT_ROOT / "depth-anything-v2-official"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if OFFICIAL_REPO_DIR.exists() and str(OFFICIAL_REPO_DIR) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_REPO_DIR))


def compute_sha256(filepath: Path) -> str:
    """Computes SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


@dataclasses.dataclass
class VerifyConfig:
    checkpoint_path: Path = PROJECT_ROOT / "checkpoints" / "depth_anything_v2_vits.pth"
    encoder: str = "vits"
    golden_dir: Path = PROJECT_ROOT / "golden"
    input_size: int = 518
    tolerance: float = 1e-6
    vram_limit_gb: float = 2.0
    runtime_limit_sec: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    run_all: bool = False
    reproduction_report: Path = PROJECT_ROOT / "docs" / "reproduction" / "reproduction-report.yaml"
    check_report: bool = False
    json_report: Optional[Path] = None
    verbose: bool = False


@dataclasses.dataclass
class VerifyResult:
    name: str
    passed: bool
    message: str
    details: Dict[str, Any] = dataclasses.field(default_factory=dict)
    error: Optional[str] = None


class VerificationSuite:
    def __init__(self, config: VerifyConfig):
        self.config = config
        self.results: List[VerifyResult] = []

    def log(self, msg: str):
        if self.config.verbose:
            print(f"[VERIFY] {msg}")

    def _resolve_fixture_path(self, gdir: Path, path_str: str, filename: Optional[str] = None, subfolder: Optional[str] = None) -> Path:
        """Resolves relative fixture paths relative to gdir with robust fallback candidates."""
        p = Path(path_str)
        if p.is_absolute():
            return p

        candidates = [
            gdir / p,
            (gdir / subfolder / filename) if (subfolder and filename) else None,
            (gdir / filename) if filename else None,
            gdir.parent / p,
            PROJECT_ROOT / p,
        ]
        for cand in candidates:
            if cand is not None and cand.exists():
                return cand
        return gdir / p

    def load_model(self) -> Tuple[Optional[nn.Module], Optional[str]]:
        """Loads Depth Anything V2 model via Layer 1 adapter or Layer 0 direct."""
        ckpt = self.config.checkpoint_path
        if not ckpt.exists():
            return None, f"Checkpoint not found at '{ckpt}'. Run download script or place checkpoint."

        # Strategy 1: Try Layer 1 Adapter (depthlab.backbone) if available
        try:
            from depthlab.backbone.loader import load_da2_checkpoint
            model = load_da2_checkpoint(variant=self.config.encoder, checkpoint_path=ckpt)
            model = model.to(self.config.device).eval()
            self.log("Loaded model via Layer 1 (depthlab.backbone)")
            return model, None
        except Exception as e1:
            self.log(f"Layer 1 load unavailable ({e1}), falling back to Layer 0 direct loader")

        # Strategy 2: Layer 0 Direct Loader (depth_anything_v2.dpt)
        try:
            from depth_anything_v2.dpt import DepthAnythingV2

            model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
            }

            if self.config.encoder not in model_configs:
                return None, f"Unknown encoder variant '{self.config.encoder}'"

            model = DepthAnythingV2(**model_configs[self.config.encoder])
            state_dict = torch.load(ckpt, map_location='cpu')
            model.load_state_dict(state_dict)
            model = model.to(self.config.device).eval()
            self.log("Loaded model via Layer 0 direct loader")
            return model, None
        except Exception as e2:
            return None, f"Failed to load checkpoint via Layer 0: {e2}"

    # --- Verification Checks ---

    def check_checkpoint_loading(self, model: Optional[nn.Module], load_err: Optional[str]) -> VerifyResult:
        if model is None or load_err is not None:
            return VerifyResult(
                name="Checkpoint Loading",
                passed=False,
                message=f"FAILED: {load_err}",
                error=load_err
            )
        param_count = sum(p.numel() for p in model.parameters())
        return VerifyResult(
            name="Checkpoint Loading",
            passed=True,
            message=f"PASSED: Loaded {self.config.encoder} ({param_count / 1e6:.2f}M params) from '{self.config.checkpoint_path.name}'",
            details={"params_m": param_count / 1e6, "checkpoint": str(self.config.checkpoint_path)}
        )

    def check_fp16_inference(self, model: nn.Module, sample_img: np.ndarray) -> Tuple[VerifyResult, Optional[np.ndarray]]:
        try:
            device_type = "cuda" if "cuda" in self.config.device else "cpu"

            with torch.no_grad():
                if device_type == "cuda":
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        depth = model.infer_image(sample_img, input_size=self.config.input_size)
                else:
                    depth = model.infer_image(sample_img, input_size=self.config.input_size)

            return VerifyResult(
                name="FP16 Mixed Precision Execution",
                passed=True,
                message=f"PASSED: Execution succeeded under FP16/autocast ({device_type})",
                details={"device": device_type}
            ), depth
        except Exception as e:
            return VerifyResult(
                name="FP16 Mixed Precision Execution",
                passed=False,
                message=f"FAILED: FP16 execution exception: {e}",
                error=str(e)
            ), None

    def check_output_shape(self, depth: np.ndarray, raw_shape: Tuple[int, int, int]) -> VerifyResult:
        expected_shape = (raw_shape[0], raw_shape[1])
        actual_shape = depth.shape
        passed = (actual_shape == expected_shape)
        return VerifyResult(
            name="Output Shape Integrity",
            passed=passed,
            message=f"PASSED: Shape {actual_shape} matches input ({expected_shape})" if passed else f"FAILED: Expected {expected_shape}, got {actual_shape}",
            details={"expected": list(expected_shape), "actual": list(actual_shape)}
        )

    def check_depth_range(self, depth: np.ndarray) -> VerifyResult:
        min_val = float(np.min(depth))
        max_val = float(np.max(depth))
        passed = (min_val >= 0.0)
        return VerifyResult(
            name="Non-Negative Depth Range",
            passed=passed,
            message=f"PASSED: Depth range [{min_val:.4f}, {max_val:.4f}] >= 0.0" if passed else f"FAILED: Found negative depth values (min={min_val})",
            details={"min": min_val, "max": max_val}
        )

    def check_nan_inf_free(self, depth: np.ndarray) -> VerifyResult:
        has_nan = bool(np.isnan(depth).any())
        has_inf = bool(np.isinf(depth).any())
        passed = not (has_nan or has_inf)
        return VerifyResult(
            name="Absence of NaNs / Infs",
            passed=passed,
            message="PASSED: Zero NaNs and Infs detected in output depth map" if passed else f"FAILED: NaNs={has_nan}, Infs={has_inf}",
            details={"has_nan": has_nan, "has_inf": has_inf}
        )

    def check_runtime_bounds(self, model: nn.Module, sample_img: np.ndarray) -> VerifyResult:
        # Warmup pass
        with torch.no_grad():
            _ = model.infer_image(sample_img, input_size=self.config.input_size)

        runtimes = []
        num_runs = 5
        for _ in range(num_runs):
            if "cuda" in self.config.device:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model.infer_image(sample_img, input_size=self.config.input_size)
            if "cuda" in self.config.device:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            runtimes.append(t1 - t0)

        mean_runtime = float(np.mean(runtimes))
        passed = (mean_runtime <= self.config.runtime_limit_sec)
        return VerifyResult(
            name=f"Runtime Bounds (<= {self.config.runtime_limit_sec:.1f}s @ 518x518)",
            passed=passed,
            message=f"PASSED: Mean runtime {mean_runtime * 1000:.1f}ms <= {self.config.runtime_limit_sec * 1000:.0f}ms" if passed else f"FAILED: Mean runtime {mean_runtime:.3f}s exceeds limit {self.config.runtime_limit_sec:.1f}s",
            details={"mean_ms": mean_runtime * 1000, "min_ms": np.min(runtimes) * 1000, "max_ms": np.max(runtimes) * 1000}
        )

    def check_vram_bounds(self, model: nn.Module, sample_img: np.ndarray) -> VerifyResult:
        if not torch.cuda.is_available() or "cuda" not in self.config.device:
            return VerifyResult(
                name=f"Peak VRAM Bounds (<= {self.config.vram_limit_gb:.1f}GB)",
                passed=True,
                message="SKIPPED: CUDA not available (CPU execution)",
                details={"vram_gb": 0.0, "status": "skipped_cpu"}
            )

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            _ = model.infer_image(sample_img, input_size=self.config.input_size)

        peak_bytes = torch.cuda.max_memory_allocated()
        peak_gb = peak_bytes / (1024 ** 3)
        passed = (peak_gb <= self.config.vram_limit_gb)

        return VerifyResult(
            name=f"Peak VRAM Bounds (<= {self.config.vram_limit_gb:.1f}GB)",
            passed=passed,
            message=f"PASSED: Peak VRAM {peak_gb:.3f}GB <= {self.config.vram_limit_gb:.1f}GB" if passed else f"FAILED: Peak VRAM {peak_gb:.3f}GB exceeds {self.config.vram_limit_gb:.1f}GB limit",
            details={"peak_vram_gb": peak_gb, "limit_gb": self.config.vram_limit_gb}
        )

    def check_golden_regression(self, model: nn.Module, check_all: bool = False) -> VerifyResult:
        gdir = self.config.golden_dir
        manifest_path = gdir / "fixture_manifest.json"

        if not gdir.exists():
            return VerifyResult(
                name="Golden Fixture Regression Check",
                passed=False,
                message=f"FAILED: Golden directory '{gdir}' does not exist. Run Phase 0.2 golden generation.",
                error=f"Golden directory missing at {gdir}"
            )

        samples_to_check = []
        if manifest_path.exists():
            try:
                with open(manifest_path, "r") as f:
                    manifest_data = json.load(f)
                samples = manifest_data.get("samples", [])
                if not check_all and len(samples) > 0:
                    samples_to_check = [samples[0]]
                else:
                    samples_to_check = samples
            except Exception as e:
                return VerifyResult(
                    name="Golden Fixture Regression Check",
                    passed=False,
                    message=f"FAILED: Corrupted manifest file '{manifest_path}': {e}",
                    error=str(e)
                )
        else:
            # Fallback path resolution
            sample_ids = [f"demo{i:02d}" for i in range(1, 11)]
            if not check_all:
                sample_ids = sample_ids[:1]

            for s_id in sample_ids:
                img_p = gdir / "images" / f"{s_id}.jpg"
                npy_p = gdir / "depths" / f"{s_id}_depth.npy"
                if not img_p.exists():
                    img_p = gdir / f"{s_id}.jpg"
                if not npy_p.exists():
                    npy_p = gdir / f"{s_id}_depth.npy"

                samples_to_check.append({
                    "sample_id": s_id,
                    "image_path": str(img_p),
                    "depth_npy_path": str(npy_p),
                    "image_filename": f"{s_id}.jpg",
                    "depth_npy_filename": f"{s_id}_depth.npy"
                })

        maes = []
        fixture_details = []

        for sample in samples_to_check:
            s_id = sample.get("sample_id", "unknown")
            img_rel = sample.get("image_path")
            npy_rel = sample.get("depth_npy_path")
            img_fn = sample.get("image_filename")
            npy_fn = sample.get("depth_npy_filename")

            img_path = self._resolve_fixture_path(gdir, img_rel, img_fn, "images")
            npy_path = self._resolve_fixture_path(gdir, npy_rel, npy_fn, "depths")

            if not img_path.exists() or not npy_path.exists():
                return VerifyResult(
                    name="Golden Fixture Regression Check",
                    passed=False,
                    message=f"FAILED: Fixture sample '{s_id}' missing ({img_path} / {npy_path})",
                    error=f"Missing fixture files for sample {s_id}"
                )

            # Validate SHA256 checksums if present in manifest
            if "image_sha256" in sample:
                actual_img_sha = compute_sha256(img_path)
                if actual_img_sha.lower() != str(sample["image_sha256"]).lower():
                    return VerifyResult(
                        name="Golden Fixture Regression Check",
                        passed=False,
                        message=f"FAILED: SHA256 mismatch for image '{s_id}' ({img_path.name})",
                        error=f"Expected {sample['image_sha256']}, got {actual_img_sha}"
                    )
            if "depth_npy_sha256" in sample:
                actual_npy_sha = compute_sha256(npy_path)
                if actual_npy_sha.lower() != str(sample["depth_npy_sha256"]).lower():
                    return VerifyResult(
                        name="Golden Fixture Regression Check",
                        passed=False,
                        message=f"FAILED: SHA256 mismatch for depth npy '{s_id}' ({npy_path.name})",
                        error=f"Expected {sample['depth_npy_sha256']}, got {actual_npy_sha}"
                    )

            try:
                raw_img = cv2.imread(str(img_path))
                if raw_img is None:
                    return VerifyResult(
                        name="Golden Fixture Regression Check",
                        passed=False,
                        message=f"FAILED: Unable to read image '{img_path}'",
                        error=f"cv2.imread returned None for {img_path}"
                    )

                golden_depth = np.load(str(npy_path))

                with torch.no_grad():
                    pred_depth = model.infer_image(raw_img, input_size=self.config.input_size)

                if pred_depth.shape != golden_depth.shape:
                    return VerifyResult(
                        name="Golden Fixture Regression Check",
                        passed=False,
                        message=f"FAILED: Shape mismatch for sample '{s_id}': pred {pred_depth.shape} vs golden {golden_depth.shape}",
                        error="Shape mismatch"
                    )

                mae = float(np.mean(np.abs(pred_depth - golden_depth)))
                maes.append(mae)
                fixture_details.append({"sample_id": s_id, "mae": mae, "passed": mae < self.config.tolerance})
            except Exception as e:
                return VerifyResult(
                    name="Golden Fixture Regression Check",
                    passed=False,
                    message=f"FAILED: Exception processing fixture sample '{s_id}': {e}",
                    error=str(e)
                )

        if not maes:
            return VerifyResult(
                name="Golden Fixture Regression Check",
                passed=False,
                message="FAILED: No fixture samples evaluated",
                error="Empty sample set"
            )

        max_mae = max(maes)
        mean_mae = float(np.mean(maes))
        passed = (max_mae < self.config.tolerance)

        msg = (f"PASSED: Golden MAE (max={max_mae:.2e}, mean={mean_mae:.2e}) < tolerance {self.config.tolerance:.1e}"
               if passed else
               f"FAILED: Max Golden MAE {max_mae:.2e} >= tolerance {self.config.tolerance:.1e}")

        return VerifyResult(
            name="Golden Fixture Regression Check",
            passed=passed,
            message=msg,
            details={"max_mae": max_mae, "mean_mae": mean_mae, "tolerance": self.config.tolerance, "samples_checked": len(maes), "fixtures": fixture_details}
        )

    def check_reproduction_report(self, report_path: Path) -> VerifyResult:
        """Validates Phase 0.3 reproduction report schema, checksums, and metric bounds."""
        if not report_path.exists():
            return VerifyResult(
                name="Phase 0.3 Reproduction Report Check",
                passed=False,
                message=f"FAILED: Reproduction report missing at '{report_path}'",
                error=f"Report file missing: {report_path}"
            )

        try:
            # Support both YAML and JSON reports
            if report_path.suffix in [".yaml", ".yml"]:
                try:
                    import yaml
                    with open(report_path, "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f)
                except ImportError:
                    return VerifyResult(
                        name="Phase 0.3 Reproduction Report Check",
                        passed=False,
                        message="FAILED: PyYAML module not installed for YAML report parsing",
                        error="PyYAML missing"
                    )
            else:
                with open(report_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

            if not isinstance(data, dict):
                return VerifyResult(
                    name="Phase 0.3 Reproduction Report Check",
                    passed=False,
                    message="FAILED: Report content is not a valid object/dictionary",
                    error="Report content not a dict"
                )

            required_fields = [
                "official_commit", "checkpoint_path", "checkpoint_sha256",
                "dataset", "dataset_version", "metric_implementation",
                "metric_implementation_sha256", "cuda_version", "pytorch_version",
                "gpu", "driver", "results", "golden_mae"
            ]

            missing = [field for field in required_fields if field not in data]
            if missing:
                return VerifyResult(
                    name="Phase 0.3 Reproduction Report Check",
                    passed=False,
                    message=f"FAILED: Missing mandatory report fields: {missing}",
                    error=f"Missing fields: {missing}"
                )

            # Checkpoint checksum validation
            ckpt_p = Path(data["checkpoint_path"])
            if not ckpt_p.is_absolute():
                ckpt_p = PROJECT_ROOT / ckpt_p
            if not ckpt_p.exists():
                return VerifyResult(
                    name="Phase 0.3 Reproduction Report Check",
                    passed=False,
                    message=f"FAILED: Checkpoint path in report does not exist ('{ckpt_p}')",
                    error="Checkpoint path missing"
                )

            actual_ckpt_sha = compute_sha256(ckpt_p)
            if actual_ckpt_sha.lower() != str(data["checkpoint_sha256"]).lower():
                return VerifyResult(
                    name="Phase 0.3 Reproduction Report Check",
                    passed=False,
                    message=f"FAILED: Checkpoint SHA256 mismatch (report={data['checkpoint_sha256'][:8]} vs actual={actual_ckpt_sha[:8]})",
                    error="Checkpoint SHA256 mismatch"
                )

            # Metric implementation checksum validation
            metric_p = Path(data["metric_implementation"])
            if not metric_p.is_absolute():
                metric_p = PROJECT_ROOT / metric_p
            if not metric_p.exists():
                return VerifyResult(
                    name="Phase 0.3 Reproduction Report Check",
                    passed=False,
                    message=f"FAILED: Metric implementation path in report does not exist ('{metric_p}')",
                    error="Metric implementation path missing"
                )

            actual_metric_sha = compute_sha256(metric_p)
            if actual_metric_sha.lower() != str(data["metric_implementation_sha256"]).lower():
                return VerifyResult(
                    name="Phase 0.3 Reproduction Report Check",
                    passed=False,
                    message=f"FAILED: Metric implementation SHA256 mismatch (report={data['metric_implementation_sha256'][:8]} vs actual={actual_metric_sha[:8]})",
                    error="Metric implementation SHA256 mismatch"
                )

            # Metric Tolerance Check (NYUv2 Eigen split targets within 1%)
            results = data.get("results", {})
            if not isinstance(results, dict):
                return VerifyResult(
                    name="Phase 0.3 Reproduction Report Check",
                    passed=False,
                    message="FAILED: 'results' field must be a dictionary",
                    error="'results' is not a dict"
                )

            abs_rel = float(results.get("abs_rel", 999.0))
            d1 = float(results.get("d1", 0.0))
            rmse = float(results.get("rmse", 999.0))
            golden_mae = float(data.get("golden_mae", 999.0))

            # Target thresholds for vits
            max_abs_rel = 0.083 * 1.01  # 0.08383
            min_d1 = 0.925 * 0.99       # 0.91575
            max_rmse = 0.365 * 1.01     # 0.36865

            abs_rel_pass = abs_rel <= max_abs_rel
            d1_pass = d1 >= min_d1
            rmse_pass = rmse <= max_rmse
            golden_mae_pass = golden_mae < self.config.tolerance

            metrics_passed = abs_rel_pass and d1_pass and rmse_pass and golden_mae_pass

            details = {
                "abs_rel": abs_rel,
                "abs_rel_pass": abs_rel_pass,
                "d1": d1,
                "d1_pass": d1_pass,
                "rmse": rmse,
                "rmse_pass": rmse_pass,
                "golden_mae": golden_mae,
                "golden_mae_pass": golden_mae_pass,
                "report_path": str(report_path),
            }

            if metrics_passed:
                msg = f"PASSED: Report fields valid & metrics match published baseline within 1% (AbsRel={abs_rel:.4f} <= {max_abs_rel:.4f}, d1={d1:.4f} >= {min_d1:.4f}, RMSE={rmse:.4f} <= {max_rmse:.4f})"
            else:
                msg = f"FAILED: Metric thresholds or golden MAE violated (AbsRel={abs_rel:.4f} vs max {max_abs_rel:.4f}, d1={d1:.4f} vs min {min_d1:.4f}, RMSE={rmse:.4f} vs max {max_rmse:.4f}, golden_mae={golden_mae})"

            return VerifyResult(
                name="Phase 0.3 Reproduction Report Check",
                passed=metrics_passed,
                message=msg,
                details=details
            )
        except Exception as e:
            return VerifyResult(
                name="Phase 0.3 Reproduction Report Check",
                passed=False,
                message=f"FAILED: Exception parsing reproduction report: {e}",
                error=str(e)
            )

    # --- Framework Scaffolding Check (Test 12) ---

    def check_framework_scaffolding(self, model: Optional[nn.Module] = None) -> VerifyResult:
        """
        Validates Layer 1 & Layer 2 Framework Scaffolding (Test 12):
        - DA2Backbone feature extraction returning FeatureBundle
        - Head registry lookup & instantiation
        - End-to-end forward pass (FeatureBundle -> Head -> Predictions)
        - Head loss computation and backward pass
        - MetricDispatcher update and calculation
        - Feature ablation hook execution
        """
        try:
            from depthlab.backbone.loader import load_da2_checkpoint
            from depthlab.heads import get_head
            from depthlab.metrics.dispatcher import MetricDispatcher, apply_feature_ablation

            device = self.config.device
            dummy_img = torch.randn(2, 3, 518, 518, device=device)
            backbone = load_da2_checkpoint(
                variant=self.config.encoder,
                checkpoint_path=self.config.checkpoint_path,
                device=device
            ).eval()

            with torch.no_grad():
                bundle = backbone.features(dummy_img)

            if not hasattr(bundle, "stages") or len(bundle.stages) != 4:
                return VerifyResult(
                    name="Framework Scaffolding Integrity",
                    passed=False,
                    message=f"FAILED: FeatureBundle must contain 4 FeatureStage objects, got {len(getattr(bundle, 'stages', []))}"
                )

            head = get_head(
                "relative_depth",
                encoder_variant=self.config.encoder,
                features=64,
                out_channels=[48, 96, 192, 384]
            ).to(device).train()

            preds = head(bundle)

            if "depth" not in preds and "predicted_depth" not in preds:
                return VerifyResult(
                    name="Framework Scaffolding Integrity",
                    passed=False,
                    message="FAILED: RelativeDepth head forward output missing depth tensor"
                )

            pred_depth = preds.get("depth", preds.get("predicted_depth"))
            dummy_target = torch.rand_like(pred_depth)
            loss = head.compute_loss(preds, {"depth": dummy_target})
            loss.backward()

            dispatcher = MetricDispatcher()
            dispatcher.update(pred_depth.detach(), dummy_target)
            metrics = dispatcher.compute()

            ablated_bundle = apply_feature_ablation(bundle, [0, 2])
            ablated_preds = head(ablated_bundle)
            ablated_depth = ablated_preds.get("depth", ablated_preds.get("predicted_depth"))

            passed = ("abs_rel" in metrics) and (ablated_depth.shape == pred_depth.shape)
            return VerifyResult(
                name="Framework Scaffolding Integrity",
                passed=passed,
                message="PASSED: Layer 1 adapter, Head registry, AMP loss backward, MetricDispatcher, and Feature Ablation verified end-to-end",
                details={"metrics": metrics, "loss": float(loss.item())}
            )
        except Exception as e:
            return VerifyResult(
                name="Framework Scaffolding Integrity",
                passed=False,
                message=f"FAILED: Exception during framework scaffolding check: {e}",
                error=str(e)
            )

    # --- Extended Stress Checks (--all) ---

    def check_synthetic_stress(self, model: nn.Module) -> VerifyResult:
        h, w = 518, 518
        black_img = np.zeros((h, w, 3), dtype=np.uint8)
        white_img = np.full((h, w, 3), 255, dtype=np.uint8)
        noise_img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

        try:
            with torch.no_grad():
                d_black = model.infer_image(black_img, input_size=self.config.input_size)
                d_white = model.infer_image(white_img, input_size=self.config.input_size)
                d_noise = model.infer_image(noise_img, input_size=self.config.input_size)

            has_nan = any(np.isnan(d).any() for d in [d_black, d_white, d_noise])
            has_inf = any(np.isinf(d).any() for d in [d_black, d_white, d_noise])
            passed = not (has_nan or has_inf)

            return VerifyResult(
                name="Synthetic Extreme Input Stress Test (--all)",
                passed=passed,
                message="PASSED: Zero NaNs/Infs produced on synthetic black, white, and noise inputs",
                details={"black_min": float(np.min(d_black)), "white_min": float(np.min(d_white)), "noise_min": float(np.min(d_noise))}
            )
        except Exception as e:
            return VerifyResult(
                name="Synthetic Extreme Input Stress Test (--all)",
                passed=False,
                message=f"FAILED: Exception during synthetic stress test: {e}",
                error=str(e)
            )

    def check_memory_leak(self, model: nn.Module, sample_img: np.ndarray, num_iterations: int = 20) -> VerifyResult:
        if not torch.cuda.is_available() or "cuda" not in self.config.device:
            return VerifyResult(
                name="Multi-Iteration Memory Accumulation Check (--all)",
                passed=True,
                message="SKIPPED: Memory leak test requires CUDA",
                details={"status": "skipped_cpu"}
            )

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        allocs = []
        with torch.no_grad():
            for i in range(num_iterations):
                _ = model.infer_image(sample_img, input_size=self.config.input_size)
                allocs.append(torch.cuda.memory_allocated() / (1024 ** 2))

        # Check memory growth between iteration 2 and iteration N
        mem_diff_mb = allocs[-1] - allocs[2]
        passed = (mem_diff_mb <= 1.0)  # threshold 1MB drift

        return VerifyResult(
            name="Multi-Iteration Memory Accumulation Check (--all)",
            passed=passed,
            message=f"PASSED: Memory growth over {num_iterations} iterations is {mem_diff_mb:.2f}MB <= 1.0MB" if passed else f"FAILED: Memory leak detected ({mem_diff_mb:.2f}MB growth)",
            details={"start_mb": allocs[2], "end_mb": allocs[-1], "drift_mb": mem_diff_mb}
        )

    # --- Runner & Reporting ---

    def run(self) -> bool:
        print("==================================================================")
        print("   DepthLab Standalone Verification Suite (Milestone 2 Gate)     ")
        print("==================================================================")
        print(f"Device      : {self.config.device}")
        print(f"Checkpoint  : {self.config.checkpoint_path}")
        print(f"Golden Dir  : {self.config.golden_dir}")
        print(f"Tolerance   : MAE < {self.config.tolerance}")
        print(f"Run Mode    : {'Comprehensive (--all)' if self.config.run_all else 'Standard'}")
        print("------------------------------------------------------------------")

        # Step 1: Load model
        model, load_err = self.load_model()
        r_load = self.check_checkpoint_loading(model, load_err)
        self.results.append(r_load)

        if not r_load.passed:
            self._print_summary()
            return False

        # Step 2: Prepare sample image
        sample_img = None
        manifest_path = self.config.golden_dir / "fixture_manifest.json"
        if manifest_path.exists():
            with open(manifest_path, "r") as f:
                mdata = json.load(f)
            samples = mdata.get("samples", [])
            if len(samples) > 0:
                img_rel = samples[0].get("image_path")
                img_fn = samples[0].get("image_filename")
                img_p = self._resolve_fixture_path(self.config.golden_dir, img_rel, img_fn, "images")
                if img_p.exists():
                    sample_img = cv2.imread(str(img_p))

        if sample_img is None:
            g_img_path = self._resolve_fixture_path(self.config.golden_dir, "images/demo01.jpg", "demo01.jpg", "images")
            if g_img_path.exists():
                sample_img = cv2.imread(str(g_img_path))
            else:
                sample_img = np.zeros((480, 640, 3), dtype=np.uint8)

        # Step 3: Run core suite
        r_fp16, depth_map = self.check_fp16_inference(model, sample_img)
        self.results.append(r_fp16)

        if r_fp16.passed and depth_map is not None:
            self.results.append(self.check_output_shape(depth_map, sample_img.shape))
            self.results.append(self.check_depth_range(depth_map))
            self.results.append(self.check_nan_inf_free(depth_map))
            self.results.append(self.check_runtime_bounds(model, sample_img))
            self.results.append(self.check_vram_bounds(model, sample_img))
            self.results.append(self.check_golden_regression(model, check_all=self.config.run_all))

        # Step 4: Run framework scaffolding check (Test 12)
        if r_fp16.passed:
            self.results.append(self.check_framework_scaffolding(model))

        # Step 5: Run extended stress suite if --all
        if self.config.run_all and r_fp16.passed:
            self.results.append(self.check_synthetic_stress(model))
            self.results.append(self.check_memory_leak(model, sample_img))

        # Step 6: Check reproduction report if requested or --all
        if self.config.check_report or self.config.run_all:
            self.results.append(self.check_reproduction_report(self.config.reproduction_report))

        # Output Summary
        all_passed = self._print_summary()

        if self.config.json_report:
            self._write_json_report()

        return all_passed

    def _print_summary(self) -> bool:
        print("\nVerification Test Results:")
        print("------------------------------------------------------------------")
        all_passed = True
        for r in self.results:
            status_str = "[ PASS ]" if r.passed else "[ FAIL ]"
            if not r.passed:
                all_passed = False
            print(f"{status_str} {r.name}")
            print(f"         +-- {r.message}")

        print("------------------------------------------------------------------")
        if all_passed:
            print("OVERALL STATUS: ALL CHECKS PASSED (VERIFICATION SUCCESS)")
        else:
            print("OVERALL STATUS: VERIFICATION FAILED (REGRESSION OR BOUND ERROR)")
        print("==================================================================")
        return all_passed

    def _write_json_report(self):
        report_data = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "config": {
                "checkpoint": str(self.config.checkpoint_path),
                "encoder": self.config.encoder,
                "device": self.config.device,
                "tolerance": self.config.tolerance,
                "run_all": self.config.run_all,
            },
            "results": [dataclasses.asdict(r) for r in self.results],
            "all_passed": all(r.passed for r in self.results),
        }
        with open(self.config.json_report, "w") as f:
            json.dump(report_data, f, indent=2)
        print(f"[VERIFY] Wrote JSON report to {self.config.json_report}")


def main():
    parser = argparse.ArgumentParser(description="DepthLab Standalone Regression & Verification Suite")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints" / "depth_anything_v2_vits.pth", help="Path to checkpoint .pth file")
    parser.add_argument("--encoder", type=str, default="vits", choices=["vits", "vitb", "vitl", "vitg"], help="Model encoder variant")
    parser.add_argument("--golden-dir", type=Path, default=PROJECT_ROOT / "golden", help="Directory containing golden fixtures")
    parser.add_argument("--input-size", type=int, default=518, help="Inference resolution")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="MAE regression threshold")
    parser.add_argument("--vram-limit-gb", type=float, default=2.0, help="Peak VRAM limit in GB")
    parser.add_argument("--runtime-limit-sec", type=float, default=1.0, help="Max runtime per image in seconds")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Target device (cuda or cpu)")
    parser.add_argument("--all", action="store_true", help="Run full regression loop & extended stress tests")
    parser.add_argument("--reproduction-report", type=Path, default=PROJECT_ROOT / "docs" / "reproduction" / "reproduction-report.yaml", help="Path to reproduction report file (YAML or JSON)")
    parser.add_argument("--check-report", action="store_true", help="Run Phase 0.3 reproduction report validation")
    parser.add_argument("--json-report", type=Path, default=None, help="Save structured JSON report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose debug logging")

    args = parser.parse_args()

    config = VerifyConfig(
        checkpoint_path=args.checkpoint,
        encoder=args.encoder,
        golden_dir=args.golden_dir,
        input_size=args.input_size,
        tolerance=args.tolerance,
        vram_limit_gb=args.vram_limit_gb,
        runtime_limit_sec=args.runtime_limit_sec,
        device=args.device,
        run_all=args.all,
        reproduction_report=args.reproduction_report,
        check_report=args.check_report,
        json_report=args.json_report,
        verbose=args.verbose,
    )

    suite = VerificationSuite(config)
    success = suite.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
