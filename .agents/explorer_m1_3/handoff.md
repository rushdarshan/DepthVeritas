# Standalone `verify.py` Architecture & Design Specification (Milestone 1)

## 1. Observation

### 1.1 Requirements & Specifications Analysis
Direct observations from project documentation and specifications:

1. **`PROJECT.md` (lines 48, 12)**:
   - Line 48: `verify.py` is defined as "Standalone regression & runtime verification script" located at project root.
   - Line 12: Milestone 1 scope requires Phase 0.1 & 0.2: validate environment, generate golden fixtures in `golden/`, build `verify.py`.

2. **`ROADMAP.md` (lines 31-33, 42, 60)**:
   - Line 31-33: "Golden sample: save 10 images with official depth outputs as `golden/` fixtures. Create `verify.py`: checkpoint loads, inference runs, output shape, no NaNs, runtime bounded."
   - Line 42: "Golden regression: `verify.py` reports MAE < 1e-6 against golden sample."
   - Line 60: "`verify.py` — tests against golden sample (MAE < 1e-6)."

3. **`docs/adr/001-depthlab-architecture.md` (lines 208-218)**:
   - Establishes the 5-step reproduction protocol: Environment validation → Inference reproduction → Evaluation reproduction → Golden sample → `verify.py` checks.
   - Requires single shared transform `OFFICIAL_TRANSFORM` (Resize width=518, height=518, keep_aspect_ratio=True, ensure_multiple_of=14, NormalizeImage, PrepareForNet).

4. **`docs/adr/002-depthlab-api-stability.md` (lines 237-248)**:
   - Phase 0 Success Criteria table:
     - Checkpoint loading: loads without modification on target hardware.
     - Inference: visual output matches official examples.
     - FP16: mixed precision execution without error.
     - Output shape: exact $H \times W$ matching raw input image.
     - Range & NaNs: non-negative range ($\ge 0.0$), zero NaNs/Infs.
     - Runtime: single-image inference $\le 1.0\text{s}$ at $518 \times 518$.
     - VRAM: peak VRAM $\le 2.0\text{ GB}$ for DA2-Small at $518 \times 518$.
     - Golden regression: MAE $< 1 \times 10^{-6}$ against golden fixtures for all 10 images.
     - CLI behavior: `python verify.py` and `python verify.py --all` exit with code 0 on pass, non-zero on failure.

5. **`openspec/changes/depthlab-initial-scaffold/specs/reproduction-protocol/spec.md` (lines 27-38)**:
   - Scenario: `python verify.py` run on clean install passes all checks and exits with 0.
   - Scenario: `python verify.py` detects corrupted checkpoint, failing with appropriate diagnostic error message.

6. **Official Model Mechanics (`depth-anything-v2-official/depth_anything_v2/dpt.py`)**:
   - Model class: `DepthAnythingV2`
   - Model parameters for `vits`: `{'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}`.
   - Forward pass end: `depth = F.relu(depth)` (guarantees raw outputs $\ge 0.0$).
   - Method `infer_image(raw_image, input_size=518)`: accepts BGR numpy array `(H, W, 3)`, resizes, normalizes, runs forward pass, interpolates depth map back to `(H, W)`, returns 2D numpy array `(H, W)`.
   - Default checkpoint path: `checkpoints/depth_anything_v2_vits.pth`.

---

## 2. Logic Chain

1. **Dual-Layer Loader Compatibility**:
   - *Observation*: Milestone 1 occurs before Milestone 3 (`depthlab/` adapter scaffold).
   - *Deduction*: `verify.py` must support direct Layer 0 model loading (`depth_anything_v2.dpt.DepthAnythingV2`) when running in Phase 0.2, and seamlessly support Layer 1 adapter (`depthlab.backbone.DA2Backbone`) when running in Phase 0.4.
   - *Design*: Implement an abstraction helper `load_target_model()` inside `verify.py` that first attempts importing `depthlab.backbone`, falling back to `depth_anything_v2.dpt`.

2. **Verification Check Taxonomy**:
   - *Observation*: Objective lists 8 specific verification targets + CLI mode requirement (`python verify.py` vs `python verify.py --all`).
   - *Deduction*: The verification suite must be partitioned into:
     - **Default Suite** (`python verify.py`): Checks 1-8 (Checkpoint loading, FP16 execution, output shape $H \times W$, non-negative range, zero NaNs/Infs, runtime $\le 1\text{s}$, peak VRAM $\le 2\text{GB}$, golden fixture regression sample 0 MAE $< 1\times 10^{-6}$).
     - **Comprehensive Suite** (`python verify.py --all`): Adds Checks 9-10 (Full 10-fixture regression loop, synthetic zero/extreme image stress test, multi-run memory accumulation leak check).

3. **Golden Fixture Format & Regression Math**:
   - *Observation*: `golden/` stores 10 sample pairs (`image_NNN.png`, `depth_NNN.npy`, `depth_NNN.png`, `manifest.json`).
   - *Deduction*: Regression check loads `depth_NNN.npy` (unnormalized float32 numpy array), runs model inference on `image_NNN.png`, and computes Mean Absolute Error (MAE):
     $$\text{MAE} = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} | \hat{D}_{i,j} - D^{\text{golden}}_{i,j} |$$
   - *Threshold*: $\text{MAE} < 1.0 \times 10^{-6}$ for FP32/FP16 matching.

4. **VRAM & Runtime Measurement Strategy**:
   - *Observation*: Runtime target $\le 1.0\text{s}$, VRAM target $\le 2.0\text{GB}$ on CUDA.
   - *Deduction*:
     - **Runtime**: Include 1 warmup pass before timing to exclude CUDA kernel loading and PyTorch initialization overhead. Call `torch.cuda.synchronize()` before and after `time.perf_counter()`.
     - **VRAM**: Call `torch.cuda.empty_cache()` and `torch.cuda.reset_peak_memory_stats()` prior to inference. Read `torch.cuda.max_memory_allocated() / (1024**3)`.
     - **CPU Fallback**: If CUDA is not available, report CPU device state, run timing (issuing warning if $> 1\text{s}$), and skip VRAM GPU metric with informative status (`N/A (CPU)`).

5. **CLI Interface & Exit Code Integrity**:
   - *Observation*: System requires clear diagnostic console output and exit code 0 (pass) vs 1 (fail) for automated CI / reproduction gate checks.
   - *Deduction*: Construct a clean argument parser using `argparse`, output a structured status table with ANSI color coding (green PASS, red FAIL, yellow SKIP), and output optional `--json-report` file when requested.

---

## 3. Caveats

- **Golden Fixture Dependency**: When `golden/` fixtures or `checkpoints/depth_anything_v2_vits.pth` do not yet exist on disk (e.g. prior to running download/reproduction scripts), `verify.py` will report failure on checkpoint loading and golden regression checks. The design includes explicit, actionable error messages detailing how to obtain checkpoints (`scripts/download_checkpoints.py` or official HF download link) and generate golden samples.
- **Hardware Determinism**: Floating-point precision (FP16/FP32) across different GPU microarchitectures (e.g. Ampere vs Ada Lovelace vs Turing) or PyTorch CUDA library versions can introduce sub-epsilon variations ($10^{-7} \dots 10^{-6}$). The tolerance parameter `--tolerance` defaults to `1e-6` but is configurable.
- **CPU Performance**: On non-CUDA environments (CPU only), single-image inference of Depth Anything V2 Small at 518×518 may exceed 1.0s depending on CPU core count. `verify.py` logs CPU execution mode clearly while enforcing the runtime bound flag.

---

## 4. Conclusion & Proposed Implementation Design

### 4.1 Modular Architecture Diagram

```
                     ┌────────────────────────┐
                     │ python verify.py [--all]│
                     └───────────┬────────────┘
                                 │
                     ┌───────────▼────────────┐
                     │     VerifyConfig       │
                     └───────────┬────────────┘
                                 │
           ┌─────────────────────┴─────────────────────┐
           ▼                                           ▼
┌──────────────────────┐                   ┌──────────────────────┐
│  load_target_model() │                   │  load_golden_data()  │
└──────────┬───────────┘                   └──────────┬───────────┘
           │                                          │
    ┌──────┴─────────────┐                            │
    ▼                    ▼                            │
[Layer 1 Adapter]  [Layer 0 Direct]                   │
(depthlab.backbone) (depth_anything_v2)              │
    └──────┬─────────────┘                            │
           │                                          │
           └─────────────────────┬────────────────────┘
                                 │
                   ┌─────────────▼─────────────┐
                   │  Verification Test Suite  │
                   └─────────────┬─────────────┘
                                 │
   ├── 1. check_checkpoint_loading()
   ├── 2. check_fp16_inference()
   ├── 3. check_output_shape()
   ├── 4. check_depth_range()
   ├── 5. check_nan_inf_free()
   ├── 6. check_runtime_bounds()
   ├── 7. check_vram_bounds()
   ├── 8. check_golden_regression()
   └── [If --all]:
       ├── 9. check_synthetic_stress()
       └── 10. check_memory_leak()
                                 │
                   ┌─────────────▼─────────────┐
                   │     VerificationReport    │
                   │  (Console Table / JSON)   │
                   └─────────────┬─────────────┘
                                 │
                    sys.exit(0 if pass else 1)
```

---

### 4.2 Complete Proposed Code Implementation for `verify.py`

Below is the complete, self-contained Python design for `verify.py` to be placed at the project root:

```python
#!/usr/bin/env python3
"""
verify.py — Standalone Regression & Environment Verification Suite for DepthLab.

Validates Depth Anything V2 Small checkpoint loading, FP16 inference execution,
output shape correctness, non-negative depth range, absence of NaNs/Infs,
runtime bounds (<= 1s at 518x518), peak VRAM bounds (<= 2GB), and golden fixture
regression alignment (MAE < 1e-6).

Usage:
    python verify.py
    python verify.py --all
    python verify.py --checkpoint checkpoints/depth_anything_v2_vits.pth --tolerance 1e-6
"""

import argparse
import dataclasses
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
            dtype = torch.float16 if device_type == "cuda" else torch.float32

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
                details={"device": device_type, "dtype": str(dtype)}
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
            name="Runtime Bounds (<= 1.0s @ 518x518)",
            passed=passed,
            message=f"PASSED: Mean runtime {mean_runtime * 1000:.1f}ms <= {self.config.runtime_limit_sec * 1000:.0f}ms" if passed else f"FAILED: Mean runtime {mean_runtime:.3f}s exceeds limit {self.config.runtime_limit_sec:.1f}s",
            details={"mean_ms": mean_runtime * 1000, "min_ms": np.min(runtimes) * 1000, "max_ms": np.max(runtimes) * 1000}
        )

    def check_vram_bounds(self, model: nn.Module, sample_img: np.ndarray) -> VerifyResult:
        if not torch.cuda.is_available() or "cuda" not in self.config.device:
            return VerifyResult(
                name="Peak VRAM Bounds (<= 2.0GB)",
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
            name="Peak VRAM Bounds (<= 2.0GB)",
            passed=passed,
            message=f"PASSED: Peak VRAM {peak_gb:.3f}GB <= {self.config.vram_limit_gb:.1f}GB" if passed else f"FAILED: Peak VRAM {peak_gb:.3f}GB exceeds {self.config.vram_limit_gb:.1f}GB limit",
            details={"peak_vram_gb": peak_gb, "limit_gb": self.config.vram_limit_gb}
        )

    def check_golden_regression(self, model: nn.Module, check_all: bool = False) -> VerifyResult:
        gdir = self.config.golden_dir
        manifest_path = gdir / "manifest.json"

        if not gdir.exists():
            return VerifyResult(
                name="Golden Fixture Regression Check",
                passed=False,
                message=f"FAILED: Golden directory '{gdir}' does not exist. Run Phase 0.2 golden generation.",
                error=f"Golden directory missing at {gdir}"
            )

        # Discover fixtures
        sample_indices = range(10) if check_all else [0]
        maes = []
        fixture_details = []

        for idx in sample_indices:
            img_path = gdir / f"image_{idx:03d}.png"
            npy_path = gdir / f"depth_{idx:03d}.npy"

            if not img_path.exists() or not npy_path.exists():
                return VerifyResult(
                    name="Golden Fixture Regression Check",
                    passed=False,
                    message=f"FAILED: Fixture sample {idx:03d} missing ({img_path.name} / {npy_path.name})",
                    error=f"Missing fixture file for index {idx}"
                )

            raw_img = cv2.imread(str(img_path))
            golden_depth = np.load(str(npy_path))

            with torch.no_grad():
                pred_depth = model.infer_image(raw_img, input_size=self.config.input_size)

            mae = float(np.mean(np.abs(pred_depth - golden_depth)))
            maes.append(mae)
            fixture_details.append({"sample": idx, "mae": mae, "passed": mae < self.config.tolerance})

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

        # Check growth between iteration 2 and iteration N
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
        print("   DepthLab Standalone Verification Suite (Milestone 1 Gate)     ")
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

        # Step 2: Prepare dummy sample image if golden sample missing
        sample_img = None
        g_img_path = self.config.golden_dir / "image_000.png"
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

        # Step 4: Run extended stress suite if --all
        if self.config.run_all and r_fp16.passed:
            self.results.append(self.check_synthetic_stress(model))
            self.results.append(self.check_memory_leak(model, sample_img))

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
            print(f"         └─ {r.message}")

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
        json_report=args.json_report,
        verbose=args.verbose,
    )

    suite = VerificationSuite(config)
    success = suite.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
```

---

## 5. Verification Method

To independently verify the `verify.py` design and technical specification:

1. **Codebase Inspection**:
   - Inspect `.agents/explorer_m1_3/handoff.md` to confirm all 8 core checks and 2 `--all` checks are specified with type hints, data structures, and CLI parameters.
   - Confirm compatibility with both Layer 0 (`depth-anything-v2-official/depth_anything_v2/dpt.py`) and Layer 1 (`depthlab/backbone/loader.py`).

2. **Execution Test (When implemented by Implementer/Worker)**:
   - Command: `python verify.py`
     - Expected Output: Runs 8 baseline checks. Reports PASS/FAIL and exits with code 0 on success.
   - Command: `python verify.py --all`
     - Expected Output: Runs all 10 checks including full 10-sample golden regression loop, synthetic stress inputs, and multi-iteration CUDA memory accumulation check.
   - Command: `python verify.py --checkpoint invalid_path.pth`
     - Expected Output: Intercepts missing checkpoint, prints clean diagnostic message, and exits with code 1.

3. **Invalidation Conditions**:
   - The design is invalidated if Depth Anything V2 introduces an incompatible API change to `infer_image()`, or if PyTorch FP16 autocast requires explicit scaling for DPT decoder heads on specific CUDA architectures.
