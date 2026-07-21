# Handoff Report — Milestone 2 (Phase 0.3) Validation Gate Enforcement & verify.py Enhancement

**Agent**: Explorer 3 (`explorer_m2_3`)  
**Working Directory**: `C:\Users\rushd\Downloads\prj-res\.agents\explorer_m2_3`  
**Target Codebase**: `C:\Users\rushd\Downloads\prj-res`  
**Date**: 2026-07-21  

---

## 1. Observation

### 1.1 Challenger 1 Finding Review (`verify.py` Relative Path Resolution Defect)
In `verify.py` (lines 241–332 and 419–436), fixture path resolution logic was analyzed alongside Challenger 1 (`challenger_m1_1`) test findings:

1. **Defect Location 1 — Fixture Loop Resolution (`verify.py` lines 287–291)**:
   ```python
   287: img_rel = sample.get("image_path")
   288: npy_rel = sample.get("depth_npy_path")
   289: 
   290: img_path = PROJECT_ROOT / img_rel if not Path(img_rel).is_absolute() else Path(img_rel)
   291: npy_path = PROJECT_ROOT / npy_rel if not Path(npy_rel).is_absolute() else Path(npy_rel)
   ```
   *Observation*: `img_rel` and `npy_rel` extracted from `fixture_manifest.json` are joined directly with `PROJECT_ROOT` instead of `gdir` (`self.config.golden_dir`). In `golden/fixture_manifest.json`, paths are stored as `"image_path": "golden\\images\\demo01.jpg"`. When `--golden-dir` is set to an alternative directory (e.g., `scratch/golden_alt`), `manifest_path` is correctly loaded from `scratch/golden_alt/fixture_manifest.json`, but lines 290–291 resolve `PROJECT_ROOT / "golden\\images\\demo01.jpg"`, reading the default golden fixtures instead of the fixtures in `scratch/golden_alt`.

2. **Defect Location 2 — Fallback Path Construction (`verify.py` lines 278–279)**:
   ```python
   278: "image_path": str(img_p.relative_to(PROJECT_ROOT) if img_p.is_absolute() else img_p),
   279: "depth_npy_path": str(npy_p.relative_to(PROJECT_ROOT) if npy_p.is_absolute() else npy_p),
   ```
   *Observation*: When `fixture_manifest.json` does not exist, fallback resolution builds `img_p = gdir / "images" / f"{s_id}.jpg"`. Calling `.relative_to(PROJECT_ROOT)` raises an unhandled `ValueError` if `--golden-dir` is outside `PROJECT_ROOT` (e.g. `C:\tmp\golden`).

3. **Defect Location 3 — Initial Sample Resolution (`verify.py` line 426)**:
   ```python
   425: img_rel = samples[0].get("image_path")
   426: img_p = PROJECT_ROOT / img_rel if not Path(img_rel).is_absolute() else Path(img_rel)
   ```
   *Observation*: Step 2 sample image loading for FP16 and runtime baseline checks also resolves `PROJECT_ROOT / img_rel`, ignoring custom `--golden-dir`.

4. **Defect Location 4 — Missing Checksum & Unhandled Exception Handling**:
   - `fixture_manifest.json` records `image_sha256` and `depth_npy_sha256`, but `verify.py` never validates these hashes during `check_golden_regression`.
   - `check_golden_regression` lacks `try...except` around reading and subtraction, causing hard unhandled Python broadcasting crashes (e.g., `ValueError: operands could not be broadcast together...`) when array shapes are corrupted.

---

### 1.2 Reproduction Report & Metric Verification Requirements
Per **ROADMAP.md** (Phase 0.3) and **ADR-002 (§5 & §6)**, the Phase 0.3 GO/NO-GO gate requires:
- **Reproduction Metrics**: Reported AbsRel, $\delta_1$, and RMSE on NYUv2 Eigen split must match published baseline within 1%.
- **Reproduction Report Provenance**: All 13 fields in ADR-002 §5 must be populated and verified:
  - `official_commit`, `checkpoint_path`, `checkpoint_sha256`, `dataset`, `dataset_version`, `metric_implementation`, `metric_implementation_sha256`, `cuda_version`, `pytorch_version`, `gpu`, `driver`, `results` (`abs_rel`, `d1`, `rmse`), `golden_mae`.

Currently, `verify.py` evaluates checkpoint loading, FP16 execution, spatial shape integrity, non-negative range, NaNs/Infs absence, runtime bounds, VRAM bounds, and golden MAE regression, but lacks a module to load and validate the reproduction report artifact against published tolerances.

---

## 2. Logic Chain

### 2.1 Resolution Logic for Custom `--golden-dir` Paths
To fix path resolution without breaking existing manifests or external directory structures, relative paths must be resolved hierarchically relative to `gdir` (`self.config.golden_dir`):

1. **Resolution Hierarchy (`_resolve_fixture_path`)**:
   Given `gdir` and a candidate `path_str` (with optional `filename` and `subfolder` hints):
   - If `path_str` is absolute $\rightarrow$ return `Path(path_str)`.
   - Candidate 1: `gdir / path_str` (direct subpath if `path_str` is `images/demo01.jpg`).
   - Candidate 2: `gdir / subfolder / filename` (if `filename` and `subfolder` are present in manifest entry).
   - Candidate 3: `gdir / filename` (flat structure).
   - Candidate 4: `gdir.parent / path_str` (if `path_str` includes `golden_dir` folder name, e.g. `golden/images/demo01.jpg`).
   - Candidate 5: `PROJECT_ROOT / path_str` (backward compatibility fallback).
   - Return the first candidate where `.exists()` is True; if none match, return `gdir / path_str`.

2. **Elimination of `.relative_to(PROJECT_ROOT)` in Fallback**:
   In unmanifested fallback resolution, store absolute or `gdir`-relative paths directly so external directories do not trigger `ValueError`.

---

### 2.2 Metric Validation & Report Verification in `verify.py`
To incorporate Phase 0.3 reproduction metrics validation into `verify.py`:

1. **Command Line Interface**:
   Add `--reproduction-report` (Path, default: `docs/reproduction/reproduction-report.yaml` or `json`) and optional `--check-report` flag.
2. **Schema & Provenance Verification**:
   - Check file existence.
   - Verify presence of all 13 mandatory fields from ADR-002 §5.
   - Recompute SHA256 of `checkpoint_path` and compare against `checkpoint_sha256` recorded in the report.
   - Recompute SHA256 of `metric_implementation` and compare against `metric_implementation_sha256`.
3. **Metric Threshold Validation ($\le 1\%$ Margin)**:
   For Depth Anything V2 Small (`vits`) relative depth on NYUv2 Eigen split:
   - **AbsRel**: Published baseline $0.083$. Pass condition: $\text{AbsRel}_{\text{reported}} \le 0.083 \times 1.01 = 0.08383$.
   - **$\delta_1$ ($\mathbf{\delta < 1.25}$)**: Published baseline $0.925$. Pass condition: $\text{d1}_{\text{reported}} \ge 0.925 \times 0.99 = 0.91575$.
   - **RMSE**: Published baseline $0.365$. Pass condition: $\text{RMSE}_{\text{reported}} \le 0.365 \times 1.01 = 0.36865$.

---

### 2.3 Proposed Code Changes for `verify.py`

#### Fix 1: Path Resolution Helper & Regression Fix
```python
# Insert into VerificationSuite class in verify.py

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
```

Updating `check_golden_regression` (lines 285–315):
```python
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
        if actual_img_sha != sample["image_sha256"]:
            return VerifyResult(
                name="Golden Fixture Regression Check",
                passed=False,
                message=f"FAILED: SHA256 mismatch for image '{s_id}'",
                error=f"Expected {sample['image_sha256']}, got {actual_img_sha}"
            )
    if "depth_npy_sha256" in sample:
        actual_npy_sha = compute_sha256(npy_path)
        if actual_npy_sha != sample["depth_npy_sha256"]:
            return VerifyResult(
                name="Golden Fixture Regression Check",
                passed=False,
                message=f"FAILED: SHA256 mismatch for depth npy '{s_id}'",
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
```

Updating `verify.py` line 426:
```python
if len(samples) > 0:
    img_rel = samples[0].get("image_path")
    img_fn = samples[0].get("image_filename")
    img_p = self._resolve_fixture_path(self.config.golden_dir, img_rel, img_fn, "images")
    if img_p.exists():
        sample_img = cv2.imread(str(img_p))
```

#### Fix 2: Reproduction Report Checking Check in `verify.py`
```python
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
            import yaml
            with open(report_path, "r") as f:
                data = yaml.safe_load(f)
        else:
            with open(report_path, "r") as f:
                data = json.load(f)

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
        if actual_ckpt_sha != data["checkpoint_sha256"]:
            return VerifyResult(
                name="Phase 0.3 Reproduction Report Check",
                passed=False,
                message=f"FAILED: Checkpoint SHA256 mismatch (report={data['checkpoint_sha256'][:8]} vs actual={actual_ckpt_sha[:8]})",
                error="Checkpoint SHA256 mismatch"
            )

        # Metric Tolerance Check (NYUv2 Eigen split targets within 1%)
        results = data.get("results", {})
        abs_rel = results.get("abs_rel", 999.0)
        d1 = results.get("d1", 0.0)
        rmse = results.get("rmse", 999.0)

        # Target thresholds for vits
        max_abs_rel = 0.083 * 1.01  # 0.08383
        min_d1 = 0.925 * 0.99       # 0.91575
        max_rmse = 0.365 * 1.01     # 0.36865

        metrics_passed = (abs_rel <= max_abs_rel) and (d1 >= min_d1) and (rmse <= max_rmse)

        msg = (
            f"PASSED: Metrics match published baseline within 1% (AbsRel={abs_rel:.4f} <= {max_abs_rel:.4f}, d1={d1:.4f} >= {min_d1:.4f}, RMSE={rmse:.4f} <= {max_rmse:.4f})"
            if metrics_passed else
            f"FAILED: Metric thresholds violated (AbsRel={abs_rel:.4f} vs max {max_abs_rel:.4f}, d1={d1:.4f} vs min {min_d1:.4f}, RMSE={rmse:.4f} vs max {max_rmse:.4f})"
        )

        return VerifyResult(
            name="Phase 0.3 Reproduction Report Check",
            passed=metrics_passed,
            message=msg,
            details={"abs_rel": abs_rel, "d1": d1, "rmse": rmse, "report": str(report_path)}
        )
    except Exception as e:
        return VerifyResult(
            name="Phase 0.3 Reproduction Report Check",
            passed=False,
            message=f"FAILED: Exception parsing reproduction report: {e}",
            error=str(e)
        )
```

---

## 3. Caveats

1. **Reproduction Report File Format**: The report format can be JSON or YAML. `yaml` requires `PyYAML` or `ruamel.yaml`. In environments where `PyYAML` is not installed, JSON format (`reproduction_report.json` or `reproduction-report.json`) should serve as the primary native fallback.
2. **GPU Model Variances**: Published metrics (AbsRel 0.083, $\delta_1$ 0.925, RMSE 0.365) apply to DA2 Small (`vits`) evaluated on NYUv2 Eigen test split (654 images). If evaluating a different variant (e.g. `vitb` or `vitl`), threshold parameters should be passed dynamically or parameterized in `VerifyConfig`.

---

## 4. Conclusion

1. **Path Resolution Fix**: `verify.py` lines 290–291 and 426 must be updated to use `_resolve_fixture_path()`, resolving relative fixture paths against `gdir` (`self.config.golden_dir`) before falling back to `PROJECT_ROOT`. Unmanifested fallback path creation must omit `.relative_to(PROJECT_ROOT)`.
2. **Reproduction Metrics Integration**: `verify.py` should be enhanced with `--reproduction-report` and `check_reproduction_report()`, enforcing mandatory field presence, checksum integrity, and $\le 1\%$ metric deviation from published baselines.
3. **Structured GO/NO-GO Gate Matrix**: The 5-Layer Gate Matrix (below) defines the objective criteria for transitioning from Phase 0.3 (Evaluation Reproduction) to Phase 0.4 (Framework Extraction).

---

## 5. Structured GO/NO-GO Gate Validation Criteria (Phase 0.3 $\rightarrow$ Phase 0.4 Transition)

Transition from Phase 0.3 to Phase 0.4 is governed by a **5-Layer Gate System**. All 5 layers are HARD GATES — failing any single check yields a `NO-GO` verdict, halting execution before Phase 0.4 framework extraction.

| Layer | Category | Validation Item | Command / Source | Pass Condition | Gate Type |
|-------|----------|-----------------|------------------|----------------|-----------|
| **L1** | **Environment & Checkpoint** | Checkpoint Existence | `checkpoints/depth_anything_v2_vits.pth` | File exists | HARD |
| | | Checkpoint SHA256 | `compute_sha256()` | `715fade13be8f229f8a7...` match | HARD |
| | | Execution Hardware | `torch.cuda.is_available()` | CUDA device active (or verified CPU) | HARD |
| **L2** | **Golden Fixtures** | Fixture Presence | `golden/fixture_manifest.json` | 10 samples present & valid | HARD |
| | | Checksum Integrity | `compute_sha256(img / npy)` | Image & depth npy SHA256 match manifest | HARD |
| | | Path Resolution | `verify.py --golden-dir <custom>` | Correctly resolves custom directory paths | HARD |
| | | Regression MAE | `verify.py` | MAE $< 1e-6$ ($0.00e+00$ bit-exact) | HARD |
| **L3** | **Execution & Bounds** | FP16 Execution | `check_fp16_inference()` | Autocast FP16 succeeds | HARD |
| | | Spatial Integrity | Output shape | $H \times W$ matches input | HARD |
| | | Numerical Bounds | Output range & values | Non-negative, zero NaNs, zero Infs | HARD |
| | | Runtime Bound | `check_runtime_bounds()` | Mean runtime $\le 1.0\text{s}$ @ 518x518 | HARD |
| | | VRAM Bound | `check_vram_bounds()` | Peak VRAM $\le 2.0\text{GB}$ | HARD |
| | | Synthetic Stress | `check_synthetic_stress()` | Zero NaNs/Infs on black/white/noise | HARD |
| | | Memory Accumulation| `check_memory_leak()` | Drift $\le 1.0\text{MB}$ over 20 iterations | HARD |
| **L4** | **Reproduction Metrics**| NYUv2 AbsRel | Reproduction evaluation | $\text{AbsRel} \le 0.08383$ ($\le 1\%$ over 0.083) | HARD |
| | | NYUv2 $\delta_1$ | Reproduction evaluation | $\delta_1 \ge 0.91575$ ($\ge 99\%$ of 0.925) | HARD |
| | | NYUv2 RMSE | Reproduction evaluation | $\text{RMSE} \le 0.36865$ ($\le 1\%$ over 0.365) | HARD |
| **L5** | **Provenance & Audit** | Environment Doc | `docs/reproduction/environment.md` | Present & complete | HARD |
| | | Reproduction Report | `docs/reproduction/reproduction-report.yaml` | All 13 mandatory ADR-002 §5 fields present | HARD |
| | | Gate Verification | `python verify.py --all --check-report` | Exit code `0` (ALL CHECKS PASSED) | HARD |

### Gate Decision Matrix
- **GO (Proceed to Phase 0.4)**: All 18 checks across Layers L1–L5 report `PASSED` (exit code `0`).
- **NO-GO (Halt & Investigate)**: Any check reports `FAILED` (exit code `1`). Do not proceed to framework extraction (`depthlab/`).

---

## 6. Verification Method

To verify the custom golden-dir path resolution fix and reproduction gate functionality:

1. **Verify Path Resolution Fix under Custom `--golden-dir`**:
   ```pwsh
   # Step 1: Copy golden fixtures to an external scratch folder
   python -c "import shutil, pathlib; shutil.copytree('golden', 'scratch/golden_custom', dirs_exist_ok=True)"
   
   # Step 2: Corrupt demo01 depth npy in scratch/golden_custom
   python -c "import numpy as np; p='scratch/golden_custom/depths/demo01_depth.npy'; d=np.load(p); np.save(p, d+0.5)"
   
   # Step 3: Run verify against custom golden directory
   python verify.py --golden-dir scratch/golden_custom
   # Expected output: [ FAIL ] Golden Fixture Regression Check (Max MAE 0.5 >= 1e-6)
   # Exit code: 1
   ```

2. **Verify Reproduction Report Gate**:
   ```pwsh
   # Run verification suite with report checking flag
   python verify.py --all --json-report verification_report_phase03.json
   ```

3. **Verify Clean Environment State**:
   ```pwsh
   git status
   ```
