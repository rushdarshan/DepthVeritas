# Handoff Report — Milestone 2 (Phase 0.3 Evaluation Reproduction & Validation Gate)

**Agent**: Worker 2 (`worker_m2`)  
**Working Directory**: `C:\Users\rushd\Downloads\prj-res\.agents\worker_m2`  
**Target Codebase**: `C:\Users\rushd\Downloads\prj-res`  
**Date**: 2026-07-21  

---

## 1. Observation

### 1.1 Deliverables Implemented & Verification Execution

The following deliverables were implemented and verified on the codebase:

1. **`docs/reproduction/environment.md`**: Created human-readable reproduction document matching explorer 2 design (`C:\Users\rushd\Downloads\prj-res\.agents\explorer_m2_2\handoff.md`), containing system hardware/software specs, provenance (git commit `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf`, checkpoint `depth_anything_v2_vits.pth` SHA256 `715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378`, metric.py SHA256 `8bfc953a8e923c4de41159adc99af2bb05e7cb29cca8fd1c209e30b442357fe0`), evaluation protocol, 7-metric reproduction comparison table within 1% margin, runtime (146.94 ms), peak VRAM (0.324 GB), and golden MAE (0.00e+00).
2. **`docs/reproduction/reproduction-report.yaml`**: Created machine-readable reproduction report metadata file containing all 13 mandatory ADR-002 §5 fields.
3. **`verify.py` Enhancement & Bug Fix**:
   - Added `compute_sha256()` function to calculate SHA256 checksums.
   - Implemented `_resolve_fixture_path()` helper method to resolve relative fixture paths relative to `gdir` (`self.config.golden_dir`) with fallbacks.
   - Updated `check_golden_regression()` to perform SHA256 fixture checksum checks against `fixture_manifest.json` and wrapped model inference / array operations in `try...except` handling.
   - Added `check_reproduction_report()` to validate report existence, SHA256 checksums of checkpoint and metric script, mandatory field presence, and 1% metric bounds.
   - Added CLI options `--reproduction-report` and `--check-report`.

### 1.2 Verification Outputs (Verbatim Command Results)

Running `python verify.py --all`:
```
==================================================================
   DepthLab Standalone Verification Suite (Milestone 2 Gate)     
==================================================================
Device      : cuda
Checkpoint  : C:\Users\rushd\Downloads\prj-res\checkpoints\depth_anything_v2_vits.pth
Golden Dir  : C:\Users\rushd\Downloads\prj-res\golden
Tolerance   : MAE < 1e-06
Run Mode    : Comprehensive (--all)
------------------------------------------------------------------

Verification Test Results:
------------------------------------------------------------------
[ PASS ] Checkpoint Loading
         +-- PASSED: Loaded vits (24.79M params) from 'depth_anything_v2_vits.pth'
[ PASS ] FP16 Mixed Precision Execution
         +-- PASSED: Execution succeeded under FP16/autocast (cuda)
[ PASS ] Output Shape Integrity
         +-- PASSED: Shape (1362, 2048) matches input ((1362, 2048))
[ PASS ] Non-Negative Depth Range
         +-- PASSED: Depth range [0.0000, 10.1250] >= 0.0
[ PASS ] Absence of NaNs / Infs
         +-- PASSED: Zero NaNs and Infs detected in output depth map
[ PASS ] Runtime Bounds (<= 1.0s @ 518x518)
         +-- PASSED: Mean runtime 145.8ms <= 1000ms
[ PASS ] Peak VRAM Bounds (<= 2.0GB)
         +-- PASSED: Peak VRAM 0.324GB <= 2.0GB
[ PASS ] Golden Fixture Regression Check
         +-- PASSED: Golden MAE (max=0.00e+00, mean=0.00e+00) < tolerance 1.0e-06
[ PASS ] Synthetic Extreme Input Stress Test (--all)
         +-- PASSED: Zero NaNs/Infs produced on synthetic black, white, and noise inputs
[ PASS ] Multi-Iteration Memory Accumulation Check (--all)
         +-- PASSED: Memory growth over 20 iterations is 0.00MB <= 1.0MB
[ PASS ] Phase 0.3 Reproduction Report Check
         +-- PASSED: Report fields valid & metrics match published baseline within 1% (AbsRel=0.0831 <= 0.0838, d1=0.9248 >= 0.9158, RMSE=0.3654 <= 0.3686)
------------------------------------------------------------------
OVERALL STATUS: ALL CHECKS PASSED (VERIFICATION SUCCESS)
==================================================================
```

Testing custom `--golden-dir` with corrupted array (`python verify.py --golden-dir scratch/golden_custom`):
```
[ FAIL ] Golden Fixture Regression Check
         +-- FAILED: SHA256 mismatch for depth npy 'demo01' (demo01_depth.npy)
------------------------------------------------------------------
OVERALL STATUS: VERIFICATION FAILED (REGRESSION OR BOUND ERROR)
```

---

## 2. Logic Chain

1. **Reproduction Documentation**: `docs/reproduction/environment.md` and `docs/reproduction/reproduction-report.yaml` satisfy ADR-001, ADR-002 §5, and ROADMAP.md Phase 0.3 requirements by capturing exact system specs, commit hashes, SHA256 checksums, evaluation protocol parameters, and metric comparison tables within 1% bounds.
2. **Fixture Path Resolution Fix**: In `verify.py`, using `_resolve_fixture_path()` allows custom `--golden-dir` paths (such as `scratch/golden_custom`) to be resolved relative to `gdir` first, preventing fallback failures or hardcoded `PROJECT_ROOT` prefixing.
3. **SHA256 & Exception Hardening**: Checking `image_sha256` and `depth_npy_sha256` ensures fixture integrity before running inference, while `try...except` catches corrupted data without crashing the runner.
4. **Validation Gate Enforcement**: `check_reproduction_report()` validates that all 13 mandatory ADR-002 fields are present, checksums match, and reproduced metrics fall within the required 1% margin.

---

## 3. Caveats

- **No caveats**: All implementation changes were minimal, fully verified, and tested against clean and edge-case inputs.

---

## 4. Conclusion

Milestone 2 deliverables (Phase 0.3 Evaluation Reproduction & Validation Gate) are fully implemented and verified. All 11 verification checks pass in `verify.py --all`.

---

## 5. Verification Method

To independently verify this implementation:

1. **Run Full Verification Suite**:
   ```bash
   python verify.py --all
   ```
   *Expected Output*: `OVERALL STATUS: ALL CHECKS PASSED (VERIFICATION SUCCESS)` (exit code 0).

2. **Verify Reproduction Report Checking CLI**:
   ```bash
   python verify.py --check-report --reproduction-report docs/reproduction/reproduction-report.yaml
   ```
   *Expected Output*: `[ PASS ] Phase 0.3 Reproduction Report Check` (exit code 0).

3. **Verify Custom Golden Directory Path Resolution & Checksum Validation**:
   ```bash
   python -c "import shutil; shutil.copytree('golden', 'scratch/test_gdir', dirs_exist_ok=True)"
   python verify.py --golden-dir scratch/test_gdir
   python -c "import shutil; shutil.rmtree('scratch/test_gdir')"
   ```
   *Expected Output*: `[ PASS ] Golden Fixture Regression Check` (exit code 0).
