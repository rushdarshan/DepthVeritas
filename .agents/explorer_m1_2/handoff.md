# Handoff Report: Golden Reference Fixtures & Verification Strategy for Milestone 1

**Agent ID:** Explorer 2 (`explorer_m1_2`)  
**Milestone:** Milestone 1 — Baseline & Parity Verification  
**Target Directory:** `C:\Users\rushd\Downloads\prj-res`  
**Working Directory:** `C:\Users\rushd\Downloads\prj-res\.agents\explorer_m1_2`  
**Date:** 2026-07-21  

---

## 1. Observation

### 1.1 Project Documentation & Governance
- **`PROJECT.md` (lines 10-12, 31, 48):** Milestone 1 specifies Phase 0.1 & 0.2: validate environment, generate golden fixtures in `golden/`, build `verify.py`.
- **`ROADMAP.md` (lines 21-34, 40-42):** Phase 0.2 mandates running official `run.py` on 10 sample images (unmodified Layer 0) and saving depth outputs to `golden/` fixtures. Phase 0.3 mandates `verify.py` reporting `MAE < 1e-6` against golden fixtures.
- **`docs/adr/001-depthlab-architecture.md` & `002-depthlab-api-stability.md`:**
  - Layer 0 (`depth-anything-v2-official/`) is **immutable** (never modified).
  - Layer 1 adapter (`DA2Backbone`, `OFFICIAL_TRANSFORM`) wraps Layer 0.
  - `golden/` resides at project root, version controlled.
- **`openspec/changes/depthlab-initial-scaffold/specs/reproduction-protocol/spec.md` (lines 11-18, 27-38):** Explicit requirement: `golden/` must contain 10 entries with input image, official depth output (`.npy` raw float array and `.png` visual map), and a `manifest.json`.

### 1.2 Upstream Codebase & Assets (`depth-anything-v2-official/`)
- **Sample Images Location:** `depth-anything-v2-official/assets/examples/` contains 20 sample images (`demo01.jpg` through `demo20.jpg`).
- **Image Characteristics:**
  - `demo01.jpg` (1362 × 2048 × 3)
  - `demo02.jpg` (1362 × 2047 × 3)
  - `demo03.jpg` (1295 × 1990 × 3)
  - `demo04.jpg` (1041 × 1600 × 3)
  - `demo05.jpg` (1332 × 2048 × 3)
  - `demo06.jpg` (1362 × 2048 × 3)
  - `demo07.jpg` (1498 × 2252 × 3)
  - `demo08.jpg` (425 × 640 × 3)
  - `demo09.jpg` (1487 × 2236 × 3)
  - `demo10.jpg` (1332 × 2048 × 3)
- **`run.py` Invocation & Behavior:**
  - CLI usage: `python depth-anything-v2-official/run.py --encoder vits --img-path assets/examples/demo01.jpg --outdir vis_depth --pred-only --input-size 518`
  - Internal execution flow (`run.py:35-68`):
    1. Instantiates `DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])`.
    2. Loads checkpoint `checkpoints/depth_anything_v2_vits.pth`.
    3. Reads image via `raw_image = cv2.imread(filename)`.
    4. Calls `depth = depth_anything.infer_image(raw_image, args.input_size)`.
    5. Normalizes `depth` to `0..255 uint8` for `cv2.imwrite` visualization.
- **`dpt.py:186-194` (`infer_image` return type):**
  - `infer_image` returns an unquantized `np.float32` 2D NumPy array of shape `(H, W)` matching the raw image height and width.

### 1.3 Execution Environment Verification
- PyTorch version: `2.12.1+cu126`
- CUDA status: `True` (1 CUDA GPU device active)

---

## 2. Logic Chain

1. **Selection of 10 Sample Images:**
   - `assets/examples/demo01.jpg` through `demo10.jpg` are ideal candidates.
   - *Rationale:* They are included directly in the official git repository clone, requiring no external dataset downloads. They cover diverse real-world scenes (indoor, outdoor, urban, close-up) and varied resolutions (from 425×640 to 1498×2252) and aspect ratios.

2. **Precision Requirement vs Output Format (`.npy` vs `.png`):**
   - Goal: `verify.py` must test predicted depth maps against golden fixtures with tolerance `MAE < 1e-6`.
   - `run.py` outputs 8-bit quantized visual PNG images (`0..255`). Quantizing floating point predictions to 256 discrete integer levels introduces quantization steps of `~1 / 255 = 0.00392`.
   - An error of `0.00392` is 3,920 times larger than the required tolerance `1e-6`.
   - *Conclusion:* Visual PNG images CANNOT be used for high-precision numerical regression testing (`MAE < 1e-6`). The golden reference fixtures MUST store the raw, unquantized `np.float32` floating point arrays returned by `infer_image()` as binary `.npy` files (`np.save()`).

3. **Golden Fixture Directory Structure (`golden/`):**
   - Structure proposed:
     ```
     golden/
     ├── fixture_manifest.json
     ├── images/
     │   ├── demo01.jpg
     │   ├── demo02.jpg
     │   └── ... (up to demo10.jpg)
     ├── depths/
     │   ├── demo01_depth.npy
     │   ├── demo02_depth.npy
     │   └── ... (up to demo10_depth.npy)
     └── vis/
         ├── demo01_vis.png
         ├── demo02_vis.png
         └── ... (up to demo10_vis.png)
     ```
   - Manifest Schema (`fixture_manifest.json`):
     - Global metadata: `version`, `model_variant` (`vits`), `checkpoint_name`, `checkpoint_sha256`, `input_size` (518), `torch_version`, `cuda_available`, `num_samples` (10).
     - Per-sample records: `sample_id`, `image_file`, `depth_npy`, `depth_vis_png`, `height`, `width`, `min_depth`, `max_depth`, `mean_depth`, `sha256_npy`.

4. **Generation Strategy:**
   - To strictly observe Layer 0 immutability while capturing high-precision `.npy` arrays, a dedicated generator script (`scripts/generate_golden.py`) will:
     1. Load `checkpoints/depth_anything_v2_vits.pth` into official `DepthAnythingV2(encoder='vits')`.
     2. Process `demo01.jpg` .. `demo10.jpg`.
     3. Call `infer_image(raw_image, input_size=518)` to obtain `float32` depth map `(H, W)`.
     4. Save raw depth array as `golden/depths/<id>_depth.npy`.
     5. Save colormapped visual depth map as `golden/vis/<id>_vis.png`.
     6. Copy input image into `golden/images/<id>.jpg`.
     7. Compute SHA256 checksums and statistics, and write `golden/fixture_manifest.json`.

5. **`verify.py` Testing Mechanism:**
   - When executed, `verify.py` will:
     1. Load model checkpoint `checkpoints/depth_anything_v2_vits.pth`.
     2. Parse `golden/fixture_manifest.json`.
     3. For each of the 10 samples:
        - Read input image using `cv2.imread(img_path)`.
        - Run `pred_depth = model.infer_image(raw_image, input_size=518)`.
        - Load `golden_depth = np.load(depth_npy_path)`.
        - Assert `pred_depth.shape == golden_depth.shape`.
        - Assert `np.all(pred_depth >= 0)`.
        - Assert `not np.isnan(pred_depth).any()` and `not np.isinf(pred_depth).any()`.
        - Calculate `mae = np.mean(np.abs(pred_depth - golden_depth))`.
        - Assert `mae < 1e-6`.
     4. Exit with status 0 on pass, or non-zero on failure with detailed diagnostic reporting.

---

## 3. Caveats

1. **Hardware / Determinism Differences:**
   - Running `infer_image` in FP32 on CPU vs CUDA GPU may yield minor floating-point differences (~1e-7 to 1e-6) due to CUDA atomic operations / cuDNN kernels. Golden fixtures generated on CUDA FP32 will match CUDA FP32 inference within machine epsilon (< 1e-7).
   - FP16 (half precision) inference MUST NOT be used for generating or testing golden fixtures because FP16 precision is limited to ~1e-3.
2. **Image IO Library Dependency:**
   - Preprocessing depends on OpenCV (`cv2.imread` returning BGR unit8 array). Using PIL (`PIL.Image.open`) introduces subtle RGB/BGR conversion and JPEG decoding variances that exceed 1e-6 tolerance. OpenCV MUST be used across all scripts.
3. **Checkpoint Availability:**
   - `checkpoints/depth_anything_v2_vits.pth` must be downloaded to `checkpoints/` prior to running fixture generation and `verify.py`.

---

## 4. Conclusion

1. **Sample Image Selection:** `depth-anything-v2-official/assets/examples/demo01.jpg` through `demo10.jpg` are selected as the 10 reference sample images for Milestone 1 golden fixtures.
2. **Fixture Format:** Fixtures MUST include `.npy` raw `np.float32` binary depth arrays alongside `.png` visualizations and a `fixture_manifest.json` metadata index.
3. **Official Invocations:** Official `run.py` provides the canonical invocation parameter baseline (`--encoder vits --input-size 518`).
4. **Verification Criteria:** `verify.py` can reliably assert `MAE < 1e-6` when comparing FP32 model inference against stored `.npy` float32 golden fixtures.

---

## 5. Verification Method

To verify these findings and plans independently:

1. **Inspect sample images:**
   ```bash
   python -c "import glob, cv2; [print(f, cv2.imread(f).shape) for f in sorted(glob.glob('depth-anything-v2-official/assets/examples/demo*.jpg'))[:10]]"
   ```
2. **Confirm float32 precision return type of `infer_image`:**
   Inspect `depth-anything-v2-official/depth_anything_v2/dpt.py:186-194`.
3. **Validate openspec contract:**
   Inspect `openspec/changes/depthlab-initial-scaffold/specs/reproduction-protocol/spec.md`.
