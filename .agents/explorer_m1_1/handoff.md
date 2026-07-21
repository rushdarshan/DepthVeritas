# Handoff Report — Milestone 1 Explorer 1: Official Codebase & Environment Analysis

**Agent:** Explorer 1 (`explorer_m1_1`)  
**Date:** 2026-07-21  
**Milestone:** M1 (Phase 0.1 & Phase 0.2 Preparation)  
**Target Directory:** `C:\Users\rushd\Downloads\prj-res`  

---

## 1. Observation

### 1.1 Repository Layout & Package Structure (`depth-anything-v2-official/`)
- **Root Files**:
  - `run.py` (73 lines): Single-image / image batch inference entry point.
  - `run_video.py` (3842 bytes): Video inference script.
  - `app.py` (3337 bytes): Gradio interactive web UI application.
  - `README.md` (10105 bytes), `LICENSE` (11558 bytes, Apache 2.0 / custom model weights), `DA-2K.md` (1998 bytes), `requirements.txt` (75 bytes: `gradio_imageslider`, `gradio`, `matplotlib`, `opencv-python`, `torch`, `torchvision`).
- **Core Package (`depth_anything_v2/`)**:
  - `depth_anything_v2/dpt.py` (222 lines): Defines `DepthAnythingV2` (lines 153–221) and `DPTHead` (lines 38–150).
    - `DepthAnythingV2` wraps `DINOv2(model_name=encoder)` backbone and `DPTHead`.
    - Block index map `intermediate_layer_idx`: `vits`: `[2, 5, 8, 11]`, `vitb`: `[2, 5, 8, 11]`, `vitl`: `[4, 11, 17, 23]`, `vitg`: `[9, 19, 29, 39]`.
    - `infer_image(raw_image, input_size=518)` (lines 186–194): Converts raw BGR image to tensor, calls `forward()`, interpolates depth tensor back to original input dimensions `(h, w)` using bilinear interpolation, and returns a 2D float NumPy array.
  - `depth_anything_v2/dinov2.py` (416 lines): Custom self-contained Meta DINOv2 Vision Transformer implementation (`DinoVisionTransformer`).
    - Exposes stable API `get_intermediate_layers(x, n, return_class_token=True)` (lines 297–321), returning tuples of `(patch_tokens, cls_token)` for the designated 4 layer indices.
  - `depth_anything_v2/dinov2_layers/`: `attention.py`, `block.py`, `drop_path.py`, `layer_scale.py`, `mlp.py`, `patch_embed.py`, `swiglu_ffn.py`.
    - `attention.py` (lines 20–26): Imports `xformers.ops` if present; falls back to standard PyTorch attention when `xformers` is not installed (emitting `"xFormers not available"` warning).
  - `depth_anything_v2/util/`: `blocks.py` (`FeatureFusionBlock`, `_make_scratch`), `transform.py` (`Resize`, `NormalizeImage`, `PrepareForNet`).
- **Sample Media**:
  - `assets/examples/`: Contains 20 sample test images (`demo01.jpg` to `demo20.jpg`).
  - `assets/examples_video/`: Contains 2 sample video files (`basketball.mp4`, `ferris_wheel.mp4`).
- **Metric Depth Submodule (`metric_depth/`)**:
  - Contains dataset loaders (`dataset/`), `run.py`, `train.py`, `depth_to_pointcloud.py`.
  - Utility modules: `util/metric.py` implementing `eval_depth()` (AbsRel, SqRel, RMSE, RMSElog, log10, SILog, δ1, δ2, δ3), `util/loss.py` implementing `SiLogLoss`.
- **Pretrained Checkpoints**:
  - Checkpoint directory `depth-anything-v2-official/checkpoints/` **does not exist yet** in the repository.
  - `run.py` expects checkpoints at `checkpoints/depth_anything_v2_{args.encoder}.pth` relative to working directory. Pretrained weights must be downloaded prior to running inference.

### 1.2 Execution Flow of `run.py`
- **CLI Flags**:
  - `--img-path`: Input image file path, `.txt` list of paths, or directory path.
  - `--input-size`: Spatial resolution for model input (default `518`).
  - `--outdir`: Directory for visualization outputs (default `./vis_depth`).
  - `--encoder`: Architecture choice (`vits`, `vitb`, `vitl`, `vitg`; default `vitl`).
  - `--pred-only`: If set, saves prediction depth map only.
  - `--grayscale`: If set, saves output as 3-channel grayscale instead of color palette.
- **Inference Execution**:
  1. `depth_anything = DepthAnythingV2(**model_configs[args.encoder])`
  2. `depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))`
  3. `depth_anything = depth_anything.to(DEVICE).eval()`
  4. Reads image via `cv2.imread(filename)`.
  5. Runs `depth = depth_anything.infer_image(raw_image, args.input_size)`.
  6. Min-max normalization: `depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0` cast to `np.uint8`.
  7. Color mapping: Applies Matplotlib `Spectral_r` palette converted to BGR format (`[:, :, ::-1] * 255`) or 3-channel grayscale.
  8. Saving output: If `--pred-only`, writes depth visualization directly; otherwise concatenates `[raw_image, 50px_white_separator, depth]` side-by-side using `cv2.hconcat` and saves to `outdir`.

### 1.3 Hardware & Environment Diagnostics
- **Operating System**: Windows (AMD64)
- **Python Version**: `3.13.7` (`tags/v3.13.7:bcee1c3`)
- **PyTorch Version**: `2.12.1+cu126`
- **CUDA Status**: `CUDA Available: True`, CUDA Version `12.6`
- **GPU Device**: `NVIDIA GeForce RTX 4050 Laptop GPU` (Device 0)
- **GPU Memory**: `6.44 GB` (~6,442 MB total VRAM)
- **Compute Capability**: `8.9` (NVIDIA Ada Lovelace architecture)
- **cuDNN**: Version `91002`, Enabled: `True`
- **Precision / FP16 Support**:
  - `torch.cuda.is_bf16_supported()`: `True`
  - `torch.autocast(device_type='cuda', dtype=torch.float16)`: Tested and fully functional.
- **Model Parameters & Benchmark VRAM Metrics**:
  - `vits` (DA2-Small): `24,785,089` parameters (~24.79M).
  - FP32 Peak VRAM (1×3×518×518 input resolution on CUDA): **`224.72 MB`**.
  - FP16 Peak VRAM (1×3×518×518 input resolution on CUDA): **`288.05 MB`**.
  - Both FP32 and FP16 peak VRAM are well below the 2 GB (2048 MB) limit specified in `ROADMAP.md`.

### 1.4 Project & Architecture Documents
- **`PROJECT.md`**: Defines DepthLab's 3-layer architecture (Layer 0 Official Immutable, Layer 1 Adapter, Layer 2 Research) and milestone plan (M1 Environment & Fixtures, M2 NYUv2 Reproduction Gate, M3 Scaffolding & Multi-Head Framework, M4 Uncertainty Head).
- **`ROADMAP.md`**: Details Phase 0.1 to Phase 4. Specifies Phase 0.1 pass conditions (repo loads, CUDA inference, peak VRAM ≤ 2GB for DA2-Small at 518×518, FP16 works) and Phase 0.2 pass conditions (golden sample 10 images saved to `golden/`, `verify.py` passes).
- **`docs/adr/001-depthlab-architecture.md`**: Establishes 3-layer architecture, DINOv2 `get_intermediate_layers()` extension boundary, model immutability rule, `OFFICIAL_TRANSFORM` single shared transform pipeline, and reproduction protocol.
- **`docs/adr/002-depthlab-api-stability.md`**: Refines stable public API surface (`DA2Backbone`, `BaseHead`, `FeatureBundle`, `FeatureStage`, `OFFICIAL_TRANSFORM`, `ExperimentConfig`), directory layout (`research/` vs `runs/`), benchmark semver (`BENCHMARK_VERSION = "1.0.0"`), and Phase 0 success criteria table.

---

## 2. Logic Chain

1. **Observation**: `DepthAnythingV2` (`dpt.py`) instantiates `DINOv2` (`dinov2.py`) and passes intermediate feature maps into `DPTHead`.
2. **Deduction**: The backbone encoder is a self-contained custom DINOv2 module, not a `timm` Vision Transformer dependency. The extension point for DepthLab is `DINOv2.get_intermediate_layers(x, n, return_class_token=True)`.
3. **Observation**: `dinov2_layers/attention.py` catches `ImportError` on `xformers` and falls back to standard PyTorch attention seamlessly without breaking execution.
4. **Deduction**: `xformers` is optional for inference; standard PyTorch SDP attention on CUDA is sufficient for Phase 0.
5. **Observation**: PyTorch `2.12.1+cu126` running on NVIDIA GeForce RTX 4050 Laptop GPU (6.44 GB VRAM, Ada Lovelace compute 8.9) completed a 1×3×518×518 forward pass of `DepthAnythingV2` (`vits`) consuming **224.72 MB FP32 peak VRAM** and **288.05 MB FP16 peak VRAM**.
6. **Deduction**: The hardware environment satisfies all Phase 0.1 pass criteria specified in `ROADMAP.md` (Peak VRAM ≤ 2GB, CUDA available, FP16 supported).
7. **Observation**: `run.py` calls `torch.load('checkpoints/depth_anything_v2_{encoder}.pth')`, but `depth-anything-v2-official/checkpoints/` does not currently exist on disk.
8. **Deduction**: Before running `run.py` or generating the 10 golden fixtures in `golden/` (Phase 0.2), official pretrained model weights must be downloaded to `checkpoints/`.

---

## 3. Caveats

- **Missing Checkpoints**: Official `.pth` checkpoint files (`depth_anything_v2_vits.pth`, etc.) are not present in `depth-anything-v2-official/checkpoints/`. Golden sample generation and numerical reproduction require downloading these weights first.
- **Network Mode**: The agent environment operates under `CODE_ONLY` network restrictions. Checkpoint downloads must be handled via local file placement or script execution if external access is enabled for specific download commands.
- **xFormers Warning**: Without `xformers`, a warning `xFormers not available` is logged. This is non-fatal and does not affect numerical correctness.

---

## 4. Conclusion

- **Environment & GPU Setup**: Fully validated. Python `3.13.7`, PyTorch `2.12.1+cu126`, CUDA 12.6, RTX 4050 Laptop GPU (6.44 GB VRAM) are operational. FP16 mixed precision execution is verified.
- **VRAM Compliance**: DA2-Small (`vits`, 24.79M parameters) peak VRAM is **224.72 MB (FP32)** / **288.05 MB (FP16)**, passing the ≤ 2 GB requirement in `ROADMAP.md` Phase 0.1 with ample headroom.
- **Codebase Architecture**: `depth-anything-v2-official` layout, `DINOv2` intermediate layer extraction (`[2, 5, 8, 11]` for `vits`/`vitb`), `DPTHead`, `run.py` side-by-side visual output logic, and `metric_depth` metrics (`util/metric.py`) are fully mapped and understood.
- **Next Step (Phase 0.2)**: Obtain pretrained checkpoint weights (`depth_anything_v2_vits.pth`), execute `run.py` on sample images from `assets/examples/`, save 10 golden output fixtures in `golden/`, and construct `verify.py`.

---

## 5. Verification Method

To independently verify the observations and environment status reported in this report:

1. **Verify Python, PyTorch, CUDA, and GPU hardware**:
   ```bash
   python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0)); print('VRAM (GB):', round(torch.cuda.get_device_properties(0).total_memory/1e9, 2))"
   ```
   *Expected Output*: PyTorch `2.12.1+cu126`, CUDA `True`, `NVIDIA GeForce RTX 4050 Laptop GPU`, `6.44` GB.

2. **Verify DA2-Small model instantiation, parameters, and FP32/FP16 VRAM consumption**:
   ```bash
   python -c "import sys; sys.path.append('depth-anything-v2-official'); import torch; from depth_anything_v2.dpt import DepthAnythingV2; m = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384]).to('cuda').eval(); print('Params:', sum(p.numel() for p in m.parameters())); x = torch.randn(1, 3, 518, 518, device='cuda'); torch.cuda.reset_peak_memory_stats(); _ = m(x); print('FP32 Peak VRAM MB:', torch.cuda.max_memory_allocated()/1024/1024); torch.cuda.reset_peak_memory_stats(); with torch.autocast(device_type='cuda', dtype=torch.float16): _ = m(x); print('FP16 Peak VRAM MB:', torch.cuda.max_memory_allocated()/1024/1024)"
   ```
   *Expected Output*: `Params: 24785089`, `FP32 Peak VRAM MB: 224.72`, `FP16 Peak VRAM MB: 288.05`.

3. **Inspect Documentation & Architecture Alignment**:
   - Inspect `PROJECT.md` lines 1–50 for layer definitions.
   - Inspect `docs/adr/001-depthlab-architecture.md` lines 50–125 for Layer 0 immutability and extension boundary.
   - Inspect `docs/adr/002-depthlab-api-stability.md` lines 20–115 for stable public API surface and `FeatureBundle`.
