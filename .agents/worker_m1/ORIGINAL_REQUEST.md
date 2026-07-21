## 2026-07-21T23:17:30Z
You are Worker 1 for Milestone 1 of DepthLab in working directory C:\Users\rushd\Downloads\prj-res.
Your working directory for metadata/handoffs is C:\Users\rushd\Downloads\prj-res\.agents\worker_m1.

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A Forensic Auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Objective:
Implement Milestone 1 deliverables:
1. Checkpoint setup: Ensure `checkpoints/depth_anything_v2_vits.pth` exists. You may create `scripts/download_checkpoints.py` to download the official model weights from `https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth` to `checkpoints/depth_anything_v2_vits.pth` or verify existing checkpoint file.
2. Golden Fixtures Generator (`scripts/generate_golden.py`): Create `scripts/generate_golden.py` to load official DA2 Small (`vits`), process `depth-anything-v2-official/assets/examples/demo01.jpg` through `demo10.jpg`, and generate:
   - `golden/images/demo01.jpg` .. `demo10.jpg`
   - `golden/depths/demo01_depth.npy` .. `demo10_depth.npy` (raw float32 depth arrays)
   - `golden/vis/demo01_vis.png` .. `demo10_vis.png` (visualized depth maps)
   - `golden/fixture_manifest.json` (metadata index with sample parameters, dimensions, statistics, sha256 checksums)
3. Standalone Verification Script (`verify.py`): Implement `verify.py` at the project root following the 10-point verification check specification in `C:\Users\rushd\Downloads\prj-res\.agents\explorer_m1_3\handoff.md` and `C:\Users\rushd\Downloads\prj-res\.agents\explorer_m1_2\handoff.md`.
   - Validate checkpoint loading, FP16 execution, correct output shape (H×W), non-negative range, zero NaNs/Infs, runtime bounds (≤1s @ 518×518), peak VRAM ≤ 2GB, and regression alignment (MAE < 1e-6 against golden sample).
   - Support `python verify.py` and `python verify.py --all`.
4. Verification: Run `python verify.py` and `python verify.py --all`, documenting exact build and test execution results.

Deliver your implementation report in `C:\Users\rushd\Downloads\prj-res\.agents\worker_m1\handoff.md` following the Handoff Protocol. Send a message to parent when complete.
