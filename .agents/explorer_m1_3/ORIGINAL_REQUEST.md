## 2026-07-21T17:43:52Z
<USER_REQUEST>
You are Explorer 3 for Milestone 1 of DepthLab in working directory C:\Users\rushd\Downloads\prj-res.
Your working directory for metadata/handoffs is C:\Users\rushd\Downloads\prj-res\.agents\explorer_m1_3.

Objective:
Investigate requirements and design for the standalone verify.py script at project root.
1. verify.py must validate:
   - Checkpoint loading (Depth Anything V2 Small checkpoint)
   - Inference execution under FP16 mixed precision
   - Correct output shape (H×W matching input)
   - Non-negative depth range
   - Absence of NaNs / Infs
   - Runtime bounds (≤ 1s per image at 518×518)
   - Peak VRAM ≤ 2GB
   - Regression check asserting depth output alignment against golden fixtures in golden/ within configurable tolerance (MAE < 1e-6 for FP32/FP16 matching).
2. Design clean, robust Python verification functions and CLI interface (`python verify.py` and `python verify.py --all`).
3. Read PROJECT.md, ROADMAP.md, and ADR-001/002.

Deliver your analysis and design in C:\Users\rushd\Downloads\prj-res\.agents\explorer_m1_3\handoff.md following the Handoff Protocol. Send a message to parent when finished.
</USER_REQUEST>
