# Handoff Report — Reviewer M3 2 (reviewer_m3_2)

**From**: Reviewer M3 2 (`reviewer_m3_2`)  
**To**: Project Orchestrator (`78803110-53f8-4299-8bf1-e0ba882962fe` / `9afb784c-cef4-4445-97fc-ab8f19ccfbd4`)  
**Date**: 2026-07-21  
**Workspace**: `C:\Users\rushd\Downloads\prj-res\.agents\reviewer_m3_2`  
**Milestone**: Milestone 3: Phase 0.4 Multi-Head Extension Framework & Scaffolding  

---

## 1. Review Summary & Verdict

**Verdict**: **REJECT** (REQUEST_CHANGES)

**Rationale**:  
While the framework scaffolding, Layer 1 backbone adapter, head registry, experiment tracking (`experiment.yaml`), and `verify.py --all` test suite (12/12 passing) are structurally sound and free of malicious integrity violations, an **execution-critical numerical bug** exists in `depthlab/metrics/depth_metrics.py`. When training/evaluating under PyTorch AMP FP16 (which is enabled by default on CUDA), intermediate prediction tensors in `torch.float16` overflow the IEEE 754 float16 half-precision maximum range (65,504.0) during least-squares scale-shift alignment (`sum_pp = torch.sum(pred_masked ** 2)`). This results in `sum_pp = inf`, `det = nan`, and forces all evaluation metrics (`AbsRel`, `RMSE`, `SILog`, `SqRel`, `log10`) to return `NaN` during validation loops (`AbsRel: nan`).

---

## 2. Findings

### [Critical] Finding 1: Float16 Overflow in Scale-Shift Alignment Causes NaN Evaluation Metrics During AMP FP16 Training

- **What**: During training with `--config config/default.yaml` on CUDA, validation epoch metrics report `AbsRel: nan`.
- **Where**: `depthlab/metrics/depth_metrics.py`, lines 10–43 (`align_depth_scale_shift` function).
- **Why**:
  1. Under `torch.amp.autocast(dtype=torch.float16)`, model outputs (`preds["depth"]`) are emitted as `torch.float16` tensors.
  2. In `align_depth_scale_shift`, `pred_masked` is a `torch.float16` tensor containing $H \times W$ pixels (e.g., $518 \times 518 = 268,324$ elements).
  3. `sum_pp = torch.sum(pred_masked ** 2)` accumulates the sum of squares in `float16`. The maximum finite value for `float16` is `65,504.0`.
  4. For any standard image resolution where average predicted depth is $\ge 0.5$, $\sum p^2 > 65504$, causing `sum_pp` to overflow to `torch.inf`.
  5. Line 34 computes `det = sum_pp * n - sum_p * sum_p`. With `sum_pp = inf`, `det` becomes `inf - inf = nan`.
  6. Line 39-40 calculates `scale` and `shift`, which both evaluate to `nan`.
  7. Line 42 returns `scale * pred + shift` (all `nan`s), corrupting all depth metrics (`abs_rel`, `rmse`, `silog`, `sq_rel`, `log10`) to `nan`.

- **Verbatim Log Output from `python train.py --config config/default.yaml`**:
  ```text
  [Trainer] Recorded reproducibility metadata to C:\Users\rushd\Downloads\prj-res\outputs\default_synthetic\experiment.yaml
  [Trainer] Starting training for 2 epochs on device: cuda
  [Trainer] Saved new best model to C:\Users\rushd\Downloads\prj-res\outputs\default_synthetic\checkpoint_best.pth
  Epoch [01/02] (6.4s) | Train Loss: 10.8152 | Val Loss: 10.8152 | AbsRel: nan | d1: 0.0000
  Epoch [02/02] (3.6s) | Train Loss: 10.8152 | Val Loss: 10.8152 | AbsRel: nan | d1: 0.0000
  [Trainer] Training complete. Artifacts saved in C:\Users\rushd\Downloads\prj-res\outputs\default_synthetic
  ```

- **Minimal Isolated Reproduction**:
  ```bash
  python -c "import torch; from depthlab.metrics.depth_metrics import compute_depth_metrics; p = torch.ones(1, 1, 518, 518, dtype=torch.float16); t = torch.ones(1, 1, 518, 518, dtype=torch.float32)*2.0; print(compute_depth_metrics(p, t, align=True))"
  ```
  *Result*:
  `{'abs_rel': nan, 'sq_rel': nan, 'rmse': nan, 'rmse_log': nan, 'silog': nan, 'log10': nan, 'delta1': 0.0, 'delta2': 0.0, 'delta3': 0.0, 'd1': 0.0, 'd2': 0.0, 'd3': 0.0}`

- **Suggested Fix Direction**:
  In `depthlab/metrics/depth_metrics.py`, explicitly cast `pred` and `target` to `float32` at the beginning of `align_depth_scale_shift` and `compute_depth_metrics`:
  ```python
  def align_depth_scale_shift(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
      pred = pred.float()
      target = target.float()
      ...
  ```

---

## 3. Verified Claims & Test Matrix

| Claim / Specification | Verification Method | Status | Details |
|---|---|---|---|
| Single Entrypoint Training (`train.py`) | Executed `python train.py --config config/default.yaml` | **PARTIAL** | Executed 2 epochs, created `outputs/default_synthetic/experiment.yaml` & `checkpoint_best.pth`, but validation `AbsRel` was `nan`. |
| Reproducibility Tracking (`experiment.yaml`) | Inspected generated `experiment.yaml` in output dir | **PASS** | Contains git commit, SHA256 checkpoint hash, seeds, PyTorch/CUDA versions, Python version. |
| Feature Ablation Hook (`--ablate-layer`) | Executed `python eval.py --config config/default.yaml --checkpoint outputs/default_synthetic/checkpoint_best.pth --ablate-layer 0 1` | **PASS** | Successfully zeroed out stages 0 and 1 features and generated summary table. |
| Verification Suite (`verify.py --all`) | Executed `python verify.py --all` | **PASS** | 12/12 tests passed (`OVERALL STATUS: ALL CHECKS PASSED`). |
| Layer 0 Immutability | Inspected `depth-anything-v2-official/` | **PASS** | Zero source files modified in official repository. |
| Integrity Check | Inspected source code for hardcoded outputs or facades | **CLEAN** | No hardcoded test outputs, no fake implementations. Scaffolding is genuinely implemented. |

---

## 4. Observation & Logic Chain

1. **Observation 1**: Executing `python train.py --config config/default.yaml` produced output with `AbsRel: nan`.
2. **Observation 2**: Running `verify.py --all` passed 12/12 tests because Test 12 in `verify.py` passes `pred_depth.detach()` directly to `MetricDispatcher` without scale-shift alignment (`align=False` or random synthetic target without FP16 overflow trigger).
3. **Logic Step**: In `depthlab/trainer.py`, `Trainer.evaluate()` runs under `autocast(dtype=torch.float16)`. `preds["depth"]` is a FP16 tensor. `MetricDispatcher` calls `compute_depth_metrics(..., align=True)`. In `align_depth_scale_shift`, `sum_pp = torch.sum(pred_masked ** 2)` is evaluated on FP16 elements. 518x518 elements = 268,324 elements. $268,324 \times 1.0 = 268,324 > 65,504$ (FP16 max limit). `sum_pp` overflows to `inf`, causing `det = inf - inf = nan`, rendering `scale` and `shift` `nan`.
4. **Conclusion**: The metric evaluation module must be patched to cast tensors to `float32` before computing reduction operations ($\sum x^2$, $\sum x y$) to avoid float16 overflow under AMP training.

---

## 5. Caveats

- **No Integrity Violations**: The code was audited for intentional shortcuts, hardcoded test results, facade classes, or self-certifying hacks. None were found. The bug is purely a precision oversight during AMP FP16 integration.
- **Reviewer Constraint**: As a Reviewer agent, I have not modified the source code (`depthlab/metrics/depth_metrics.py`), adhering to the strict "Review-only — do NOT modify implementation code" constraint.

---

## 6. Verification Method

To verify the issue and confirm resolution once fixed:

1. **Reproduce Bug**:
   ```bash
   python -c "import torch; from depthlab.metrics.depth_metrics import compute_depth_metrics; p = torch.ones(1, 1, 518, 518, dtype=torch.float16); t = torch.ones(1, 1, 518, 518, dtype=torch.float32)*2.0; print(compute_depth_metrics(p, t, align=True))"
   ```
   *Expected Current Output*: `abs_rel: nan`.

2. **Verify Training Evaluation Output**:
   ```bash
   python train.py --config config/default.yaml
   ```
   *Expected Post-Fix Output*: Epoch log should report numeric `AbsRel` value (e.g. `AbsRel: 0.7547`), not `nan`.

3. **Full Suite Verification**:
   ```bash
   python verify.py --all
   ```
   *Expected Output*: `12/12` tests passed.
