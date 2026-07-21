---
date: 2026-07-21
parent: docs/brainstorms/2026-07-21-da2-multihead-framework-requirements.md
status: draft
---

# Plan: Multi-Head Extension Framework for Depth Anything V2

**Timeline:** 2 weeks framework → 6 weeks heads (5+)
**Hardware:** RTX 4050 6GB, single GPU, 16GB system RAM
**Entry point:** `python train.py --config experiments/EXP-XXX/experiment.yaml`
**Design principle:** Engineering layer (config, logging, checkpointing, plugin system) is separate from Science layer (hypotheses, benchmark, evaluation, ablations, statistical analysis). Platform changes must never affect scientific results.

---

## Implementation Units

### IU1 — Project scaffold, backbone loading, and LoRA injection (Days 1-2)

**Objective:** Bootable repo with a frozen DA2-Small backbone, configurable LoRA adapters, and a smoke-test that a forward pass runs.

**Files created:**
- `train.py` — single entry point, parses config, orchestrates run
- `config/default.yaml` — canonical config with all keys documented
- `backbone/loader.py` — `load_da2_backbone(path, lora_rank)` → frozen backbone + optional LoRA adapters injected into attention projections
- `backbone/lora.py` — inline `LoRALinear` module (no PEFT dep)
- `heads/__init__.py` — registry decorator `@register_head(name)` + `build_head(config)`
- `heads/base.py` — `BaseHead` abstract class
- `requirements.txt` — torch≥2.0, torchvision, timm, tensorboard, pyyaml, einops
- `.gitignore`

**Key decisions:**
- LoRA rank=8 default, applied after loading pretrained weights; head code never touches backbone internals.
- Registry uses a `_HEAD_REGISTRY` dict with a decorator — no metaclass magic, no YAML-to-code gen.
- Backbone frozen by `requires_grad_(False)` at load time; unfreeze is a config flag.

**Risks:** DA2 checkpoint format may diverge from timm ViT. Mitigation: load via official DA2 repo's `load_checkpoint` first, then strip head weights.

**Accept:** `python train.py --config config/default.yaml` prints backbone param count, LoRA param count, and runs one forward/backward on dummy data without error.

---

### IU2 — Pluggable head interface + relative depth baseline (Days 3-5)

**Objective:** `BaseHead` contract proven by implementing the first head (relative depth) that matches the DA2 paper's original decoder — proving the interface works before adding more heads.

**Files created:**
- `heads/base.py` — `BaseHead(nn.Module)` with `forward(features: list[Tensor]) -> dict`, `compute_loss(preds, batch) -> Tensor`, `compute_metrics(preds, batch) -> dict[str, float]`
- `heads/relative_depth.py` — DINOv2-head decoder (DPT-like reassemble + fusion), `SILogLoss`, relative-depth metric suite (AbsRel, δ1-δ3, RMSE)

**Key decisions:**
- `forward` returns `dict` (not namedtuple). Keys are head-specific; the shared harness logs all scalars from the returned dict and passes the full dict to both `compute_loss` and `compute_metrics`.
- `BaseHead` is an abstract class with `@abstractmethod` on all three methods, but it is intentionally shallow — no shared forward logic, no hooks. A head is "just a module with a contract."
- Metrics are computed in the head so each head owns its evaluation; the harness only aggregates.

**Accept:** Train relative-depth head on NYUv2. After convergence, AbsRel is within ±1% of published DA2-Small result (paper: ~0.128 AbsRel on NYUv2).

---

### IU3 — Dataloader registry + config-driven training harness (Days 6-10)

**Objective:** Single YAML config drives dataset selection, augmentation, optimizer, schedule, and checkpointing. Adding a new dataset pipeline is one file + one decorator.

**Files created:**
- `data/__init__.py` — `_DATASET_REGISTRY` with `@register_dataset(name)` decorator
- `data/nyuv2.py` — NYUv2 dataset class, standard transforms (crop 416×544, color jitter, normalization)
- `data/kitti.py` — KITTI Eigen split, metric depth preprocessing
- `data/diode.py` — DIODE dataset loader
- `data/sintel.py` — Sintel video frames + flow (for temporal heads)
- `data/bsds500.py` — BSDS500 edge ground truth
- `data/transforms.py` — shared augmentation pipeline factory (resize, crop, normalize, to-tensor)
- `trainer.py` — training loop: dataloader construction from config → epoch loop with fp16 (GradScaler), gradient accumulation, AdamW, cosine LR schedule, TensorBoard logging

**Files modified:**
- `train.py` — wires trainer, backbone, head, data; parses config; sets up checkpointing

**Key decisions:**
- Dataset config entries are flat YAML dicts: `dataset.name: nyuv2`, `dataset.root: /data/nyuv2`, `dataset.split: train`, `dataset.image_size: [416, 544]`. The registry maps name → class; the class receives the remaining keys as kwargs.
- Trainer is a single `Trainer` class with `fit()`, not a set of free functions. Internal loop is ~80 lines; no accelerator/lightning wrapper.
- Experiment directory structure: `experiments/EXP-XXX/`. Each experiment folder contains `experiment.yaml` (canonical config snapshot saved at run start), `checkpoints/`, `tensorboard/`, `visualizations/`, and `metrics.json` (written at end of each eval).
- Every run automatically captures and logs: git commit (`git rev-parse HEAD`), CUDA version, PyTorch version, random seed, dataset version, benchmark version, start/end timestamps. Written to `experiment.yaml` as `_metadata` at run start.
- Checkpoint directory: `experiments/EXP-XXX/checkpoints/backbone.pt` (written once), `head-{epoch:03d}.pt` (every epoch). TensorBoard logs in `experiments/EXP-XXX/tensorboard/`.
- One-command reproducibility: `python train.py --config experiments/EXP-XXX/experiment.yaml` reruns the exact same experiment (metadata is preserved, new run gets a child EXP-ID).
- fp16 via `torch.cuda.amp.autocast` + `GradScaler`. Falls back to fp32 if GPU does not support amp (RTX 4050 does).

**Risks:** NYUv2 HDF5 loading speed. Mitigation: cache preprocessed crops as PNGs on first epoch; subsequent runs load from cache. LoRA + fp16 on 6GB may OOM with batch > 4. Mitigation: default batch=4, grad_accum=8 for effective batch 32, configurable downward.

**Accept:** `train.py --config experiments/nyu_relative.yaml` trains from scratch, checkpoints every epoch, resumes from latest. All registered datasets load correctly on `--dry-run`.

---

### IU4 — Metric suite + experiment tracking + visualization (Days 11-14)

**Objective:** Shared evaluation functions, TensorBoard dashboards, visual output (depth maps, error overlays, comparison grids). The framework is feature-complete; heads 2+ add only head-specific code.

**Files created:**
- `metrics/__init__.py` — dispatcher: maps metric name → function
- `metrics/depth.py` — AbsRel, δ1/δ2/δ3, RMSE, SILog
- `metrics/uncertainty.py` — ECE, AUSE, sharpness
- `metrics/temporal.py` — flow-warped photometric error, flicker
- `metrics/normals.py` — mean angular error, accuracy @ 11.25°/22.5°/30°
- `metrics/edges.py` — F-measure (ODS/OIS), precision, recall
- `visualize/depth.py` — turbo colormap rendering, percentile normalization, side-by-side error overlay
- `visualize/grid.py` — comparison grid: N scenes × M heads (or N scenes × M datasets)
- `visualize/video.py` — stacked-frame video for temporal models
- `eval.py` — evaluation-only entry point, loads checkpoint, runs metrics + saves visualizations, optional `--ablate-layer N` for DINOv2 Feature Ablation Atlas

**Files modified:**
- `trainer.py` — calls `head.compute_metrics` at eval epoch, logs to TensorBoard, triggers visualization callback on best checkpoint
- `heads/base.py` — add optional `visualize(preds, batch, save_dir)` method; base implementation is a no-op

**Key decisions:**
- Metric functions are pure functions of `(pred, target)`, not class methods. The dispatcher maps names like `"AbsRel"` to functions, so any metric can be called from any context (train eval, ablation script, standalone).
- DINOv2 Feature Ablation Atlas: `eval.py --ablate-layer 6` replaces the feature from block 6 with zeros before passing to the head, logging metric delta. Running `--ablate-layer 0..11` across 12 blocks generates the full atlas.
- Visualization outputs go to `runs/{experiment_name}/visualizations/{epoch}/`. Images are logged to TensorBoard as figure scalars; PNGs saved to disk for later inspection.

**Accept:** Running `eval.py --config experiments/nyu_relative.yaml --checkpoint runs/nyu_relative/checkpoints/head-020.pt` produces per-metric stdout table, TensorBoard scalars, depth map PNGs, and error overlays. `--ablate-layer 4` shows metric degradation for that block.

---

### IU5 — Second head (uncertainty) as validation + documentation (Days ~Week 3, overlaps with experiments)

**Objective:** Prove the "≤100 lines new code" claim by implementing an uncertainty estimation head. Demonstrates the framework is ready for the 6-week experiment phase.

**Files created:**
- `heads/uncertainty.py` — decoder with two output heads (depth + log-variance), `compute_loss` uses learned log-variance loss (Kendall et al. 2017), `compute_metrics` reports ECE + AUSE + sharpness

**Files modified:**
- `experiments/nyu_uncertainty.yaml` — config: points to `head.type: uncertainty`, dataset unchanged from relative-depth config, all other params identical

**Key decisions:**
- A second head validates the framework abstraction. It uses the same backbone config, same dataloader (NYUv2), same trainer — the YAML diff from the relative-depth experiment is exactly 3 lines: `head.type`, `head.name`, and `experiment_name`.
- This IU is intentionally deferred to week 3 to allow a week of buffer on IU1-IU4. If framework hits all acceptance criteria by day 14, this starts immediately.

**Accept:** `train.py --config experiments/nyu_uncertainty.yaml` trains to convergence without touching shared code. Outputs depth + uncertainty map side-by-side. Total head-specific code: ~90 lines.

---

## Timeline

| IU | Days | Deliverable |
|----|------|-------------|
| IU1 | 1-2 | Bootable scaffold, frozen backbone + LoRA, forward-pass smoke test |
| IU2 | 3-5 | BaseHead contract, relative-depth head, first training run |
| IU3 | 6-10 | Dataloader registry, training harness, checkpointing, TensorBoard |
| IU4 | 11-14 | Metric suite, eval.py, visualization, feature ablation |
| IU5 | Wk3 | Uncertainty head ≤100 lines, proving framework abstraction |

Buffer: days 5, 10, 14 are slack/buffer days. If IU1-IU4 slip by ≤2 days total, IU5 starts on schedule day 15.

---

## Computational Budget

| Resource | Budget | Notes |
|----------|--------|-------|
| GPU memory | ≤6 GB | RTX 4050 hard limit |
| Single experiment | ≤24 hours | Training + eval combined |
| Full ablation suite | ≤7 days | 3 seeds × 4 variants × 24h |
| Storage | ≤500 GB | Datasets + checkpoints + results |
| Checkpoint retention | Keep best + last only | Delete intermediate epochs |

---

## Success Criteria Mapping

| Criteria | Coverage |
|----------|----------|
| SC1: 5+ heads from `train.py --config` | IU3 harness + IU2 registry |
| SC2: New head ≤100 lines, zero infra changes | IU5 validates |
| SC3: Relative depth matches DA2 ±1% AbsRel | IU2 acceptance gate |
| SC4: Heads 2-5 ≤1 week each | Framework complete at IU4; IU5 proves ≤1 week |
| SC5: Shared backbone checkpoint | IU3 separate backbone/head save |
| SC6: `--ablate-layer` Feature Ablation Atlas | IU4 `eval.py` flag |

---

## Dependencies

- DA2-Small pretrained checkpoint downloaded from official DA2 repo (github.com/DepthAnything/Depth-Anything-V2) to `checkpoints/da2-small.pt`
- NYUv2 dataset at configurable path (`/data/nyuv2`)
- KITTI, DIODE, Sintel, BSDS500 — optional for framework validation, required for heads
- PyTorch ≥2.0 + CUDA runtime (driver already present for RTX 4050)
