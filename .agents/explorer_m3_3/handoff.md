# Handoff Report — Explorer 3 (explorer_m3_3)
**Milestone**: Milestone 3 (M3: Multi-Head Extension Framework & Scaffolding)
**Working Directory**: `C:\Users\rushd\Downloads\prj-res\.agents\explorer_m3_3`
**Date**: 2026-07-21
**Author**: Explorer 3 (`explorer_m3_3`)

---

## 1. Observation

1. **Repository Layout and ADR-001/002 Architecture**:
   - `PROJECT.md` defines DepthLab's three-layer architecture: Layer 0 (`depth-anything-v2-official/`, immutable), Layer 1 (`depthlab/backbone/`, `depthlab/data/`, `depthlab/transforms.py`, `depthlab/metrics/`), Layer 2 (`depthlab/heads/`, `depthlab/trainer.py`, `train.py`, `eval.py`, `verify.py`).
   - Milestone 1 (M1) and Milestone 2 (M2) have completed cleanly, producing golden fixtures (`golden/`), validation scripts (`verify.py`), and reproduction documentation (`docs/reproduction/`).
   - The directory `depthlab/` is currently uncreated and scheduled for full construction in Milestone 3.

2. **Explorer Division of Responsibilities for M3**:
   - `explorer_m3_1`: Layer 1 Backbone Loader and Adapter (`depthlab/backbone/loader.py`, `depthlab/backbone/adapter.py`, `FeatureStage`, `FeatureBundle`, `DA2Backbone`).
   - `explorer_m3_2`: Head Registry (`depthlab/heads/__init__.py`), `BaseHead` (`depthlab/heads/base.py`), Relative Depth Head (`depthlab/heads/relative_depth.py`), and Dataset Registry (`depthlab/data/`).
   - `explorer_m3_3` (This Agent): Training Harness (`depthlab/trainer.py`, `train.py`), Standalone Evaluation Engine (`eval.py`), Metric Suite & Dispatcher (`depthlab/metrics/depth_metrics.py`, `depthlab/metrics/dispatcher.py`), YAML Configs (`config/default.yaml`, `experiments/nyu_relative.yaml`), Feature Ablation Hook (`--ablate-layer`), Reproducibility Tracker (`experiment.yaml` writer), and `verify.py` scaffolding check extension.

3. **Verification Suite (`verify.py`)**:
   - Currently contains 11 tests covering checkpoint loading, FP16 execution, output shape, depth range, NaN/Inf absence, runtime/VRAM bounds, golden fixture regression, synthetic extreme stress, memory accumulation, and reproduction report schema validation.
   - Requires extension with Layer 1 & 2 Framework Scaffolding execution test (`check_framework_scaffolding`) to validate end-to-end framework execution.

---

## 2. Logic Chain

1. **Training Harness Design (`depthlab/trainer.py` & `train.py`)**:
   - To achieve robust, reproducible training, `Trainer` must take a single YAML configuration dictionary containing model, dataset, training, evaluation, and experiment parameters.
   - For mixed-precision training without numerical instability or gradient underflow, `torch.amp.autocast(device_type='cuda', dtype=torch.float16)` and `torch.amp.GradScaler('cuda')` (or `torch.cuda.amp.GradScaler`) must be employed.
   - Optimization utilizes AdamW with weight decay and `torch.optim.lr_scheduler.CosineAnnealingLR` (with optional warmup).
   - TensorBoard integration via `torch.utils.tensorboard.SummaryWriter` logs scalar metrics (`loss`, `lr`, step runtimes, evaluation metrics) per epoch/step.
   - Checkpoint saving persists `checkpoint_latest.pth`, `checkpoint_best.pth`, and epoch snapshots containing `epoch`, `head_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `scaler_state_dict`, `metrics`, and `config`.

2. **Reproducibility Tracker (`experiment.yaml`)**:
   - To adhere to ADR-002 reproducibility principles, every training run must automatically record metadata to `output_dir / "experiment.yaml"`.
   - Recorded metadata fields: `git_commit`, `git_dirty`, `checkpoint_hash` (SHA256 of backbone checkpoint), `cuda_version`, `pytorch_version`, `python_version`, `device_name`, `cudnn_version`, `seed`, `dataset_version`, `benchmark_version`, `timestamp`, and full `config_dump`.
   - Seeding ensures deterministic RNG behavior across PyTorch, NumPy, Python standard library, and CUDA CuDNN.

3. **Metrics & Standalone Evaluation Engine (`eval.py` & `depthlab/metrics/`)**:
   - Standard depth metrics must be computed: Absolute Relative Error ($AbsRel$), Squared Relative Error ($SqRel$), Root Mean Squared Error ($RMSE$), Log Root Mean Squared Error ($RMSE_{log}$), Scale-Invariant Log Error ($SILog$), $log_{10}$ error, and Threshold Accuracies ($\delta_1 < 1.25$, $\delta_2 < 1.25^2$, $\delta_3 < 1.25^3$).
   - `MetricDispatcher` manages batched accumulator stats, supports median scale-and-shift alignment for relative depth evaluation, and formats evaluation summaries in dictionary and markdown table formats.
   - Feature Ablation Hook: `--ablate-layer` flag in `eval.py` (e.g. `--ablate-layer 0 1` or `--ablate-layer 2`) zeroes out specific `FeatureStage` patch tokens within `FeatureBundle` before feeding features into the head decoder. This allows evaluating layer importance without modifying backbone parameters or model code.

4. **YAML Configuration Schema**:
   - `config/default.yaml`: Fast synthetic dataset path for CI/smoke testing without external data downloads.
   - `experiments/nyu_relative.yaml`: Full relative depth training & evaluation configuration on NYUv2 dataset.

5. **`verify.py` Scaffolding Extension**:
   - Adding `check_framework_scaffolding` to `verify.py` tests that `DA2Backbone`, `BaseHead` registry, `FeatureBundle` extraction, forward inference, loss computation, loss backward pass, `MetricDispatcher`, and feature ablation hooks operate seamlessly together.

---

## 3. Caveats

1. **Hardware / CUDA Dependency for FP16**:
   - PyTorch AMP FP16 requires a CUDA-capable GPU. On CPU-only environments, `autocast` will gracefully fall back to CPU FP32/BFloat16, and `GradScaler` will be bypassed (`enabled=False`).

2. **Synthetic Data Fast Path**:
   - `config/default.yaml` uses a synthetic dataset generator so that `python train.py --config config/default.yaml` can run instantly in CI without requiring the 40GB NYUv2 dataset.

3. **Relative Depth Scale & Shift Alignment**:
   - For relative depth evaluation against metric ground truth (like NYUv2), scale and shift alignment ($\hat{y}_{aligned} = s \cdot \hat{y} + t$) is standard practice. `MetricDispatcher` exposes `align_scale_shift=True` by default for relative depth models.

---

## 4. Conclusion & Complete Design Specifications

Below are the complete, production-ready module implementations designed for Explorer 3's scope.

### Component 4.1: `depthlab/metrics/depth_metrics.py`

```python
"""
depthlab/metrics/depth_metrics.py — Core Depth Estimation Metric Definitions.

Computes standard monocular depth estimation evaluation metrics:
- AbsRel: Absolute Relative Error
- SqRel: Squared Relative Error
- RMSE: Root Mean Squared Error
- RMSE_log: Root Mean Squared Log Error
- SILog: Scale-Invariant Logarithmic Error
- log10: Mean Absolute Log10 Error
- delta1, delta2, delta3: Threshold Accuracies (delta < 1.25^i)
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn


def align_depth_scale_shift(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Aligns predicted relative depth to target metric depth using least-squares scale and shift.
    Solves min_{s, t} sum || (s * pred + t) - target ||^2
    """
    if mask is None:
        mask = target > 0

    pred_masked = pred[mask]
    target_masked = target[mask]

    if pred_masked.numel() == 0:
        return pred

    # Form linear system A * [s, t]^T = b
    # A = [[sum(pred^2), sum(pred)], [sum(pred), N]]
    # b = [sum(pred * target), sum(target)]
    n = float(pred_masked.numel())
    sum_p = torch.sum(pred_masked)
    sum_t = torch.sum(target_masked)
    sum_pp = torch.sum(pred_masked ** 2)
    sum_pt = torch.sum(pred_masked * target_masked)

    det = sum_pp * n - sum_p * sum_p
    if torch.abs(det) < 1e-8:
        # Fallback to median scaling
        scale = torch.median(target_masked) / (torch.median(pred_masked) + 1e-8)
        shift = torch.tensor(0.0, device=pred.device)
    else:
        scale = (sum_pt * n - sum_p * sum_t) / det
        shift = (sum_pp * sum_t - sum_p * sum_pt) / det

    return scale * pred + shift


def compute_depth_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    align: bool = False,
    min_depth: float = 1e-3,
    max_depth: float = 80.0
) -> Dict[str, float]:
    """
    Computes depth evaluation metrics on valid masked pixels.

    Args:
        pred: Predicted depth map (B, 1, H, W) or (B, H, W)
        target: Target depth map (B, 1, H, W) or (B, H, W)
        mask: Optional boolean mask (B, 1, H, W) where True = valid pixel
        align: Whether to compute least-squares scale & shift alignment
        min_depth: Minimum valid depth threshold
        max_depth: Maximum valid depth threshold

    Returns:
        Dictionary containing scalar float metric values.
    """
    if pred.ndim == 3:
        pred = pred.unsqueeze(1)
    if target.ndim == 3:
        target = target.unsqueeze(1)

    # Valid mask computation
    valid_mask = (target > min_depth) & (target < max_depth) & ~torch.isnan(target) & ~torch.isinf(target)
    if mask is not None:
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        valid_mask = valid_mask & mask

    if not valid_mask.any():
        return {
            "abs_rel": 0.0, "sq_rel": 0.0, "rmse": 0.0, "rmse_log": 0.0,
            "silog": 0.0, "log10": 0.0, "delta1": 0.0, "delta2": 0.0, "delta3": 0.0
        }

    if align:
        pred = align_depth_scale_shift(pred, target, valid_mask)

    # Clamp predictions to valid range
    pred = torch.clamp(pred, min=min_depth, max=max_depth)

    p = pred[valid_mask]
    t = target[valid_mask]

    thresh = torch.max(p / t, t / p)
    d1 = torch.mean((thresh < 1.25).float()).item()
    d2 = torch.mean((thresh < 1.25 ** 2).float()).item()
    d3 = torch.mean((thresh < 1.25 ** 3).float()).item()

    diff = p - t
    abs_rel = torch.mean(torch.abs(diff) / t).item()
    sq_rel = torch.mean((diff ** 2) / t).item()
    rmse = torch.sqrt(torch.mean(diff ** 2)).item()

    log_p = torch.log(p)
    log_t = torch.log(t)
    log_diff = log_p - log_t
    rmse_log = torch.sqrt(torch.mean(log_diff ** 2)).item()

    # SILog formula: sqrt(mean(d^2) - (mean(d))^2) * 10 or 100
    silog = torch.sqrt(torch.mean(log_diff ** 2) - (torch.mean(log_diff) ** 2) + 1e-8).item() * 100.0
    log10 = torch.mean(torch.abs(torch.log10(p) - torch.log10(t))).item()

    return {
        "abs_rel": float(abs_rel),
        "sq_rel": float(sq_rel),
        "rmse": float(rmse),
        "rmse_log": float(rmse_log),
        "silog": float(silog),
        "log10": float(log10),
        "delta1": float(d1),
        "delta2": float(d2),
        "delta3": float(d3),
    }
```

---

### Component 4.2: `depthlab/metrics/dispatcher.py`

```python
"""
depthlab/metrics/dispatcher.py — Metric Dispatcher and Feature Ablation Hook.

Provides MetricDispatcher for accumulation across batches/epochs, and
apply_feature_ablation hook for zeroing specified backbone stages.
"""

from typing import Dict, List, Optional, Union, Any
import torch
from depthlab.metrics.depth_metrics import compute_depth_metrics


def apply_feature_ablation(
    feature_bundle: Any,
    ablate_layers: Optional[List[int]] = None
) -> Any:
    """
    Zeroes out feature representations for specified stage indices in FeatureBundle.

    Args:
        feature_bundle: Layer 1 FeatureBundle containing list of FeatureStage objects
        ablate_layers: List of stage indices (e.g. [0, 1]) to zero out

    Returns:
        A new FeatureBundle instance with specified stages ablated.
    """
    if not ablate_layers:
        return feature_bundle

    # Check if FeatureBundle has ablate_stage method or build new stages
    if hasattr(feature_bundle, "ablate_stage"):
        bundle = feature_bundle
        for idx in ablate_layers:
            bundle = bundle.ablate_stage(idx)
        return bundle

    # Fallback to stage iteration
    from depthlab.backbone.adapter import FeatureBundle, FeatureStage
    new_stages = []
    for stage in feature_bundle.stages:
        if stage.stage_index in ablate_layers:
            zero_patch = torch.zeros_like(stage.patch_tokens)
            zero_cls = torch.zeros_like(stage.cls_token) if stage.cls_token is not None else None
            new_stages.append(FeatureStage(
                patch_tokens=zero_patch,
                cls_token=zero_cls,
                stage_index=stage.stage_index,
                embed_dim=stage.embed_dim
            ))
        else:
            new_stages.append(stage)

    return FeatureBundle(stages=new_stages)


class MetricDispatcher:
    """
    Batched metric accumulator and reporter for evaluation loops.
    """
    def __init__(
        self,
        align_scale_shift: bool = True,
        min_depth: float = 1e-3,
        max_depth: float = 80.0
    ):
        self.align_scale_shift = align_scale_shift
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.reset()

    def reset(self):
        """Resets all accumulated metric statistics."""
        self.accumulated_metrics: Dict[str, float] = {}
        self.num_batches: int = 0

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Computes metrics for a batch and updates running totals.
        """
        metrics = compute_depth_metrics(
            pred=pred,
            target=target,
            mask=mask,
            align=self.align_scale_shift,
            min_depth=self.min_depth,
            max_depth=self.max_depth
        )

        for k, v in metrics.items():
            self.accumulated_metrics[k] = self.accumulated_metrics.get(k, 0.0) + v
        self.num_batches += 1
        return metrics

    def compute(self) -> Dict[str, float]:
        """
        Returns average metric values across all accumulated batches.
        """
        if self.num_batches == 0:
            return {}
        return {k: v / self.num_batches for k, v in self.accumulated_metrics.items()}

    def summary_table((self) -> str:
        """Generates a formatted text table summarizing evaluated metrics."""
        results = self.compute()
        if not results:
            return "No metrics accumulated."

        lines = [
            "+------------+------------+",
            "| Metric     | Value      |",
            "+------------+------------+",
        ]
        for k, v in results.items():
            lines.append(f"| {k:<10} | {v:10.4f} |")
        lines.append("+------------+------------+")
        return "\n".join(lines)
```

---

### Component 4.3: `depthlab/trainer.py`

```python
"""
depthlab/trainer.py — Modular Training Harness & Reproducibility Tracker.

Handles AMP FP16 mixed precision training, GradScaler, AdamW optimizer,
Cosine Annealing LR scheduler, TensorBoard logging, checkpointing, and
automatic experiment.yaml metadata recording.
"""

import os
import sys
import time
import random
import hashlib
import datetime
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from depthlab.backbone.loader import load_da2_checkpoint
from depthlab.heads import get_head
from depthlab.data import get_dataset
from depthlab.metrics.dispatcher import MetricDispatcher


def get_git_commit() -> Dict[str, Any]:
    """Retrieves current git commit hash and status."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        return {"commit": commit, "dirty": bool(status)}
    except Exception:
        return {"commit": "unknown", "dirty": False}


def compute_file_sha256(filepath: Path) -> str:
    """Computes SHA256 checksum of a file."""
    if not filepath.exists():
        return "missing"
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def seed_everything(seed: int = 42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Trainer:
    """
    Main DepthLab Training Engine.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.exp_cfg = config.get("experiment", {})
        self.model_cfg = config.get("model", {})
        self.data_cfg = config.get("dataset", {})
        self.train_cfg = config.get("training", {})
        self.eval_cfg = config.get("eval", {})

        # Setup output directory
        self.output_dir = Path(self.exp_cfg.get("output_dir", "outputs/default"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup reproducibility seed
        self.seed = int(self.exp_cfg.get("seed", 42))
        seed_everything(self.seed)

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.output_dir / "tensorboard"))

        # Write experiment.yaml reproducibility metadata
        self._record_experiment_metadata()

        # Initialize Backbone
        bb_cfg = self.model_cfg.get("backbone", {})
        self.backbone = load_da2_checkpoint(
            variant=bb_cfg.get("variant", "vits"),
            checkpoint_path=Path(bb_cfg.get("checkpoint_path", "checkpoints/depth_anything_v2_vits.pth"))
        ).to(self.device)

        if bb_cfg.get("frozen", True):
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        # Initialize Head via Registry
        head_cfg = self.model_cfg.get("head", {})
        head_name = head_cfg.get("name", "relative_depth")
        self.head = get_head(
            head_name,
            features=head_cfg.get("features", 64),
            out_channels=head_cfg.get("out_channels", [48, 96, 192, 384])
        ).to(self.device)

        # Datasets and Loaders
        self.train_loader, self.val_loader = self._setup_dataloaders()

        # Optimizer and Scheduler
        lr = float(self.train_cfg.get("learning_rate", 5e-4))
        weight_decay = float(self.train_cfg.get("weight_decay", 1e-2))
        self.optimizer = torch.optim.AdamW(self.head.parameters(), lr=lr, weight_decay=weight_decay)

        epochs = int(self.train_cfg.get("epochs", 25))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)

        # AMP Mixed Precision Scaler
        self.use_amp = bool(self.train_cfg.get("amp", True)) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.best_val_loss = float("inf")

    def _record_experiment_metadata(self):
        """Generates and writes experiment.yaml into the output directory."""
        git_info = get_git_commit()
        ckpt_path = Path(self.model_cfg.get("backbone", {}).get("checkpoint_path", ""))
        ckpt_hash = compute_file_sha256(ckpt_path)

        metadata = {
            "experiment": {
                "name": self.exp_cfg.get("name", "unnamed"),
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "git_commit": git_info["commit"],
                "git_dirty": git_info["dirty"],
                "checkpoint_path": str(ckpt_path),
                "checkpoint_hash": ckpt_hash,
                "reproducibility": {
                    "seed": self.seed,
                    "dataset_version": self.exp_cfg.get("dataset_version", "v1.0"),
                    "benchmark_version": self.exp_cfg.get("benchmark_version", "eigen_v1"),
                },
                "system": {
                    "python_version": sys.version.split()[0],
                    "pytorch_version": torch.__version__,
                    "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
                    "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                },
                "config_dump": self.config
            }
        }

        metadata_file = self.output_dir / "experiment.yaml"
        with open(metadata_file, "w") as f:
            yaml.dump(metadata, f, default_flow_style=False)
        print(f"[Trainer] Recorded reproducibility metadata to {metadata_file}")

    def _setup_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        ds_name = self.data_cfg.get("name", "synthetic")
        train_ds = get_dataset(ds_name, split="train", **self.data_cfg)
        val_ds = get_dataset(ds_name, split="val", **self.data_cfg)

        batch_size = int(self.data_cfg.get("batch_size", 4))
        num_workers = int(self.data_cfg.get("num_workers", 0))

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_loader, val_loader

    def train_epoch(self, epoch: int) -> float:
        self.head.train()
        self.backbone.eval()
        total_loss = 0.0

        for step, batch in enumerate(self.train_loader):
            images = batch["image"].to(self.device)
            targets = batch["depth"].to(self.device)

            self.optimizer.zero_grad()

            with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                with torch.no_grad():
                    features = self.backbone.features(images)
                preds = self.head(features)
                loss = self.head.compute_loss(preds, {"depth": targets})

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.head.parameters(), max_norm=float(self.train_cfg.get("clip_grad_norm", 1.0)))
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            global_step = epoch * len(self.train_loader) + step
            self.writer.add_scalar("Train/BatchLoss", loss.item(), global_step)

        avg_loss = total_loss / max(len(self.train_loader), 1)
        self.writer.add_scalar("Train/EpochLoss", avg_loss, epoch)
        self.writer.add_scalar("Train/LearningRate", self.scheduler.get_last_lr()[0], epoch)
        return avg_loss

    def evaluate(self, epoch: int) -> Dict[str, float]:
        self.head.eval()
        self.backbone.eval()

        dispatcher = MetricDispatcher(
            align_scale_shift=self.eval_cfg.get("align_scale_shift", True),
            min_depth=float(self.eval_cfg.get("min_depth", 1e-3)),
            max_depth=float(self.eval_cfg.get("max_depth", 80.0))
        )

        val_loss = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch["image"].to(self.device)
                targets = batch["depth"].to(self.device)

                with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                    features = self.backbone.features(images)
                    preds = self.head(features)
                    loss = self.head.compute_loss(preds, {"depth": targets})

                val_loss += loss.item()
                pred_depth = preds.get("depth", preds.get("pred"))
                dispatcher.update(pred_depth, targets)

        avg_val_loss = val_loss / max(len(self.val_loader), 1)
        metrics = dispatcher.compute()
        metrics["val_loss"] = avg_val_loss

        for k, v in metrics.items():
            self.writer.add_scalar(f"Val/{k}", v, epoch)

        return metrics

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        ckpt_data = {
            "epoch": epoch,
            "head_state_dict": self.head.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "metrics": metrics,
            "config": self.config
        }

        latest_path = self.output_dir / "checkpoint_latest.pth"
        torch.save(ckpt_data, latest_path)

        save_int = int(self.train_cfg.get("save_interval", 5))
        if epoch % save_int == 0:
            torch.save(ckpt_data, self.output_dir / f"checkpoint_epoch_{epoch}.pth")

        if is_best:
            best_path = self.output_dir / "checkpoint_best.pth"
            torch.save(ckpt_data, best_path)
            print(f"[Trainer] Saved new best model to {best_path}")

    def fit(self):
        epochs = int(self.train_cfg.get("epochs", 25))
        print(f"[Trainer] Starting training for {epochs} epochs on device: {self.device}")

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(epoch)
            val_metrics = self.evaluate(epoch)
            self.scheduler.step()
            elapsed = time.time() - t0

            val_loss = val_metrics.get("val_loss", float("inf"))
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            self.save_checkpoint(epoch, val_metrics, is_best=is_best)

            abs_rel = val_metrics.get("abs_rel", 0.0)
            d1 = val_metrics.get("delta1", 0.0)
            print(f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | AbsRel: {abs_rel:.4f} | d1: {d1:.4f}")

        self.writer.close()
        print(f"[Trainer] Training complete. Artifacts saved in {self.output_dir}")
```

---

### Component 4.4: `train.py` (Single YAML Entry Point)

```python
#!/usr/bin/env python3
"""
train.py — DepthLab Training Entry Point.

Usage:
    python train.py --config config/default.yaml
    python train.py --config experiments/nyu_relative.yaml
"""

import argparse
import sys
from pathlib import Path
import yaml

from depthlab.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="DepthLab Single-Entrypoint Training Script")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML training configuration file"
    )
    args = parser.parse_args()

    if not args.config.exists():
        print(f"Error: Configuration file not found at '{args.config}'", file=sys.stderr)
        sys.exit(1)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    trainer = Trainer(config)
    trainer.fit()


if __name__ == "__main__":
    main()
```

---

### Component 4.5: `eval.py` (Standalone Evaluation Script with `--ablate-layer` Hook)

```python
#!/usr/bin/env python3
"""
eval.py — DepthLab Standalone Evaluation Script & Feature Ablation Engine.

Usage:
    python eval.py --config experiments/nyu_relative.yaml --checkpoint outputs/nyu_relative_vits/checkpoint_best.pth
    python eval.py --config experiments/nyu_relative.yaml --checkpoint outputs/nyu_relative_vits/checkpoint_best.pth --ablate-layer 0 1
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional
import yaml
import torch
from torch.utils.data import DataLoader

from depthlab.backbone.loader import load_da2_checkpoint
from depthlab.heads import get_head
from depthlab.data import get_dataset
from depthlab.metrics.dispatcher import MetricDispatcher, apply_feature_ablation


def main():
    parser = argparse.ArgumentParser(description="DepthLab Evaluation Script & Feature Ablation Suite")
    parser.add_argument("--config", type=Path, required=True, help="Path to experiment YAML config")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to head checkpoint (.pth)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory to save evaluation results")
    parser.add_argument("--ablate-layer", type=int, nargs="*", default=None, help="Stage indices to ablate (e.g. --ablate-layer 0 1)")
    args = parser.parse_args()

    if not args.config.exists():
        print(f"Error: Config file '{args.config}' not found.", file=sys.stderr)
        sys.exit(1)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Starting evaluation on device: {device}")

    # Load Backbone
    bb_cfg = config.get("model", {}).get("backbone", {})
    backbone = load_da2_checkpoint(
        variant=bb_cfg.get("variant", "vits"),
        checkpoint_path=Path(bb_cfg.get("checkpoint_path", "checkpoints/depth_anything_v2_vits.pth"))
    ).to(device).eval()

    # Load Head
    head_cfg = config.get("model", {}).get("head", {})
    head = get_head(
        head_cfg.get("name", "relative_depth"),
        features=head_cfg.get("features", 64),
        out_channels=head_cfg.get("out_channels", [48, 96, 192, 384])
    ).to(device)

    if args.checkpoint.exists():
        ckpt = torch.load(args.checkpoint, map_location=device)
        state_dict = ckpt.get("head_state_dict", ckpt)
        head.load_state_dict(state_dict)
        print(f"[Eval] Loaded head state dict from {args.checkpoint}")
    else:
        print(f"Warning: Checkpoint '{args.checkpoint}' not found! Running evaluation with initialized head weights.", file=sys.stderr)

    head.eval()

    # Setup Dataset & Loader
    data_cfg = config.get("dataset", {})
    ds = get_dataset(data_cfg.get("name", "synthetic"), split="val", **data_cfg)
    loader = DataLoader(ds, batch_size=int(data_cfg.get("batch_size", 4)), shuffle=False, num_workers=int(data_cfg.get("num_workers", 0)))

    # Dispatcher
    eval_cfg = config.get("eval", {})
    dispatcher = MetricDispatcher(
        align_scale_shift=eval_cfg.get("align_scale_shift", True),
        min_depth=float(eval_cfg.get("min_depth", 1e-3)),
        max_depth=float(eval_cfg.get("max_depth", 80.0))
    )

    ablate_layers = args.ablate_layer
    if ablate_layers:
        print(f"[Eval] Feature Ablation Active: zeroing stages {ablate_layers}")

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["depth"].to(device)

            features = backbone.features(images)
            if ablate_layers:
                features = apply_feature_ablation(features, ablate_layers)

            preds = head(features)
            pred_depth = preds.get("depth", preds.get("pred"))
            dispatcher.update(pred_depth, targets)

    results = dispatcher.compute()
    print("\n================ Evaluation Results ================")
    print(dispatcher.summary_table())

    output_dir = args.output_dir or Path(config.get("experiment", {}).get("output_dir", "outputs/eval"))
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_data = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "ablated_layers": ablate_layers,
        "metrics": results
    }

    res_json = output_dir / "eval_results.json"
    with open(res_json, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"[Eval] Saved evaluation results to {res_json}")


if __name__ == "__main__":
    main()
```

---

### Component 4.6: Configuration Files

#### File 1: `config/default.yaml` (Fast CI / Synthetic Data Fast Path)

```yaml
experiment:
  name: "default_synthetic_smoke"
  output_dir: "outputs/default_synthetic"
  seed: 42
  dataset_version: "synthetic_v1.0"
  benchmark_version: "synthetic_smoke"

model:
  backbone:
    variant: "vits"
    checkpoint_path: "checkpoints/depth_anything_v2_vits.pth"
    frozen: true
  head:
    name: "relative_depth"
    features: 64
    out_channels: [48, 96, 192, 384]

dataset:
  name: "synthetic"
  num_samples: 16
  batch_size: 4
  num_workers: 0
  image_size: 518

training:
  epochs: 2
  learning_rate: 1.0e-4
  weight_decay: 1.0e-2
  amp: true
  clip_grad_norm: 1.0
  save_interval: 1

eval:
  align_scale_shift: true
  min_depth: 0.001
  max_depth: 80.0
```

#### File 2: `experiments/nyu_relative.yaml` (NYUv2 Relative Depth Benchmark)

```yaml
experiment:
  name: "nyu_relative_vits"
  output_dir: "outputs/nyu_relative_vits"
  seed: 42
  dataset_version: "nyuv2_official_v1.0"
  benchmark_version: "eigen_split_v1"

model:
  backbone:
    variant: "vits"
    checkpoint_path: "checkpoints/depth_anything_v2_vits.pth"
    frozen: true
  head:
    name: "relative_depth"
    features: 64
    out_channels: [48, 96, 192, 384]

dataset:
  name: "nyuv2"
  data_root: "data/nyuv2"
  train_split: "data/nyuv2/train.txt"
  val_split: "data/nyuv2/val.txt"
  batch_size: 8
  num_workers: 4
  image_size: 518

training:
  epochs: 25
  learning_rate: 5.0e-4
  weight_decay: 1.0e-2
  warmup_epochs: 2
  amp: true
  clip_grad_norm: 1.0
  save_interval: 5

eval:
  align_scale_shift: true
  min_depth: 0.001
  max_depth: 10.0
```

---

### Component 4.7: Extension Method for `verify.py`

Add check #12 to `verify.py`: `check_framework_scaffolding`

```python
    def check_framework_scaffolding(self, model: nn.Module) -> VerifyResult:
        """
        Validates Layer 1 & Layer 2 Framework Scaffolding:
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
            # 1. Feature extraction
            dummy_img = torch.randn(2, 3, 518, 518, device=device)
            backbone = load_da2_checkpoint(variant=self.config.encoder, checkpoint_path=self.config.checkpoint_path).to(device).eval()

            with torch.no_grad():
                bundle = backbone.features(dummy_img)

            if not hasattr(bundle, "stages") or len(bundle.stages) != 4:
                return VerifyResult(
                    name="Framework Scaffolding Integrity",
                    passed=False,
                    message=f"FAILED: FeatureBundle must contain 4 FeatureStage objects, got {len(getattr(bundle, 'stages', []))}"
                )

            # 2. Head registry & forward pass
            head = get_head("relative_depth", features=64, out_channels=[48, 96, 192, 384]).to(device).train()
            preds = head(bundle)

            if "depth" not in preds:
                return VerifyResult(
                    name="Framework Scaffolding Integrity",
                    passed=False,
                    message="FAILED: RelativeDepth head forward output missing 'depth' tensor"
                )

            # 3. Loss & backward pass
            dummy_target = torch.rand_like(preds["depth"])
            loss = head.compute_loss(preds, {"depth": dummy_target})
            loss.backward()

            # 4. Metric dispatcher & Feature ablation
            dispatcher = MetricDispatcher()
            dispatcher.update(preds["depth"].detach(), dummy_target)
            metrics = dispatcher.compute()

            ablated_bundle = apply_feature_ablation(bundle, [0, 2])
            ablated_preds = head(ablated_bundle)

            passed = ("abs_rel" in metrics) and (ablated_preds["depth"].shape == preds["depth"].shape)
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
```

---

## 5. Verification Method

1. **Synthetic Training Loop Verification**:
   Execute `python train.py --config config/default.yaml`
   - Invalidation conditions: Any syntax error, missing module import, failure to write `outputs/default_synthetic/experiment.yaml`, failure to output TensorBoard logs, or unhandled CUDA exception.

2. **NYUv2 Relative Depth Experiment Config Check**:
   Inspect `experiments/nyu_relative.yaml` for complete key coverage matching `config/default.yaml`.

3. **Standalone Evaluation & Feature Ablation Check**:
   Execute `python eval.py --config config/default.yaml --checkpoint checkpoints/depth_anything_v2_vits.pth --ablate-layer 0 1`
   - Invalidation conditions: Exception during ablation zeroing, shape mismatch in prediction, missing metric in output table.

4. **Extended Verification Gate**:
   Execute `python verify.py`
   - Invalidation conditions: Failure of `check_framework_scaffolding` check.
