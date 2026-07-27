"""
depthlab/trainer.py — Modular Training Harness & Reproducibility Tracker.

Handles AMP FP16 mixed precision training, GradScaler, AdamW optimizer,
Cosine Annealing LR scheduler, TensorBoard logging, checkpointing, and
automatic experiment.yaml metadata recording.
"""

import sys
import time
import random
import hashlib
import datetime
from contextlib import nullcontext
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
except (ImportError, ModuleNotFoundError):
    class SummaryWriter:
        def __init__(self, log_dir=None, **kwargs):
            self.log_dir = log_dir
        def add_scalar(self, tag, scalar_value, global_step=None, **kwargs):
            pass
        def close(self):
            pass

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

from depthlab.backbone.loader import load_da2_checkpoint
from depthlab.heads import get_head
from depthlab.data import get_dataset
from depthlab.metrics.dispatcher import MetricDispatcher


def get_git_commit() -> Dict[str, Any]:
    """Retrieves current git commit hash and status."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, cwd=str(PROJECT_ROOT)
        ).decode("utf-8").strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, cwd=str(PROJECT_ROOT)
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

        # Setup output directory. If no explicit output_dir is supplied, create
        # a spec-compatible runs/EXP-NNN folder.
        self.output_dir = Path(self.exp_cfg.get("output_dir") or self._next_experiment_dir())
        if not self.output_dir.is_absolute():
            self.output_dir = PROJECT_ROOT / self.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.visualization_dir = self.output_dir / "visualizations"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.visualization_dir.mkdir(parents=True, exist_ok=True)

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
        ckpt_path = bb_cfg.get("checkpoint_path", "checkpoints/depth_anything_v2_vits.pth")
        self.backbone = load_da2_checkpoint(
            variant=bb_cfg.get("variant", "vits"),
            checkpoint_path=ckpt_path,
            device=str(self.device),
            freeze=bb_cfg.get("frozen", True)
        )

        # Initialize Head via Registry
        head_cfg = self.model_cfg.get("head", {})
        head_name = head_cfg.get("name", "relative_depth")
        encoder_variant = bb_cfg.get("variant", "vits")
        head_params = dict(head_cfg)
        head_params.pop("name", None)
        head_params.setdefault("encoder_variant", encoder_variant)
        self.head = get_head(head_name, **head_params).to(self.device)

        # Datasets and Loaders
        self.train_loader, self.val_loader = self._setup_dataloaders()

        # Optimizer and Scheduler
        lr = float(self.train_cfg.get("learning_rate", 5e-4))
        weight_decay = float(self.train_cfg.get("weight_decay", 1e-2))
        parameters = list(self.head.parameters()) + [parameter for parameter in self.backbone.parameters() if parameter.requires_grad]
        self.optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)

        epochs = int(self.train_cfg.get("epochs", 25))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)

        # AMP Mixed Precision Scaler
        self.use_amp = bool(self.train_cfg.get("amp", True)) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.best_val_loss = float("inf")
        self.start_epoch = 1

        if self.train_cfg.get("resume", False):
            self._resume_latest()

    @staticmethod
    def _next_experiment_dir() -> Path:
        runs_dir = PROJECT_ROOT / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        existing = []
        for child in runs_dir.glob("EXP-*"):
            try:
                existing.append(int(child.name.split("-")[1]))
            except (IndexError, ValueError):
                continue
        next_id = (max(existing) + 1) if existing else 1
        return Path("runs") / f"EXP-{next_id:03d}"

    def _record_experiment_metadata(self):
        """Generates and writes experiment.yaml into the output directory."""
        git_info = get_git_commit()
        ckpt_path = Path(self.model_cfg.get("backbone", {}).get("checkpoint_path", "checkpoints/depth_anything_v2_vits.pth"))
        if not ckpt_path.is_absolute():
            ckpt_path = PROJECT_ROOT / ckpt_path
        ckpt_hash = compute_file_sha256(ckpt_path)

        start_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        config_snapshot = dict(self.config)
        config_snapshot["_metadata"] = {
            "name": self.exp_cfg.get("name", "unnamed"),
            "start_timestamp": start_timestamp,
            "git_commit": git_info["commit"],
            "git_dirty": git_info["dirty"],
            "checkpoint_path": str(ckpt_path),
            "checkpoint_sha256": ckpt_hash,
            "seed": self.seed,
            "dataset_version": self.exp_cfg.get("dataset_version", self.data_cfg.get("version", "unknown")),
            "benchmark_version": self.exp_cfg.get("benchmark_version", self.config.get("benchmark", {}).get("version", "unknown")),
            "python_version": sys.version.split()[0],
            "pytorch_version": str(torch.__version__),
            "cuda_version": str(torch.version.cuda) if torch.cuda.is_available() and torch.version.cuda else None,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        }

        metadata = {
            "experiment": {
                "name": self.exp_cfg.get("name", "unnamed"),
                "timestamp": start_timestamp,
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
                    "pytorch_version": str(torch.__version__),
                    "cuda_version": str(torch.version.cuda) if torch.cuda.is_available() and torch.version.cuda else None,
                    "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                },
                "config_dump": self.config
            }
        }

        metadata_file = self.output_dir / "experiment.yaml"
        with open(metadata_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(config_snapshot, f, sort_keys=False)
        with open(self.output_dir / "metadata.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(metadata, f, sort_keys=False)
        print(f"[Trainer] Recorded reproducibility metadata to {metadata_file}")

    def _setup_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        data_kwargs = dict(self.data_cfg)
        ds_name = data_kwargs.pop("name", "synthetic")
        train_ds = get_dataset(ds_name, split="train", **data_kwargs)
        val_ds = get_dataset(ds_name, split="val", **data_kwargs)

        batch_size = int(self.data_cfg.get("batch_size", 4))
        num_workers = int(self.data_cfg.get("num_workers", 0))

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_loader, val_loader

    def train_epoch(self, epoch: int) -> float:
        self.head.train()
        backbone_trainable = any(parameter.requires_grad for parameter in self.backbone.parameters())
        self.backbone.train(backbone_trainable)
        total_loss = 0.0

        for step, batch in enumerate(self.train_loader):
            images = batch["image"].to(self.device)
            targets = batch["depth"].to(self.device)
            valid_mask = batch.get("valid_mask")
            if valid_mask is not None:
                valid_mask = valid_mask.to(self.device)

            self.optimizer.zero_grad()

            with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                grad_context = nullcontext() if backbone_trainable else torch.no_grad()
                with grad_context:
                    features = self.backbone.features(images)
                preds = self.head(features)
                loss = self.head.compute_loss(preds, {"depth": targets, "valid_mask": valid_mask})

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.head.parameters(), max_norm=float(self.train_cfg.get("clip_grad_norm", 1.0)))
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.head.parameters(), max_norm=float(self.train_cfg.get("clip_grad_norm", 1.0)))
                self.optimizer.step()

            total_loss += loss.item()
            global_step = (epoch - 1) * len(self.train_loader) + step
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
                valid_mask = batch.get("valid_mask")
                if valid_mask is not None:
                    valid_mask = valid_mask.to(self.device)

                with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                    features = self.backbone.features(images)
                    preds = self.head(features)
                    loss = self.head.compute_loss(preds, {"depth": targets, "valid_mask": valid_mask})

                val_loss += loss.item()
                pred_depth = preds.get("depth", preds.get("predicted_depth"))
                dispatcher.update(pred_depth, targets, valid_mask)

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

        latest_path = self.checkpoint_dir / "checkpoint_latest.pth"
        torch.save(ckpt_data, latest_path)
        backbone_path = self.checkpoint_dir / "backbone.pth"
        if not backbone_path.exists():
            torch.save(self.backbone.official_model.state_dict(), backbone_path)
        torch.save(self.head.state_dict(), self.checkpoint_dir / f"head-{self.model_cfg.get('head', {}).get('name', 'head')}-latest.pth")

        save_int = int(self.train_cfg.get("save_interval", 5))
        if epoch % save_int == 0:
            torch.save(ckpt_data, self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth")

        if is_best:
            best_path = self.checkpoint_dir / "checkpoint_best.pth"
            torch.save(ckpt_data, best_path)
            print(f"[Trainer] Saved new best model to {best_path}")

    def _resume_latest(self):
        latest_path = self.checkpoint_dir / "checkpoint_latest.pth"
        if not latest_path.exists():
            print(f"[Trainer] Resume requested but no checkpoint exists at {latest_path}")
            return
        ckpt = torch.load(latest_path, map_location=self.device)
        self.head.load_state_dict(ckpt["head_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.start_epoch = int(ckpt.get("epoch", 0)) + 1
        self.best_val_loss = float(ckpt.get("metrics", {}).get("val_loss", self.best_val_loss))
        print(f"[Trainer] Resumed from {latest_path} at epoch {self.start_epoch}")

    def fit(self):
        epochs = int(self.train_cfg.get("epochs", 25))
        print(f"[Trainer] Starting training for {epochs} epochs on device: {self.device}")

        for epoch in range(self.start_epoch, epochs + 1):
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
            d1 = val_metrics.get("delta1", val_metrics.get("d1", 0.0))
            print(f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | AbsRel: {abs_rel:.4f} | d1: {d1:.4f}")

        self.writer.close()
        print(f"[Trainer] Training complete. Artifacts saved in {self.output_dir}")
