"""Train the promptable scale predictor on a precomputed feature manifest."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import PromptableScaleDataset
from .scale_predictor import ScalePredictor


def train_scale_predictor(manifest: str | Path, output: str | Path, epochs: int = 50,
                          batch_size: int = 64, learning_rate: float = 3e-4,
                          device: str = "cpu") -> ScalePredictor:
    dataset = PromptableScaleDataset(manifest)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    input_dim = int(dataset[0]["features"].numel())
    model = ScalePredictor(input_dim=input_dim, use_patch_features=input_dim > 768).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _ in range(epochs):
        model.train()
        for batch in loader:
            prediction = model(batch["features"].to(device))
            loss = torch.nn.functional.l1_loss(prediction, batch["scale"].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.save(output)
    return model
