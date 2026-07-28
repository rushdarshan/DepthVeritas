"""MC-dropout baseline without permanently changing model training mode."""

from __future__ import annotations

from typing import Callable, Dict

import torch
import torch.nn as nn


@torch.no_grad()
def mc_dropout_prediction(model: nn.Module, inputs: torch.Tensor, runs: int = 20,
                          extract: Callable[[object], torch.Tensor] | None = None) -> Dict[str, torch.Tensor]:
    extract = extract or (lambda output: output["depth"] if isinstance(output, dict) else output)
    dropout = [module for module in model.modules() if isinstance(module, nn.Dropout)]
    prior = [module.training for module in dropout]
    for module in dropout:
        module.train()
    try:
        predictions = torch.stack([extract(model(inputs)) for _ in range(runs)])
    finally:
        for module, was_training in zip(dropout, prior):
            module.train(was_training)
    return {"depth": predictions.mean(0), "uncertainty": predictions.var(0, unbiased=False)}
