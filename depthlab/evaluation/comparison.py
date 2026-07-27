"""Stable CSV/Markdown/LaTeX table generation for multi-method evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict


def comparison_table(results: Dict[str, Dict[str, Dict[str, float]]], output: str | Path) -> None:
    newline = r"\\"
    lines = [r"\begin{tabular}{llrrrrrr}", "Dataset & Method & AbsRel & RMSE & $\\delta_1$ & ECE & AURC & NLL " + newline, r"\hline"]
    for dataset, methods in results.items():
        for method, metrics in methods.items():
            lines.append(f"{dataset} & {method} & {metrics.get('abs_rel', 0):.4f} & {metrics.get('rmse', 0):.4f} & {metrics.get('d1', metrics.get('delta1', 0)):.4f} & {metrics.get('ece', 0):.4f} & {metrics.get('aurc', 0):.4f} & {metrics.get('nll', 0):.4f} " + newline)
    lines.append(r"\end{tabular}")
    Path(output).write_text("\n".join(lines) + "\n", encoding="utf-8")
