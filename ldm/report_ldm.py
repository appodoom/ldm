"""Lightweight run figures and comparison exports."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .experiment_workflow import compare_runs


def save_history_plot(run_dir: str | Path) -> Path:
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    history = metrics["history"]
    if not history:
        raise ValueError("No validation history")
    figure, ax = plt.subplots()
    training_path = run_dir / "training_history.jsonl"
    if training_path.exists():
        training = [json.loads(line) for line in training_path.read_text(encoding="utf-8").splitlines()]
        if training:
            ax.plot([point["step"] for point in training],
                    [point["loss"] for point in training], alpha=0.6, label="Training")
    ax.plot([point["step"] for point in history],
            [point["val_loss"] for point in history], marker="o", label="Validation")
    ax.set(xlabel="Optimizer step", ylabel="Loss", title=run_dir.name)
    ax.legend()
    ax.grid(True)
    target = run_dir / "validation_loss.png"
    figure.savefig(target, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return target


def save_comparison(run_dirs: list[str | Path], output: str | Path) -> Path:
    rows = compare_runs(run_dirs)
    if not rows:
        raise ValueError("No runs selected")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output
