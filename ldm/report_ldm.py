"""Lightweight run figures and comparison exports."""

from __future__ import annotations

import csv
import json
import math
import textwrap
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


def save_test_accuracy_comparison(run_dirs: list[str | Path], output: str | Path) -> Path:
    """Plot completed runs with explicit dataset/metric comparison groups."""
    import matplotlib.pyplot as plt

    rows = [row for row in compare_runs(run_dirs)
            if isinstance(row['test_token_accuracy'], (float, int))
            and math.isfinite(row['test_token_accuracy'])]
    if not rows:
        raise ValueError('No finite test accuracy values to plot')
    group_ids = {}
    for row in rows:
        identity = (row['dataset_fingerprint'], row['split_version'], row['metric_version'])
        if row['split_version'] is None or row['metric_version'] is None:
            identity = ('unverified', row['run_id'])
        row['comparison_group'] = group_ids.setdefault(identity, len(group_ids) + 1)
    figure, ax = plt.subplots(figsize=(12, max(4, len(rows) * 1.2)))
    colors = [plt.get_cmap('tab10')((row['comparison_group'] - 1) % 10) for row in rows]
    bars = ax.barh(range(len(rows)), [100 * row['test_token_accuracy'] for row in rows], color=colors)
    labels = [textwrap.fill(row['run_id'], width=45, break_on_hyphens=False) +
              f"\nGroup {row['comparison_group']}" for row in rows]
    ax.set_yticks(range(len(rows)), labels, fontsize=9)
    ax.invert_yaxis()
    ax.bar_label(bars, fmt='%.2f%%', padding=3)
    ax.set(xlabel='Test token accuracy (%)', xlim=(0, 105), title='Best-checkpoint test accuracy')
    ax.grid(axis='x', alpha=0.25)
    figure.text(0.01, 0.01, 'Compare within the same group: matching dataset, split, and metric versions.', fontsize=9)
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150, bbox_inches='tight')
    plt.close(figure)
    with output.with_suffix('.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output
