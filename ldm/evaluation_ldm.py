"""Unique-target, atomic-token evaluation for prepared LDM chunks."""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F


METRIC_VERSION = 3


from .metric_counts import TokenCounts, _ratio


def format_percentage(value: float | None) -> str:
    return "N/A" if value is None else f"{100 * value:.2f}%"


def print_evaluation_metrics(report: dict, split: str) -> None:
    """Print the saved evaluation measures without recalculating any metrics."""
    metrics = report["combined"]

    names = (("token", "token_accuracy"), ("HIT", "hit_accuracy"),
             ("structure", "structure_accuracy"), ("hit count", "hit_count_accuracy"),
             ("skeleton HIT", "skeleton_hit_accuracy"))
    summary = " | ".join(f"{name} {format_percentage(metrics[key])}" for name, key in names)
    print(f"{split} loss {metrics['loss']:.4f} | {summary}", flush=True)
    print(f"{split} structural tokens (accuracy, precision, recall, F1, target count):", flush=True)
    for token, values in metrics["structure_by_type"].items():
        print(f"  {token}: accuracy={format_percentage(values['accuracy'])} "
              f"precision={format_percentage(values['precision'])} recall={format_percentage(values['recall'])} "
              f"F1={format_percentage(values['f1'])} targets={values['target']}", flush=True)
    outcomes = metrics["decorated_outcomes"]
    if outcomes["total"]:
        print(f"{split} decoration ({outcomes['total']} targets): "
              f"exact={format_percentage(metrics['decorated_accuracy'])}, "
              f"wrong type={format_percentage(metrics['decorated_wrong_type_rate'])}, "
              f"correct plain HIT={format_percentage(metrics['decorated_plain_hit_accuracy'])} | "
              f"counts: exact={outcomes['exact_decorated']}, "
              f"wrong type={outcomes['wrong_decorated_type']}, "
              f"plain HIT={outcomes['correct_plain_hit']}, other={outcomes['other']}", flush=True)


@torch.no_grad()
def evaluate_model(model, dataset, batch_size: int, device: torch.device) -> dict:
    """Evaluate each owned target once; optional old masks produce N/A metrics."""
    model.eval()
    vocab = list(model.vocab)
    counts = TokenCounts(vocab)
    gate_sum: dict[int, np.ndarray] = {}
    gate_weight = 0
    skeleton_by_file: dict[str, dict[str, int]] = {}
    for start in range(0, len(dataset), batch_size):
        rows = list(range(start, min(start + batch_size, len(dataset))))
        batch = dataset.batch(rows, device)
        labels = batch["labels"]
        if not bool((labels != -100).any()):
            continue
        outputs = model(**{key: value for key, value in batch.items() if key != "labels"})
        logits = outputs.logits
        losses = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
                                 ignore_index=-100, reduction="none").reshape_as(labels)
        preds = logits.argmax(dim=-1).cpu().numpy()
        targets = labels.cpu().numpy()
        loss_values = losses.float().cpu().numpy()
        skeleton = np.asarray(dataset.skeleton[rows]) if dataset.skeleton is not None else None
        hit_count = np.asarray(dataset.hit_count[rows])
        input_ids = batch["input_ids"].cpu().numpy()
        if skeleton is not None and dataset.source_file is not None:
            for row_index, file_index in enumerate(np.asarray(dataset.source_file[rows])):
                key = dataset.file_names[int(file_index)]
                file_counts = skeleton_by_file.setdefault(key, {"correct": 0, "total": 0})
                selected = skeleton[row_index] & (targets[row_index] != -100)
                file_counts["total"] += int(selected.sum())
                file_counts["correct"] += int(((targets[row_index] == preds[row_index]) & selected).sum())
        valid = targets != -100
        skeleton_valid = (skeleton & valid) if skeleton is not None else np.zeros_like(valid)
        counts.add(targets[valid], preds[valid], loss_values[valid], skeleton_valid[valid])
        if model.config.dataset_variant == "with_subd":
            for row_index, position in np.argwhere(hit_count):
                token = vocab[int(input_ids[row_index, position])]
                if not token.startswith("SUBD_"):
                    continue
                expected = int(token[5:])
                predicted = preds[row_index, position:]
                actual = 0
                for pred_id in predicted:
                    if not counts.hit_ids[pred_id]:
                        break
                    actual += 1
                counts.hit_count_total += 1
                counts.hit_count_correct += int(actual == expected)
        valid_count = int((targets != -100).sum())
        for layer_index, block in enumerate(model.core.blocks, 1):
            gate = getattr(block, "last_gate_usage", None)
            if gate is not None:
                gate_sum[layer_index] = gate_sum.get(layer_index, np.zeros(4)) + (
                    gate.float().cpu().numpy() * valid_count
                )
        gate_weight += valid_count
    if counts.total == 0:
        raise ValueError("Evaluation split has no scored targets")
    report = {"combined": counts.result(), "decoration": model.config.dataset_decoration}
    report["metric_version"] = METRIC_VERSION
    report["structured_gate_usage"] = ({str(index): (values / gate_weight).tolist()
                                          for index, values in gate_sum.items()}
                                         if gate_weight else {})
    report["skeleton_by_file"] = {name: {**counts,
                                        "accuracy": _ratio(counts["correct"], counts["total"])}
                                  for name, counts in skeleton_by_file.items() if counts["total"]}
    return report
