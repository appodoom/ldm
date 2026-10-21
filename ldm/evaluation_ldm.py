"""Unique-target, atomic-token evaluation for prepared LDM chunks."""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F


METRIC_VERSION = 2


from .metric_counts import TokenCounts, _ratio


@torch.no_grad()
def evaluate_model(model, dataset, batch_size: int, device: torch.device) -> dict:
    """Evaluate each owned target once; optional old masks produce N/A metrics."""
    model.eval()
    vocab = list(model.vocab)
    groups = {"combined": TokenCounts(vocab), "ordinary": TokenCounts(vocab),
              "decorated": TokenCounts(vocab)}
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
        kinds = np.asarray(dataset.source_kind[rows])
        skeleton = np.asarray(dataset.skeleton[rows]) if dataset.skeleton is not None else None
        hit_count = np.asarray(dataset.hit_count[rows])
        input_ids = batch["input_ids"].cpu().numpy()
        if skeleton is not None and dataset.source_file is not None:
            for row_index, file_index in enumerate(np.asarray(dataset.source_file[rows])):
                key = dataset.file_names[int(file_index)]
                counts = skeleton_by_file.setdefault(key, {"correct": 0, "total": 0})
                selected = skeleton[row_index] & (targets[row_index] != -100)
                counts["total"] += int(selected.sum())
                counts["correct"] += int(((targets[row_index] == preds[row_index]) & selected).sum())
        for group_name, group in groups.items():
            rows_for_group = (np.ones(len(rows), dtype=bool) if group_name == "combined"
                              else kinds == (0 if group_name == "ordinary" else 1))
            valid = (targets != -100) & rows_for_group[:, None]
            skeleton_valid = (skeleton & valid) if skeleton is not None else np.zeros_like(valid)
            group.add(targets[valid], preds[valid], loss_values[valid], skeleton_valid[valid])
            if model.config.dataset_variant == "with_subd":
                for row_index, position in np.argwhere(hit_count & rows_for_group[:, None]):
                    token = vocab[int(input_ids[row_index, position])]
                    if not token.startswith("SUBD_"):
                        continue
                    expected = int(token[5:])
                    predicted = preds[row_index, position:]
                    actual = 0
                    for pred_id in predicted:
                        if not group.hit_ids[pred_id]:
                            break
                        actual += 1
                    group.hit_count_total += 1
                    group.hit_count_correct += int(actual == expected)
        valid_count = int((targets != -100).sum())
        for layer_index, block in enumerate(model.core.blocks, 1):
            gate = getattr(block, "last_gate_usage", None)
            if gate is not None:
                gate_sum[layer_index] = gate_sum.get(layer_index, np.zeros(4)) + (
                    gate.float().cpu().numpy() * valid_count
                )
        gate_weight += valid_count
    if groups["combined"].total == 0:
        raise ValueError("Evaluation split has no scored targets")
    report = {name: group.result() for name, group in groups.items()}
    report["metric_version"] = METRIC_VERSION
    report["structured_gate_usage"] = ({str(index): (values / gate_weight).tolist()
                                          for index, values in gate_sum.items()}
                                         if gate_weight else {})
    report["skeleton_by_file"] = {name: {**counts,
                                        "accuracy": _ratio(counts["correct"], counts["total"])}
                                  for name, counts in skeleton_by_file.items() if counts["total"]}
    return report
