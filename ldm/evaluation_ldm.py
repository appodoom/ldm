"""Unique-target, atomic-token evaluation for prepared LDM chunks."""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F


METRIC_VERSION = 1


def _ratio(numerator: int | float, denominator: int | float):
    return float(numerator / denominator) if denominator else None


def _prf(tp: int, predicted: int, target: int) -> dict:
    precision, recall = _ratio(tp, predicted), _ratio(tp, target)
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None and precision + recall else
          0.0 if precision is not None and recall is not None else None)
    return {"precision": precision, "recall": recall, "f1": f1,
            "true_positive": int(tp), "predicted": int(predicted), "target": int(target)}


class _Counts:
    def __init__(self, vocab: list[str]):
        self.vocab = vocab
        n = len(vocab)
        self.targets = np.zeros(n, dtype=np.int64)
        self.predictions = np.zeros(n, dtype=np.int64)
        self.true_positives = np.zeros(n, dtype=np.int64)
        self.loss_sum = 0.0
        self.total = 0
        self.correct = 0
        self.hit_total = self.hit_correct = self.hit_predicted = self.hit_presence_tp = 0
        self.structure_total = self.structure_correct = 0
        self.decorated_total = self.decorated_correct = self.decorated_predicted = 0
        self.decorated_presence_tp = self.decorated_wrong_type = 0
        self.skeleton_total = self.skeleton_correct = 0
        self.hit_count_total = self.hit_count_correct = 0
        self.hit_ids = np.array([tok.startswith("HIT_") for tok in vocab])
        self.structure_ids = np.array([tok in {"<SOC>", "<EOC>", "<SOB>", "<EOB>"}
                                       or tok.startswith("SUBD_") for tok in vocab])
        self.decorated_ids = np.array([tok.endswith("_decorated") for tok in vocab])

    def add(self, targets: np.ndarray, predictions: np.ndarray, losses: np.ndarray,
            skeleton: np.ndarray) -> None:
        if not len(targets):
            return
        correct = targets == predictions
        self.total += len(targets)
        self.correct += int(correct.sum())
        self.loss_sum += float(losses.sum())
        self.targets += np.bincount(targets, minlength=len(self.vocab))
        self.predictions += np.bincount(predictions, minlength=len(self.vocab))
        self.true_positives += np.bincount(targets[correct], minlength=len(self.vocab))
        hit_target, hit_pred = self.hit_ids[targets], self.hit_ids[predictions]
        self.hit_total += int(hit_target.sum())
        self.hit_predicted += int(hit_pred.sum())
        self.hit_correct += int((correct & hit_target).sum())
        self.hit_presence_tp += int((hit_target & hit_pred).sum())
        structure = self.structure_ids[targets]
        self.structure_total += int(structure.sum())
        self.structure_correct += int((correct & structure).sum())
        decorated_target = self.decorated_ids[targets]
        decorated_pred = self.decorated_ids[predictions]
        self.decorated_total += int(decorated_target.sum())
        self.decorated_predicted += int(decorated_pred.sum())
        self.decorated_correct += int((correct & decorated_target).sum())
        self.decorated_presence_tp += int((decorated_target & decorated_pred).sum())
        self.decorated_wrong_type += int((decorated_target & decorated_pred & ~correct).sum())
        self.skeleton_total += int(skeleton.sum())
        self.skeleton_correct += int((correct & skeleton).sum())

    def result(self) -> dict:
        per_token = {}
        precisions, recalls, f1s = [], [], []
        weighted_precision = weighted_recall = weighted_f1 = 0.0
        for i, token in enumerate(self.vocab):
            support, predicted, tp = (int(self.targets[i]), int(self.predictions[i]),
                                      int(self.true_positives[i]))
            values = _prf(tp, predicted, support)
            values["accuracy"] = _ratio(tp, support)
            per_token[token] = values
            if support or predicted:
                p, r, f = values["precision"] or 0.0, values["recall"] or 0.0, values["f1"] or 0.0
                precisions.append(p)
                recalls.append(r)
                f1s.append(f)
                weighted_precision += support * p
                weighted_recall += support * r
                weighted_f1 += support * f
        ornament_by_type = {token: per_token[token] for token in self.vocab
                            if token.endswith("_decorated")}
        structure_by_type = {token: per_token[token] for token in self.vocab
                             if token in {"<SOC>", "<EOC>", "<SOB>", "<EOB>"}
                             or token.startswith("SUBD_")}
        return {
            "loss": _ratio(self.loss_sum, self.total),
            "target_count": self.total,
            "token_accuracy": _ratio(self.correct, self.total),
            "hit_accuracy": _ratio(self.hit_correct, self.hit_total),
            "hit_presence": _prf(self.hit_presence_tp, self.hit_predicted, self.hit_total),
            "hit_token": _prf(self.hit_correct, self.hit_predicted, self.hit_total),
            "structure_accuracy": _ratio(self.structure_correct, self.structure_total),
            "hit_count_accuracy": _ratio(self.hit_count_correct, self.hit_count_total),
            "hit_count_total": self.hit_count_total,
            "decorated_accuracy": _ratio(self.decorated_correct, self.decorated_total),
            "decorated_wrong_type": self.decorated_wrong_type,
            "decorated_presence": _prf(self.decorated_presence_tp,
                                       self.decorated_predicted, self.decorated_total),
            "decorated_by_type": ornament_by_type,
            "skeleton_hit_accuracy": _ratio(self.skeleton_correct, self.skeleton_total),
            "skeleton_hit_total": self.skeleton_total,
            "token_macro": {"precision": _ratio(sum(precisions), len(precisions)),
                            "recall": _ratio(sum(recalls), len(recalls)),
                            "f1": _ratio(sum(f1s), len(f1s))},
            "token_weighted": {"precision": _ratio(weighted_precision, self.total),
                               "recall": _ratio(weighted_recall, self.total),
                               "f1": _ratio(weighted_f1, self.total)},
            "structure_by_type": structure_by_type,
            "per_token": per_token,
        }


@torch.no_grad()
def evaluate_model(model, dataset, batch_size: int, device: torch.device) -> dict:
    """Evaluate each owned target once; optional old masks produce N/A metrics."""
    model.eval()
    vocab = list(model.vocab)
    groups = {"combined": _Counts(vocab), "ordinary": _Counts(vocab),
              "decorated": _Counts(vocab)}
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
