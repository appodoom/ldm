"""Pure NumPy counters for unique-target atomic rhythm metrics."""

import numpy as np

from .constants import DECORATED_SUFFIX


def _ratio(numerator: int | float, denominator: int | float):
    return float(numerator / denominator) if denominator else None


def _prf(tp: int, predicted: int, target: int) -> dict:
    precision, recall = _ratio(tp, predicted), _ratio(tp, target)
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None and precision + recall else
          0.0 if precision is not None and recall is not None else None)
    return {"precision": precision, "recall": recall, "f1": f1,
            "true_positive": int(tp), "predicted": int(predicted), "target": int(target)}


class TokenCounts:
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
        self.decorated_plain_correct = 0
        self.decoration_outcomes = {}
        self.skeleton_total = self.skeleton_correct = 0
        self.hit_count_total = self.hit_count_correct = 0
        self.hit_ids = np.array([tok.startswith("HIT_") for tok in vocab])
        self.structure_ids = np.array([tok in {"<SOC>", "<EOC>", "<SOB>", "<EOB>"}
                                       or tok.startswith("SUBD_") for tok in vocab])
        self.decorated_ids = np.array([tok.endswith(DECORATED_SUFFIX) for tok in vocab])
        self.plain_ids = np.array([vocab.index(tok.removesuffix(DECORATED_SUFFIX))
                                   if tok.endswith(DECORATED_SUFFIX) and tok.removesuffix(DECORATED_SUFFIX) in vocab
                                   else -1 for tok in vocab])

    def add(self, targets: np.ndarray, predictions: np.ndarray, losses: np.ndarray,
            skeleton: np.ndarray) -> None:
        if not len(targets):
            return
        n = len(self.vocab)
        matrix = np.bincount(targets * n + predictions, minlength=n * n).reshape(n, n)
        self.add_confusion(matrix)
        self.loss_sum += float(losses.sum())
        self.skeleton_total += int(skeleton.sum())
        self.skeleton_correct += int(((targets == predictions) & skeleton).sum())

    def add_confusion(self, matrix: np.ndarray) -> None:
        """Shared metric definitions for NumPy and device-accumulated counts."""
        targets, predictions, correct = matrix.sum(1), matrix.sum(0), matrix.diagonal()
        self.targets += targets
        self.predictions += predictions
        self.true_positives += correct
        self.total += int(targets.sum())
        self.correct += int(correct.sum())
        for name, selected in (("hit", self.hit_ids), ("structure", self.structure_ids),
                               ("decorated", self.decorated_ids)):
            setattr(self, name + "_total", getattr(self, name + "_total") + int(targets[selected].sum()))
            setattr(self, name + "_correct", getattr(self, name + "_correct") + int(correct[selected].sum()))
        for name, selected in (("hit", self.hit_ids), ("decorated", self.decorated_ids)):
            setattr(self, name + "_predicted", getattr(self, name + "_predicted") + int(predictions[selected].sum()))
            setattr(self, name + "_presence_tp", getattr(self, name + "_presence_tp") +
                    int(matrix[np.ix_(selected, selected)].sum()))
        wrong_total = plain_total = 0
        for token_id in np.flatnonzero(self.decorated_ids):
            total = int(targets[token_id])
            if not total:
                continue
            exact = int(correct[token_id])
            wrong = int(matrix[token_id, self.decorated_ids].sum()) - exact
            plain_id = self.plain_ids[token_id]
            plain = int(matrix[token_id, plain_id]) if plain_id >= 0 else 0
            values = {"total": total, "exact_decorated": exact, "wrong_decorated_type": wrong,
                      "correct_plain_hit": plain, "other": total - exact - wrong - plain}
            saved = self.decoration_outcomes.setdefault(self.vocab[token_id], dict.fromkeys(values, 0))
            for key, value in values.items():
                saved[key] += value
            wrong_total += wrong
            plain_total += plain
        self.decorated_wrong_type += wrong_total
        self.decorated_plain_correct += plain_total

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
                            if token.endswith(DECORATED_SUFFIX)}
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
            "decorated_wrong_type_rate": _ratio(self.decorated_wrong_type, self.decorated_total),
            "decorated_plain_hit_correct": self.decorated_plain_correct,
            "decorated_plain_hit_accuracy": _ratio(self.decorated_plain_correct, self.decorated_total),
            "decorated_outcomes": {'total': self.decorated_total,
                'exact_decorated': self.decorated_correct, 'wrong_decorated_type': self.decorated_wrong_type,
                'correct_plain_hit': self.decorated_plain_correct,
                'other': self.decorated_total - self.decorated_correct - self.decorated_wrong_type - self.decorated_plain_correct},
            "decorated_outcomes_by_type": self.decoration_outcomes,
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


