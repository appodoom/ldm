"""Pure NumPy counters for unique-target atomic rhythm metrics."""

import numpy as np


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
        self.decorated_ids = np.array([tok.endswith("_decorated") for tok in vocab])
        self.plain_ids = np.array([vocab.index(tok.removesuffix('_decorated'))
                                   if tok.endswith('_decorated') and tok.removesuffix('_decorated') in vocab
                                   else -1 for tok in vocab])

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
        plain_correct = decorated_target & (predictions == self.plain_ids[targets])
        self.decorated_plain_correct += int(plain_correct.sum())
        for token_id in np.unique(targets[decorated_target]):
            selected = targets == token_id
            counts = self.decoration_outcomes.setdefault(self.vocab[token_id],
                {'total': 0, 'exact_decorated': 0, 'wrong_decorated_type': 0, 'correct_plain_hit': 0, 'other': 0})
            exact = int((selected & correct).sum())
            wrong = int((selected & decorated_pred & ~correct).sum())
            plain = int((selected & plain_correct).sum())
            total = int(selected.sum())
            counts['total'] += total
            counts['exact_decorated'] += exact
            counts['wrong_decorated_type'] += wrong
            counts['correct_plain_hit'] += plain
            counts['other'] += total - exact - wrong - plain
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


