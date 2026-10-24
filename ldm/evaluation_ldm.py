"""Unique-target, atomic-token evaluation for prepared LDM chunks."""

from __future__ import annotations

import torch
from torch.nn import functional as F


METRIC_VERSION = 3


from .metric_counts import TokenCounts, _ratio
from .constants import MAX_SUBDIVISION


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


class DeviceCounts:
    """Fixed-size counters; full predictions never leave the evaluation device."""

    def __init__(self, vocab, file_names, device):
        self.counts = TokenCounts(vocab)
        self.file_names = file_names
        n = len(vocab)
        self.confusion = torch.zeros(n * n, dtype=torch.long, device=device)
        self.loss = torch.zeros((), dtype=torch.float64, device=device)
        # Overall skeleton correct/total, then hit-count correct/total.
        self.totals = torch.zeros(4, dtype=torch.long, device=device)
        self.files = torch.zeros((2, len(file_names)), dtype=torch.long, device=device)
        self.hit_ids = torch.as_tensor(self.counts.hit_ids, device=device)
        self.subdivisions = torch.tensor([int(token[5:]) if token.startswith("SUBD_") else 0
                                         for token in vocab], device=device)

    def add(self, input_ids, targets, predictions, losses, metadata, with_subd):
        valid = targets != -100
        safe_targets = targets.clamp_min(0)
        indices = (safe_targets * len(self.counts.vocab) + predictions).flatten()
        self.confusion.scatter_add_(0, indices, valid.flatten().long())
        self.loss += losses.float().sum().double()
        correct = targets == predictions
        skeleton = metadata.get("skeleton")
        skeleton = torch.zeros_like(valid) if skeleton is None else skeleton & valid
        self.totals[0] += (correct & skeleton).sum()
        self.totals[1] += skeleton.sum()
        if "source_file" in metadata:
            self.files[0].scatter_add_(0, metadata["source_file"], (correct & skeleton).sum(1))
            self.files[1].scatter_add_(0, metadata["source_file"], skeleton.sum(1))
        if with_subd:
            expected = self.subdivisions[input_ids]
            selected = metadata["hit_count"] & (expected > 0)
            predicted_hits = self.hit_ids[predictions]
            # Cap at max subdivision + 1: longer runs are necessarily incorrect.
            alive = torch.ones_like(predicted_hits)
            actual = torch.zeros_like(predictions)
            for offset in range(min(MAX_SUBDIVISION + 1, predictions.shape[1])):
                shifted = F.pad(predicted_hits[:, offset:], (0, offset), value=False)
                alive = alive & shifted
                actual += alive
            self.totals[2] += ((actual == expected) & selected).sum()
            self.totals[3] += selected.sum()

    def result(self):
        counts = TokenCounts(self.counts.vocab)
        counts.add_confusion(self.confusion.cpu().numpy().reshape(len(counts.vocab), -1))
        counts.loss_sum = self.loss.item()
        (counts.skeleton_correct, counts.skeleton_total,
         counts.hit_count_correct, counts.hit_count_total) = self.totals.cpu().tolist()
        files = self.files.cpu().numpy()
        by_file = {name: {"correct": int(files[0, i]), "total": int(files[1, i]),
                          "accuracy": _ratio(int(files[0, i]), int(files[1, i]))}
                   for i, name in enumerate(self.file_names) if files[1, i]}
        if not counts.total:
            raise ValueError("Evaluation split has no scored targets")
        return counts.result(), by_file


@torch.no_grad()
def evaluate_model(model, dataset, batch_size: int, device: torch.device, *, loader=None) -> dict:
    """Evaluate owned targets using ordered loading and device-side counters."""
    from .data_loading import evaluation_loader, batch_to_device, close_loader

    if not dataset.evaluation:
        raise ValueError("Evaluation requires ChunkDataset(evaluation=True)")
    model.eval()
    counts = DeviceCounts(list(model.vocab), dataset.file_names, device)
    gate_sum = {}
    gate_weight = torch.zeros((), dtype=torch.long, device=device)
    owned_loader = loader is None
    loader = evaluation_loader(dataset, batch_size, device) if owned_loader else loader
    iterator = None
    try:
        iterator = iter(loader)
        for cpu_batch in iterator:
            # Check ownership on the CPU before transfer, avoiding a CUDA sync.
            if not bool((cpu_batch["labels"] != -100).any()):
                continue
            batch = batch_to_device(cpu_batch, device)
            metadata = batch.pop("evaluation")
            labels = batch.pop("labels")
            logits = model(**batch).logits
            losses = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
                                     ignore_index=-100, reduction="none").reshape_as(labels)
            counts.add(batch["input_ids"], labels, logits.argmax(dim=-1), losses, metadata,
                       model.config.dataset_variant == "with_subd")
            valid_count = (labels != -100).sum()
            for layer_index, block in enumerate(model.core.blocks, 1):
                gate = getattr(block, "last_gate_usage", None)
                if gate is not None:
                    gate_sum[layer_index] = gate_sum.get(layer_index, torch.zeros_like(gate)) + (
                        gate.float() * valid_count)
            gate_weight += valid_count
    finally:
        if owned_loader or not loader.persistent_workers:
            close_loader(loader, iterator)
    combined, by_file = counts.result()
    weight = gate_weight.item()
    return {"combined": combined, "decoration": model.config.dataset_decoration,
            "metric_version": METRIC_VERSION, "skeleton_by_file": by_file,
            "structured_gate_usage": {str(index): (values / weight).cpu().tolist()
                                      for index, values in gate_sum.items()} if weight else {}}
