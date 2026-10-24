"""Memory-mapped prepared data and CPU batch preparation for GPU training."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .rhythm_ldm import STRUCTURED_ROUTING_FIELDS
from .validation import validate_rules


class ChunkDataset:
    def __init__(self, folder: str | Path, split: str, *, model_config=None,
                 evaluation: bool = False):
        folder = Path(folder)
        self.folder, self.split = folder, split
        self.evaluation = evaluation
        self.structured = model_config is None or model_config.attention_layout != "global"
        self.ids = np.load(folder / f"{split}.npy", mmap_mode="r")
        file_path = folder / f"{split}_source_file.npy"
        self.source_file = np.load(file_path, mmap_mode="r") if evaluation and file_path.exists() else None
        self.file_names = [entry["file"] for entry in json.loads(
            (folder / "dataset_manifest.json").read_text(encoding="utf-8"))["files"]]
        self.beat_idx = np.load(folder / f"{split}_beat_idx.npy", mmap_mode="r")
        self.beat_pos = np.load(folder / f"{split}_beat_pos.npy", mmap_mode="r")
        self.score = np.load(folder / f"{split}_score_mask.npy", mmap_mode="r")
        self.hit_count = (np.load(folder / f"{split}_hit_count_mask.npy", mmap_mode="r")
                          if evaluation else None)
        skeleton_path = folder / f"{split}_skeleton_mask.npy"
        self.skeleton = np.load(skeleton_path, mmap_mode="r") if evaluation and skeleton_path.exists() else None
        self.routing = {field: np.load(folder / f"{split}_{field}.npy", mmap_mode="r")
                        for field in STRUCTURED_ROUTING_FIELDS} if self.structured else {}
        validate_rules({
            "arrays": (all(len(array) == len(self.ids) for array in
                       (self.beat_idx, self.beat_pos, self.score, *self.routing.values())),
                       "Prepared array row counts disagree"),
            "hit_count_mask": (self.hit_count is None or len(self.hit_count) == len(self.ids),
                               "Hit-count mask row count disagrees"),
            "skeleton_mask": (self.skeleton is None or len(self.skeleton) == len(self.ids),
                              "Skeleton mask row count disagrees"),
            "source_file": (self.source_file is None or len(self.source_file) == len(self.ids),
                            "Source-file row count disagrees"),
        })

    def __getstate__(self):
        # Reopen memory maps in spawned workers instead of pickling array contents.
        return {"folder": self.folder, "split": self.split, "evaluation": self.evaluation,
                "structured": self.structured}

    def __setstate__(self, state):
        from types import SimpleNamespace
        self.__init__(state["folder"], state["split"], evaluation=state["evaluation"],
                      model_config=SimpleNamespace(attention_layout=
                          "alternating_global_structured" if state["structured"] else "global"))

    def __getitem__(self, rows):
        # The sampler supplies a whole batch for vectorized NumPy reads.
        return self.batch(rows, torch.device("cpu"))

    def __len__(self):
        return len(self.ids)

    def batch(self, rows, device: torch.device):
        rows = list(map(int, rows))
        tokens = torch.as_tensor(np.asarray(self.ids[rows]), dtype=torch.long, device=device)
        labels = tokens[:, 1:].clone()
        score = torch.as_tensor(np.asarray(self.score[rows]), dtype=torch.bool, device=device)
        labels[~score] = -100
        routing = {field: torch.as_tensor(np.asarray(array[rows, :-1]),
                                          dtype=torch.bool if field.endswith("_mask") else torch.long,
                                          device=device)
                   for field, array in self.routing.items()}
        batch = {"input_ids": tokens[:, :-1], "labels": labels,
                "beat_idx_ids": torch.as_tensor(np.asarray(self.beat_idx[rows, :-1]),
                                                dtype=torch.long, device=device),
                "beat_pos_ids": torch.as_tensor(np.asarray(self.beat_pos[rows, :-1]),
                                                dtype=torch.long, device=device),
                }
        if self.structured:
            batch["structured_routing"] = routing
        if self.evaluation:
            metadata = {"hit_count": torch.as_tensor(np.asarray(self.hit_count[rows]),
                                                    dtype=torch.bool, device=device)}
            if self.skeleton is not None:
                metadata["skeleton"] = torch.as_tensor(np.asarray(self.skeleton[rows]),
                                                       dtype=torch.bool, device=device)
            if self.source_file is not None:
                metadata["source_file"] = torch.as_tensor(np.asarray(self.source_file[rows]),
                                                          dtype=torch.long, device=device)
            batch["evaluation"] = metadata
        return batch


def chunk_loader(dataset, batches: list, *, device: torch.device, num_workers: int = 2,
                 pin_memory: bool = True, persistent_workers: bool = True,
                 prefetch_factor: int = 2, seed: int = 1337):
    """Use an independent RNG so worker startup cannot alter model/dropout randomness."""
    if num_workers < 0 or prefetch_factor < 1:
        raise ValueError("num_workers must be nonnegative; prefetch_factor must be positive")
    workers = num_workers if device.type == "cuda" else 0
    options = dict(batch_size=None, sampler=batches, num_workers=workers,
                   pin_memory=pin_memory and device.type == "cuda",
                   generator=torch.Generator().manual_seed(seed))
    if workers:
        options.update(persistent_workers=persistent_workers, prefetch_factor=prefetch_factor,
                       multiprocessing_context="spawn")
    return DataLoader(dataset, **options)


def batch_to_device(batch: dict, device: torch.device) -> dict:
    return {key: batch_to_device(value, device) if isinstance(value, dict)
            else value.to(device, non_blocking=device.type == "cuda")
            for key, value in batch.items()}


def evaluation_loader(dataset, batch_size: int, device: torch.device, **options):
    """Ordered batches, including the final partial batch, with a separate RNG."""
    batches = [list(range(start, min(start + batch_size, len(dataset))))
               for start in range(0, len(dataset), batch_size)]
    return chunk_loader(dataset, batches, device=device, **options)


def close_loader(loader, iterator=None) -> None:
    """Release persistent workers, including when evaluation/training raises."""
    iterator = getattr(loader, "_iterator", None) if iterator is None else iterator
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if shutdown is not None:
        shutdown()
