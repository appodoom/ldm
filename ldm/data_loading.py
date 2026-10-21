"""Memory-mapped prepared data and CPU batch preparation for GPU training."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .rhythm_ldm import STRUCTURED_ROUTING_FIELDS


class ChunkDataset:
    def __init__(self, folder: str | Path, split: str):
        folder = Path(folder)
        self.folder, self.split = folder, split
        self.ids = np.load(folder / f"{split}.npy", mmap_mode="r")
        self.source_kind = np.load(folder / f"{split}_source_kind.npy", mmap_mode="r")
        file_path = folder / f"{split}_source_file.npy"
        self.source_file = np.load(file_path, mmap_mode="r") if file_path.exists() else None
        self.file_names = [entry["file"] for entry in json.loads(
            (folder / "dataset_manifest.json").read_text(encoding="utf-8"))["files"]]
        self.beat_idx = np.load(folder / f"{split}_beat_idx.npy", mmap_mode="r")
        self.beat_pos = np.load(folder / f"{split}_beat_pos.npy", mmap_mode="r")
        self.score = np.load(folder / f"{split}_score_mask.npy", mmap_mode="r")
        self.hit_count = np.load(folder / f"{split}_hit_count_mask.npy", mmap_mode="r")
        skeleton_path = folder / f"{split}_skeleton_mask.npy"
        self.skeleton = np.load(skeleton_path, mmap_mode="r") if skeleton_path.exists() else None
        self.routing = {field: np.load(folder / f"{split}_{field}.npy", mmap_mode="r")
                        for field in STRUCTURED_ROUTING_FIELDS}
        if not all(len(array) == len(self.ids) for array in
                   (self.source_kind, self.beat_idx, self.beat_pos, self.score,
                    self.hit_count, *self.routing.values())):
            raise ValueError("Prepared array row counts disagree")
        if self.skeleton is not None and len(self.skeleton) != len(self.ids):
            raise ValueError("Skeleton mask row count disagrees")
        if self.source_file is not None and len(self.source_file) != len(self.ids):
            raise ValueError("Source-file row count disagrees")

    def __getstate__(self):
        # Reopen memory maps in spawned workers instead of pickling array contents.
        return {"folder": self.folder, "split": self.split}

    def __setstate__(self, state):
        self.__init__(state["folder"], state["split"])

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
        return {"input_ids": tokens[:, :-1], "labels": labels,
                "beat_idx_ids": torch.as_tensor(np.asarray(self.beat_idx[rows, :-1]),
                                                dtype=torch.long, device=device),
                "beat_pos_ids": torch.as_tensor(np.asarray(self.beat_pos[rows, :-1]),
                                                dtype=torch.long, device=device),
                "structured_routing": routing}


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
