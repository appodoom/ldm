"""Local training loop over prepared split arrays; no notebook-only model code."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .configuration_ldm import LDMConfig
from .modeling_ldm import LDMModel
from .rhythm_ldm import STRUCTURED_ROUTING_FIELDS


class ChunkDataset:
    def __init__(self, folder: str | Path, split: str):
        folder = Path(folder)
        self.ids = np.load(folder / f"{split}.npy", mmap_mode="r")
        self.source_kind = np.load(folder / f"{split}_source_kind.npy", mmap_mode="r")
        self.beat_idx = np.load(folder / f"{split}_beat_idx.npy", mmap_mode="r")
        self.beat_pos = np.load(folder / f"{split}_beat_pos.npy", mmap_mode="r")
        self.score = np.load(folder / f"{split}_score_mask.npy", mmap_mode="r")
        self.routing = {field: np.load(folder / f"{split}_{field}.npy", mmap_mode="r")
                        for field in STRUCTURED_ROUTING_FIELDS}
        if not all(len(array) == len(self.ids) for array in
                   (self.source_kind, self.beat_idx, self.beat_pos, self.score, *self.routing.values())):
            raise ValueError("Prepared array row counts disagree")

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


def _rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _save(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


@torch.no_grad()
def evaluate(model: LDMModel, dataset: ChunkDataset, batch_size: int, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_targets = 0
    for start in range(0, len(dataset), batch_size):
        batch = dataset.batch(range(start, min(start + batch_size, len(dataset))), device)
        count = int((batch["labels"] != -100).sum().item())
        if not count:
            continue
        logits = model(**{key: value for key, value in batch.items() if key != "labels"}).logits
        total_loss += float(F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                            batch["labels"].reshape(-1), ignore_index=-100,
                                            reduction="sum").item())
        total_targets += count
    if not total_targets:
        raise ValueError("Evaluation split has no scored targets")
    return total_loss / total_targets


def train(dataset_dir: str | Path, run_dir: str | Path, config: LDMConfig, *,
          epochs: int, batch_size: int, learning_rate: float, seed: int = 1337,
          save_every_steps: int = 200, log_every_steps: int = 50, resume: bool = False,
          device: str | None = None) -> dict:
    dataset_dir = Path(dataset_dir)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_bytes = (dataset_dir / "dataset_manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    import hashlib
    fingerprint = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest["variant"] != config.dataset_variant or manifest["vocab_hash"] != config.vocab_hash:
        raise ValueError("Dataset variant/vocabulary differs from model config")
    if min(epochs, batch_size, save_every_steps, log_every_steps) < 1 or learning_rate <= 0:
        raise ValueError("Training counts and learning rate must be positive")
    run_config = {"dataset_fingerprint": fingerprint, "model_config": config.to_dict(),
                  "batch_size": batch_size, "learning_rate": learning_rate,
                  "seed": seed, "save_every_steps": save_every_steps,
                  "decorated_weight": manifest.get("decorated_weight")}
    run_config_path = run_dir / "run_config.json"
    if resume:
        # JSON turns integer dictionary keys (for example id2label) into strings.
        # Compare the saved file to the same serialized representation.
        saved_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        if saved_config != json.loads(json.dumps(run_config)):
            raise ValueError("Resume settings differ from the original run")
    elif run_config_path.exists() or (run_dir / "last.pt").exists():
        raise ValueError("Run directory already contains a run; use --resume or another path")
    else:
        run_config_path.write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
        dataset_ref = {key: manifest[key] for key in
                       ("format_version", "preparation_version", "normalization_version",
                        "split_version", "variant", "vocab_hash",
                        "source_hash", "decorated_weight", "context_size", "stride",
                        "max_beats_per_cycle")}
        (run_dir / "dataset_ref.json").write_text(json.dumps(dataset_ref, indent=2) + "\n",
                                                  encoding="utf-8")
    train_data = ChunkDataset(dataset_dir, "train")
    val_data = ChunkDataset(dataset_dir, "val")
    test_data = ChunkDataset(dataset_dir, "test")
    if min(len(train_data), len(val_data), len(test_data)) == 0:
        raise ValueError("All three splits need training chunks")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = LDMModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    epoch_start = 0
    batch_start = 0
    step = 0
    best = float("inf")
    history = []
    checkpoint_path = run_dir / "last.pt"
    if resume:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint["dataset_fingerprint"] != fingerprint or checkpoint["config"] != config.to_dict():
            raise ValueError("Resume checkpoint has different dataset/config")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        _restore_rng(checkpoint["rng"])
        epoch_start, batch_start, step = checkpoint["epoch"], checkpoint["batch"], checkpoint["step"]
        best = checkpoint["best_val_loss"]
        history = checkpoint["history"]
    print(f"Training {config.dataset_variant} on {device}: "
          f"{len(train_data)} train, {len(val_data)} validation, "
          f"{len(test_data)} test chunks; target {epochs} epochs", flush=True)
    if resume:
        print(f"Resuming at epoch {epoch_start + 1}, batch {batch_start + 1}, "
              f"global step {step}", flush=True)
    for epoch in range(epoch_start, epochs):
        epoch_started = time.perf_counter()
        generator = torch.Generator().manual_seed(seed + epoch)
        decorated_weight = manifest.get("decorated_weight") or 1.0
        if decorated_weight == 1.0:
            order = torch.randperm(len(train_data), generator=generator).tolist()
        else:
            weights = torch.as_tensor(np.where(np.asarray(train_data.source_kind) == 1,
                                               decorated_weight, 1.0), dtype=torch.double)
            order = torch.multinomial(weights, len(train_data), replacement=True,
                                      generator=generator).tolist()
        batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
        print(f"Epoch {epoch + 1}/{epochs}: {len(batches)} batches", flush=True)
        logged_loss = 0.0
        logged_targets = 0
        for batch_number in range(batch_start if epoch == epoch_start else 0, len(batches)):
            model.train()
            batch = train_data.batch(batches[batch_number], device)
            target_count = int((batch["labels"] != -100).sum().item())
            if not target_count:
                continue
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            step += 1
            logged_loss += float(loss.detach().item()) * target_count
            logged_targets += target_count
            if step % log_every_steps == 0 or batch_number + 1 == len(batches):
                print(f"Epoch {epoch + 1}/{epochs} batch {batch_number + 1}/{len(batches)} "
                      f"step {step} | train loss {logged_loss / logged_targets:.4f} "
                      f"| elapsed {time.perf_counter() - epoch_started:.0f}s", flush=True)
                logged_loss = 0.0
                logged_targets = 0
            if step % save_every_steps == 0:
                _save(checkpoint_path, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(), "rng": _rng_state(), "config": config.to_dict(),
                    "dataset_fingerprint": fingerprint, "epoch": epoch, "batch": batch_number + 1,
                    "step": step, "best_val_loss": best, "history": history})
                print(f"Saved resume checkpoint at step {step}", flush=True)
        batch_start = 0
        print(f"Epoch {epoch + 1}/{epochs}: validating...", flush=True)
        val_loss = evaluate(model, val_data, batch_size, device)
        history.append({"epoch": epoch + 1, "val_loss": val_loss})
        if val_loss < best:
            best = val_loss
            _save(run_dir / "best.pt", {"model": model.state_dict(), "config": config.to_dict(),
                "dataset_fingerprint": fingerprint, "epoch": epoch + 1, "val_loss": val_loss})
            print(f"Saved best checkpoint (validation loss {val_loss:.4f})", flush=True)
        _save(checkpoint_path, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(), "rng": _rng_state(), "config": config.to_dict(),
            "dataset_fingerprint": fingerprint, "epoch": epoch + 1, "batch": 0,
            "step": step, "best_val_loss": best, "history": history})
        print(f"Epoch {epoch + 1}/{epochs} complete | val loss {val_loss:.4f} "
              f"| best {best:.4f} | elapsed {time.perf_counter() - epoch_started:.0f}s",
              flush=True)
    print("Evaluating best checkpoint on test split...", flush=True)
    best_checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best_checkpoint["model"], strict=True)
    test_loss = evaluate(model, test_data, batch_size, device)
    result = {"dataset_fingerprint": fingerprint, "best_val_loss": best,
              "test_loss": test_loss, "steps": step, "history": history}
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Test loss {test_loss:.4f}; metrics saved to {run_dir / 'metrics.json'}", flush=True)
    return result
