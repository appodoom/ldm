"""Local training loop over prepared split arrays; no notebook-only model code."""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from .configuration_ldm import LDMConfig
from .evaluation_ldm import evaluate_model
from .modeling_ldm import LDMModel
from .rhythm_ldm import STRUCTURED_ROUTING_FIELDS


class ChunkDataset:
    def __init__(self, folder: str | Path, split: str):
        folder = Path(folder)
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


def train(dataset_dir: str | Path, run_dir: str | Path, config: LDMConfig, *,
          epochs: int, batch_size: int, learning_rate: float, seed: int = 1337,
          save_every_steps: int = 200, log_every_steps: int = 50, resume: bool = False,
          device: str | None = None, eval_batch_size: int | None = None,
          eval_every_steps: int | None = None, weight_decay: float | None = None,
          grad_clip: float | None = None, amp: bool = True,
          max_steps: int | None = None, max_train_time_minutes: float | None = None,
          stop_on_nonfinite_loss: bool = True, divergence_threshold: float | None = None,
          early_stopping: bool = False, early_stopping_patience: int = 5,
          early_stopping_min_delta: float = 0.001,
          early_stopping_warmup_evals: int = 0) -> dict:
    dataset_dir = Path(dataset_dir)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_bytes = (dataset_dir / "dataset_manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    import hashlib
    fingerprint = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest["variant"] != config.dataset_variant or manifest["vocab_hash"] != config.vocab_hash:
        raise ValueError("Dataset variant/vocabulary differs from model config")
    eval_batch_size = eval_batch_size or batch_size
    eval_every_steps = eval_every_steps or 0
    if min(epochs, batch_size, eval_batch_size, save_every_steps, log_every_steps) < 1 or learning_rate <= 0:
        raise ValueError("Training counts and learning rate must be positive")
    run_config = {"dataset_fingerprint": fingerprint, "model_config": config.to_dict(),
                  "batch_size": batch_size, "learning_rate": learning_rate,
                  "seed": seed, "save_every_steps": save_every_steps,
                  "decorated_weight": manifest.get("decorated_weight")}
    new_controls = {"eval_batch_size": eval_batch_size, "eval_every_steps": eval_every_steps,
                    "weight_decay": weight_decay, "grad_clip": grad_clip, "amp": amp,
                    "max_steps": max_steps, "max_train_time_minutes": max_train_time_minutes,
                    "stop_on_nonfinite_loss": stop_on_nonfinite_loss,
                    "divergence_threshold": divergence_threshold,
                    "early_stopping": early_stopping,
                    "early_stopping_patience": early_stopping_patience,
                    "early_stopping_min_delta": early_stopping_min_delta,
                    "early_stopping_warmup_evals": early_stopping_warmup_evals}
    run_config_path = run_dir / "run_config.json"
    if resume:
        # JSON turns integer dictionary keys (for example id2label) into strings.
        # Compare the saved file to the same serialized representation.
        saved_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        for key, value in saved_config.items():
            if key in run_config and value != json.loads(json.dumps(run_config[key])):
                raise ValueError(f"Resume setting differs: {key}")
        if "controls" in saved_config and saved_config["controls"] != new_controls:
            raise ValueError("Resume training controls differ from the original run")
    elif run_config_path.exists() or (run_dir / "last.pt").exists():
        raise ValueError("Run directory already contains a run; use --resume or another path")
    else:
        run_config["controls"] = new_controls
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=0.01 if weight_decay is None else weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and amp)
    epoch_start = 0
    batch_start = 0
    step = 0
    best = float("inf")
    history = []
    eval_events = 0
    stale_evals = 0
    best_step = 0
    stop_reason = "completed"
    started = time.perf_counter()
    checkpoint_path = run_dir / "last.pt"
    if resume:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint["dataset_fingerprint"] != fingerprint or checkpoint["config"] != config.to_dict():
            raise ValueError("Resume checkpoint has different dataset/config")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        # CPU runs disable scaling and save an empty state. A resumed CUDA run
        # may initialize a fresh scaler when the source checkpoint used CPU.
        if scaler.is_enabled() and checkpoint["scaler"]:
            scaler.load_state_dict(checkpoint["scaler"])
        _restore_rng(checkpoint["rng"])
        epoch_start, batch_start, step = checkpoint["epoch"], checkpoint["batch"], checkpoint["step"]
        best = checkpoint["best_val_loss"]
        history = checkpoint["history"]
        eval_events = checkpoint.get("eval_events", 0)
        stale_evals = checkpoint.get("stale_evals", 0)
        best_step = checkpoint.get("best_step")
    def save_last(epoch_number: int, next_batch: int) -> None:
        _save(checkpoint_path, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(), "rng": _rng_state(), "config": config.to_dict(),
            "dataset_fingerprint": fingerprint, "epoch": epoch_number, "batch": next_batch,
            "step": step, "best_val_loss": best, "history": history,
            "eval_events": eval_events, "stale_evals": stale_evals, "best_step": best_step})
    def validate(epoch_number: int) -> bool:
        nonlocal best, eval_events, stale_evals, best_step
        print(f"Validating at step {step}...", flush=True)
        report = evaluate_model(model, val_data, eval_batch_size, device)
        val_loss = report["combined"]["loss"]
        eval_events += 1
        history.append({"epoch": epoch_number, "step": step, "val_loss": val_loss,
                        "val_token_accuracy": report["combined"]["token_accuracy"]})
        with (run_dir / "evaluations.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"epoch": epoch_number, "step": step, "split": "val",
                                     "report": report}) + "\n")
        if val_loss < best - early_stopping_min_delta:
            best, best_step, stale_evals = val_loss, step, 0
            _save(run_dir / "best.pt", {"model": model.state_dict(), "config": config.to_dict(),
                "dataset_fingerprint": fingerprint, "epoch": epoch_number, "step": step,
                "val_loss": val_loss})
            print(f"Saved best checkpoint (validation loss {val_loss:.4f})", flush=True)
        elif eval_events > early_stopping_warmup_evals:
            stale_evals += 1
        print(f"Validation loss {val_loss:.4f} | accuracy {report['combined']['token_accuracy']:.4f} "
              f"| best {best:.4f}", flush=True)
        return bool(early_stopping and stale_evals >= early_stopping_patience)
    print(f"Training {config.dataset_variant} on {device}: "
          f"{len(train_data)} train, {len(val_data)} validation, "
          f"{len(test_data)} test chunks; target {epochs} epochs; "
          f"{sum(p.numel() for p in model.parameters()):,} parameters", flush=True)
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
        last_evaluation_step = -1
        last_batch = -1
        for batch_number in range(batch_start if epoch == epoch_start else 0, len(batches)):
            last_batch = batch_number
            model.train()
            batch = train_data.batch(batches[batch_number], device)
            target_count = int((batch["labels"] != -100).sum().item())
            if not target_count:
                continue
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda" and amp):
                loss = model(**batch).loss
            if not math.isfinite(float(loss.detach().item())):
                if stop_on_nonfinite_loss:
                    save_last(epoch, batch_number)
                    raise FloatingPointError(f"Nonfinite training loss at step {step}")
            scaler.scale(loss).backward()
            if grad_clip is not None and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            step += 1
            logged_loss += float(loss.detach().item()) * target_count
            logged_targets += target_count
            if step % log_every_steps == 0 or batch_number + 1 == len(batches):
                window_loss = logged_loss / logged_targets
                elapsed = max(time.perf_counter() - started, 1)
                memory_gib = (torch.cuda.max_memory_allocated(device) / 2**30
                              if device.type == "cuda" else None)
                print(f"Epoch {epoch + 1}/{epochs} batch {batch_number + 1}/{len(batches)} "
                      f"step {step} | train loss {window_loss:.4f} "
                      f"| elapsed {time.perf_counter() - epoch_started:.0f}s "
                      f"| {step / elapsed:.2f} steps/s" +
                      (f" | GPU {memory_gib:.2f} GiB"
                       if device.type == "cuda" else ""), flush=True)
                with (run_dir / "training_history.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"epoch": epoch + 1, "step": step,
                                             "loss": window_loss, "targets": logged_targets,
                                             "elapsed_seconds": elapsed,
                                             "gpu_memory_gib": memory_gib}) + "\n")
                logged_loss = 0.0
                logged_targets = 0
            if step % save_every_steps == 0:
                save_last(epoch, batch_number + 1)
                print(f"Saved resume checkpoint at step {step}", flush=True)
            if eval_every_steps and step % eval_every_steps == 0:
                should_stop = validate(epoch + 1)
                last_evaluation_step = step
                save_last(epoch, batch_number + 1)
                if should_stop:
                    stop_reason = "early_stopped"
                    break
            if divergence_threshold is not None and float(loss.detach().item()) > divergence_threshold:
                stop_reason = "diverged"
            elif max_steps is not None and step >= max_steps:
                stop_reason = "max_steps"
            elif max_train_time_minutes is not None and time.perf_counter() - started >= 60 * max_train_time_minutes:
                stop_reason = "max_train_time"
            if stop_reason != "completed":
                save_last(epoch, batch_number + 1)
                break
        if stop_reason != "completed":
            if last_evaluation_step != step:
                validate(epoch + 1)
            save_last(epoch, last_batch + 1)
            print(f"Stopped: {stop_reason} at epoch {epoch + 1}, step {step}", flush=True)
            break
        batch_start = 0
        if last_evaluation_step != step and validate(epoch + 1):
            stop_reason = "early_stopped"
        save_last(epoch + 1, 0)
        print(f"Epoch {epoch + 1}/{epochs} complete "
              f"| best {best:.4f} | elapsed {time.perf_counter() - epoch_started:.0f}s",
              flush=True)
        if stop_reason != "completed":
            break
    print("Evaluating best checkpoint on test split...", flush=True)
    best_checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best_checkpoint["model"], strict=True)
    test_report = evaluate_model(model, test_data, eval_batch_size, device)
    test_loss = test_report["combined"]["loss"]
    result = {"dataset_fingerprint": fingerprint, "best_val_loss": best,
              "best_step": best_step, "test_loss": test_loss, "steps": step,
              "stop_reason": stop_reason, "history": history, "test_report": test_report}
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Test loss {test_loss:.4f}; metrics saved to {run_dir / 'metrics.json'}", flush=True)
    return result
