"""Local training loop over prepared split arrays; no notebook-only model code."""

from __future__ import annotations

from copy import deepcopy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from .configuration_ldm import LDMConfig
from .evaluation_ldm import evaluate_model, format_percentage, print_evaluation_metrics
from .modeling_ldm import LDMModel
from .data_loading import ChunkDataset, chunk_loader, batch_to_device
from .dataset_integrity import verify_dataset
from .validation import validate_rules


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
          grad_clip: float | None = None, amp: bool = True, allow_tf32: bool = True,
          num_workers: int = 2, pin_memory: bool = True,
          persistent_workers: bool = True, prefetch_factor: int = 2,
          max_steps: int | None = None, max_train_time_minutes: float | None = None,
          stop_on_nonfinite_loss: bool = True, divergence_threshold: float | None = None,
          early_stopping: bool = False, early_stopping_patience: int = 5,
          early_stopping_min_delta: float = 0.001,
          early_stopping_warmup_evals: int = 0) -> dict:
    # Transformers updates runtime fields on config during model construction.
    # Keep the caller's experiment settings unchanged across independent runs.
    config = deepcopy(config)
    dataset_dir = Path(dataset_dir)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_bytes = (dataset_dir / "dataset_manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    fingerprint = verify_dataset(dataset_dir, model_config=config.to_dict())
    eval_batch_size = batch_size if eval_batch_size is None else eval_batch_size
    eval_every_steps = 0 if eval_every_steps is None else eval_every_steps
    validate_rules({
        **{name: (value >= 1, "must be positive") for name, value in {
            "epochs": epochs, "batch_size": batch_size, "eval_batch_size": eval_batch_size,
            "save_every_steps": save_every_steps, "log_every_steps": log_every_steps,
            "prefetch_factor": prefetch_factor}.items()},
        "num_workers": (num_workers >= 0, "must be nonnegative"),
        "eval_every_steps": (eval_every_steps >= 0, "must be nonnegative"),
        "learning_rate": (math.isfinite(learning_rate) and learning_rate > 0, "must be positive and finite"),
    })
    run_config = {"dataset_fingerprint": fingerprint, "model_config": config.to_dict(),
                  "batch_size": batch_size, "learning_rate": learning_rate,
                  "seed": seed, "save_every_steps": save_every_steps,
                  "decoration": manifest["decoration"]}
    new_controls = {"eval_batch_size": eval_batch_size, "eval_every_steps": eval_every_steps,
                    "weight_decay": weight_decay, "grad_clip": grad_clip, "amp": amp, "allow_tf32": allow_tf32,
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
        rules = {key: (value == json.loads(json.dumps(run_config[key])), f"Resume setting differs: {key}")
                 for key, value in saved_config.items() if key in run_config}
        rules["controls"] = ("controls" not in saved_config or
            {"allow_tf32": True, **saved_config["controls"]} == new_controls,
            "Resume training controls differ from the original run")
        validate_rules(rules)
    elif run_config_path.exists() or (run_dir / "last.pt").exists():
        raise ValueError("Run directory already contains a run; use --resume or another path")
    else:
        run_config["controls"] = new_controls
        run_config_path.write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
        dataset_ref = {key: manifest[key] for key in
                       ("preparation_version", "normalization_rule", "split_rule", "variant", "vocab_hash",
                        "source_hash", "decoration", "context_size", "stride",
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
    from .model_core import configure_global_flash, configure_cuda_precision
    precision = configure_cuda_precision(device, allow_tf32=allow_tf32)
    attention_backend = configure_global_flash(model.core, amp=amp)
    (run_dir / "attention_backend.json").write_text(
        json.dumps(attention_backend, indent=2), encoding="utf-8")
    runtime = {"device": str(device), **precision, "amp": device.type == "cuda" and amp,
               "grad_scaler": device.type == "cuda" and amp,
               "num_workers": num_workers if device.type == "cuda" else 0,
               "pin_memory": pin_memory and device.type == "cuda",
               "persistent_workers": persistent_workers and num_workers > 0 and device.type == "cuda",
               "prefetch_factor": prefetch_factor if num_workers > 0 and device.type == "cuda" else None}
    (run_dir / "runtime_settings.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    print(f"[runtime] {json.dumps(runtime)}", flush=True)
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
        if epochs < epoch_start + int(batch_start > 0):
            raise ValueError("Target epochs cannot be below the checkpoint's completed or active epoch")
        best = checkpoint["best_val_loss"]
        history = checkpoint["history"]
        eval_events = checkpoint.get("eval_events", 0)
        stale_evals = checkpoint.get("stale_evals", 0)
        best_step = checkpoint.get("best_step")
    session_start_step = step
    from .model_checks import validate_core
    print('Checking attention caches, causality, padding, and disabled streams...', flush=True)
    checks = validate_core(model.core, list(model.vocab), config.dataset_variant)
    checks.update(device=str(device), torch_version=torch.__version__)
    (run_dir / 'model_checks.json').write_text(json.dumps(checks, indent=2) + '\n', encoding='utf-8')
    print('Model correctness checks passed', flush=True)
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
        print_evaluation_metrics(report, "Validation")
        print(f"Best validation loss {best:.4f}", flush=True)
        return bool(early_stopping and stale_evals >= early_stopping_patience)
    print(f"Training {config.dataset_variant} on {device}: "
          f"{len(train_data)} train, {len(val_data)} validation, "
          f"{len(test_data)} test chunks; target {epochs} epochs; "
          f"{sum(p.numel() for p in model.parameters()):,} parameters", flush=True)
    if resume:
        print(f"Resuming at epoch {epoch_start + 1}, batch {batch_start + 1}, "
              f"global step {step}", flush=True)
    pending_batches = []
    loader = chunk_loader(train_data, pending_batches, device=device, num_workers=num_workers,
                         pin_memory=pin_memory, persistent_workers=persistent_workers,
                         prefetch_factor=prefetch_factor, seed=seed)
    if max_steps is not None and step >= max_steps:
        stop_reason = "max_steps"
        print(f"Maximum steps already reached ({step}); skipping updates", flush=True)
    batch_iterator = None
    try:
        for epoch in range(epoch_start, epochs) if stop_reason == "completed" else ():
            epoch_started = time.perf_counter()
            generator = torch.Generator().manual_seed(seed + epoch)
            order = torch.randperm(len(train_data), generator=generator).tolist()
            batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
            print(f"Epoch {epoch + 1}/{epochs}: {len(batches)} batches", flush=True)
            logged_loss = 0.0
            logged_targets = 0
            last_evaluation_step = -1
            last_batch = -1
            first_batch = batch_start if epoch == epoch_start else 0
            pending_batches[:] = batches[first_batch:]
            batch_iterator = iter(loader)
            for batch_number, cpu_batch in enumerate(batch_iterator, start=first_batch):
                last_batch = batch_number
                model.train()
                batch = batch_to_device(cpu_batch, device)
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
                          f"| {(step - session_start_step) / elapsed:.2f} steps/s" +
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
    finally:
        # DataLoader has no public close method. Explicit shutdown also covers
        # exceptions whose traceback still holds the active worker iterator.
        shutdown = getattr(batch_iterator, "_shutdown_workers", None)
        if shutdown is not None:
            shutdown()
        del loader
    print("Evaluating best checkpoint on test split...", flush=True)
    best_checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best_checkpoint["model"], strict=True)
    test_report = evaluate_model(model, test_data, eval_batch_size, device)
    print_evaluation_metrics(test_report, "Test")
    test_loss = test_report["combined"]["loss"]
    result = {"dataset_fingerprint": fingerprint, "best_val_loss": best,
              "best_step": best_step, "test_loss": test_loss, "steps": step,
              "stop_reason": stop_reason, "history": history, "test_report": test_report}
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Test loss {test_loss:.4f} | Test accuracy "
          f"{format_percentage(test_report['combined']['token_accuracy'])}; "
          f"metrics saved to {run_dir / 'metrics.json'}", flush=True)
    return result
