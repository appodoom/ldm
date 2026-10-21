"""Dataset cache and explicit lifecycle for named LDM experiments."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from contextlib import redirect_stderr, redirect_stdout
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .experiment_config import RunConfig
from .constants import PREPARATION_VERSION
from .validation import validate_rules
from .dataset_integrity import dataset_fingerprint, verify_dataset


def _source_hash(source_dir: Path) -> str:
    files = sorted(source_dir.glob("*.derbake"))
    if not files:
        raise ValueError(f"No .derbake files in {source_dir}")
    digest = hashlib.sha256()
    for source in files:
        digest.update(source.name.encode())
        digest.update(hashlib.sha256(source.read_bytes()).digest())
    return digest.hexdigest()


def prepare_or_load_dataset(config: RunConfig, source_dir: str | Path,
                            cache_root: str | Path) -> Path:
    """Prepare/reuse exactly one source for the selected subdivision/decoration axes."""
    from .dataset_ldm import prepare_dataset
    from .vocabulary import atomic_vocabulary, vocabulary_hash

    source_dir = Path(source_dir)
    recipe = {"variant": config.data.variant.value, "decoration": config.data.decoration.value,
              "context_size": config.data.context_size, "stride": config.data.resolved_stride,
              "max_beats_per_cycle": config.data.max_beats_per_cycle,
              "source_hash": _source_hash(source_dir),
              "vocab_hash": vocabulary_hash(atomic_vocabulary(config.data.variant.value)),
              "preparation_version": PREPARATION_VERSION}
    fingerprint = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    target = Path(cache_root) / f"{config.data.variant.value}-{config.data.decoration.value}-{fingerprint[:12]}"
    manifest_path = target / "dataset_manifest.json"
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        differences = [key for key, value in recipe.items() if saved.get(key) != value]
        if differences:
            raise ValueError(f"Cached dataset recipe differs: {', '.join(differences)}")
        verify_dataset(target)
        print(f"Using prepared dataset {target}", flush=True)
        return target
    if target.exists() and any(target.iterdir()):
        raise ValueError(f"Incomplete dataset cache at {target}; inspect it before retrying")
    print(f"Preparing dataset {target}", flush=True)
    prepare_dataset(source_dir, target, variant=config.data.variant.value,
                    decoration=config.data.decoration.value, context_size=config.data.context_size,
                    max_beats_per_cycle=config.data.max_beats_per_cycle, stride=config.data.resolved_stride)
    return target


def model_config_for_run(run: RunConfig, dataset: str | Path):
    from .configuration_ldm import LDMConfig
    from .vocabulary import atomic_vocabulary, vocabulary_hash

    manifest = json.loads((Path(dataset) / "dataset_manifest.json").read_text(encoding="utf-8"))
    expected = {"variant": run.data.variant.value,
                "context_size": run.data.context_size,
                "stride": run.data.resolved_stride,
                "max_beats_per_cycle": run.data.max_beats_per_cycle,
                "decoration": run.data.decoration.value}
    mismatches = [key for key, value in expected.items() if manifest.get(key) != value]
    if mismatches:
        raise ValueError(f"Prepared dataset recipe differs from run: {mismatches}")
    vocab = atomic_vocabulary(run.data.variant.value)
    maxima = {key: max(int(value[key]) for value in manifest["stats"].values())
              for key in ("max_tokens_per_beat", "max_beat_slots_per_chunk", "max_cycles_per_chunk")}
    m = run.model
    return LDMConfig(vocab_size=len(vocab), vocab_hash=vocabulary_hash(vocab),
                     dataset_variant=run.data.variant.value, dataset_decoration=run.data.decoration.value,
                     context_size=run.data.context_size,
                     max_beats_per_cycle=run.data.max_beats_per_cycle,
                     number_of_layers=m.layers, number_of_heads=m.heads,
                     number_of_embeddings=m.embeddings, mlp_ratio=m.mlp_ratio,
                     dropout=m.dropout, attention_layout=m.attention.value,
                     activation_name=m.activation.value, position_encoding=m.position.value,
                     rope_theta=m.rope_theta, rope_fraction=m.rope_fraction,
                     tie_weights=m.tie_weights, use_type_embeddings=m.use_type_embeddings,
                     concat_type_embeddings=m.concat_type_embeddings,
                     use_inside_beat_stream=m.streams.inside_beat,
                     use_beat_memory_stream=m.streams.beat_memory,
                     use_cycle_memory_stream=m.streams.cycle_memory, **maxima)


def make_run_id(config: RunConfig, now: datetime | None = None) -> str:
    if config.name is not None:
        return config.name
    now = now or datetime.now(timezone.utc)
    m, d = config.model, config.data
    return (f"run_{now:%Y%m%d-%H%M%S}_{d.variant.value}_{d.decoration.value}_ctx{d.context_size}_"
            f"s{d.resolved_stride}_L{m.layers}H{m.heads}E{m.embeddings}_{config.fingerprint[:8]}")


def check_run_name(config: RunConfig, runs_root: str | Path,
                   registry: Iterable[RunConfig] | None = None) -> str:
    """Fail early for a reused custom name; automatic names remain unique attempts."""
    if config.name is None:
        return make_run_id(config)
    if registry is not None:
        matches = [index for index, value in enumerate(registry) if value.name == config.name]
        if len(matches) > 1:
            raise ValueError(f"Custom run name {config.name!r} is used by multiple presets "
                             f"at positions {matches}")
    path = Path(runs_root) / config.name
    if path.exists():
        raise FileExistsError(f"Run name {config.name!r} already exists at {path}; "
                              "choose another name or use Resume")
    return config.name


def _code_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=Path(__file__).resolve().parent.parent,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _train_kwargs(config: RunConfig) -> dict:
    t, e = config.train, config.evaluation
    return dict(epochs=t.epochs, batch_size=t.batch_size, learning_rate=t.learning_rate,
                seed=t.seed, save_every_steps=t.save_every_steps,
                log_every_steps=t.log_every_steps, eval_batch_size=e.batch_size,
                eval_every_steps=e.every_steps, weight_decay=t.weight_decay,
                grad_clip=t.grad_clip, amp=t.amp, allow_tf32=t.allow_tf32,
                num_workers=t.num_workers, pin_memory=t.pin_memory,
                persistent_workers=t.persistent_workers, prefetch_factor=t.prefetch_factor,
                max_steps=t.max_steps,
                max_train_time_minutes=t.max_train_time_minutes,
                stop_on_nonfinite_loss=t.stop_on_nonfinite_loss,
                divergence_threshold=t.divergence_threshold,
                early_stopping=t.early_stopping,
                early_stopping_patience=t.early_stopping_patience,
                early_stopping_min_delta=t.early_stopping_min_delta,
                early_stopping_warmup_evals=t.early_stopping_warmup_evals)


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def start_run(config: RunConfig, dataset: str | Path, runs_root: str | Path,
              *, device: str | None = None) -> Path:

    runs_root = Path(runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)
    run_name = check_run_name(config, runs_root)
    suffix = 1
    while True:
        run_dir = runs_root / (run_name if suffix == 1 else f"{run_name}_{suffix}")
        try:
            run_dir.mkdir()
            break
        except FileExistsError as exc:
            if config.name is not None:
                raise FileExistsError(f"Run name {run_dir.name!r} already exists; "
                                      "choose another name or use Resume") from exc
            suffix += 1
    dataset = Path(dataset).resolve()
    record = {"run_id": run_dir.name, "status": "created", "config": config.to_dict(),
              "config_fingerprint": config.fingerprint, "dataset_path": str(dataset),
              "dataset_fingerprint": dataset_fingerprint(dataset),
              "code_commit": _code_commit(), "python": platform.python_version(),
              "created_utc": datetime.now(timezone.utc).isoformat()}
    _write_json(run_dir / "experiment.json", record)
    with (run_dir / "training.log").open("a", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(_Tee(sys.stdout, log)), redirect_stderr(_Tee(sys.stderr, log)):
            print(f"New run: {run_dir}", flush=True)
            execute_run(run_dir, device=device)
    return run_dir


def execute_run(run_dir: str | Path, *, device: str | None = None,
                resume: bool = False, target_epochs: int | None = None) -> dict:
    from .train_ldm import train

    run_dir = Path(run_dir)
    path = run_dir / "experiment.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    config = RunConfig.from_dict(record["config"])
    dataset = Path(record["dataset_path"])
    kwargs = _train_kwargs(config)
    has_checkpoint = (run_dir / "last.pt").exists()
    validate_rules({
        "dataset": (dataset_fingerprint(dataset) == record["dataset_fingerprint"],
                    "Prepared dataset fingerprint differs from run record"),
        "resume": (not resume or has_checkpoint, "No resumable last.pt in this run"),
        "new_run": (resume or not has_checkpoint, "This run already has a checkpoint; use resume_run"),
        "target_epochs": (target_epochs is None or target_epochs >= kwargs["epochs"],
                          "Resume target epochs cannot be below the saved target"),
    })
    if target_epochs is not None:
        kwargs["epochs"] = target_epochs
    record["status"] = "training"
    _write_json(path, record)
    try:
        result = train(dataset, run_dir, model_config_for_run(config, dataset),
                       **kwargs, device=device, resume=resume)
    except BaseException as exc:
        record["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        record["last_error"] = repr(exc)
        _write_json(path, record)
        raise
    record.pop("last_error", None)
    record["status"] = result["stop_reason"]
    record["last_step"] = result["steps"]
    record["best_step"] = result["best_step"]
    record["best_val_loss"] = result["best_val_loss"]
    record["updated_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(path, record)
    return result


def resume_run(run_dir: str | Path, *, target_epochs: int | None = None,
               device: str | None = None) -> dict:
    run_dir = Path(run_dir)
    with (run_dir / "training.log").open("a", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(_Tee(sys.stdout, log)), redirect_stderr(_Tee(sys.stderr, log)):
            return execute_run(run_dir, resume=True, target_epochs=target_epochs,
                               device=device)


def inspect_run(run_dir: str | Path, *, include_metrics: bool = True) -> dict:
    run_dir = Path(run_dir)
    record = json.loads((run_dir / "experiment.json").read_text(encoding="utf-8"))
    for name in ("last.pt", "best.pt", "metrics.json", "training.log"):
        path = run_dir / name
        record[name] = ({"bytes": path.stat().st_size,
                         "modified_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()}
                        if path.exists() else None)
    if include_metrics and (run_dir / "metrics.json").exists():
        record["metrics"] = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    return record


def list_runs(runs_root: str | Path) -> list[dict]:
    return [inspect_run(folder, include_metrics=False) for folder in sorted(Path(runs_root).iterdir())
            if (folder / "experiment.json").exists()] if Path(runs_root).exists() else []


def compare_runs(run_dirs: list[str | Path]) -> list[dict]:
    rows = []
    for path in run_dirs:
        record = inspect_run(path)
        metrics = record.get("metrics", {})
        reference = Path(path) / "dataset_ref.json"
        split_rule = json.loads(reference.read_text(encoding="utf-8"))["split_rule"]
        rows.append({"run_id": record["run_id"], "status": record["status"],
                     "config_hash": record["config_fingerprint"],
                     "dataset_fingerprint": record["dataset_fingerprint"],
                     "split_rule": split_rule,
                     "variant": record["config"]["data"]["variant"],
                     "decoration": record["config"]["data"]["decoration"],
                     "best_step": metrics.get("best_step"),
                     "val_loss": metrics.get("best_val_loss"),
                     "test_loss": metrics.get("test_loss"),
                     "test_token_accuracy": metrics.get("test_report", {}).get("combined", {}).get("token_accuracy"),
                     "metric_version": metrics.get("test_report", {}).get("metric_version")})
    fingerprints = {row["dataset_fingerprint"] for row in rows}
    versions = {row["metric_version"] for row in rows}
    splits = {row["split_rule"] for row in rows}
    for row in rows:
        row["comparable"] = (len(fingerprints) == len(versions) == len(splits) == 1
                             and row['metric_version'] is not None)
    return rows
