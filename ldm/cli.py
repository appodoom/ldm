"""Local preparation, training, snapshot, and validation commands."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def _prepare(args):
    from .dataset_ldm import prepare_dataset
    result = prepare_dataset(args.ordinary, args.output, variant=args.variant,
                             context_size=args.context_size,
                             max_beats_per_cycle=args.max_beats_per_cycle,
                             decorated_dir=args.decorated, stride=args.stride,
                             decorated_weight=args.decorated_weight)
    print(json.dumps(result["stats"], indent=2))


def _train(args):
    from .configuration_ldm import LDMConfig
    from .train_ldm import train
    from .vocabulary import atomic_vocabulary, vocabulary_hash
    manifest = json.loads((args.dataset / "dataset_manifest.json").read_text(encoding="utf-8"))
    vocab = atomic_vocabulary(manifest["variant"])
    maxima = {key: max(int(value[key]) for value in manifest["stats"].values())
              for key in ("max_tokens_per_beat", "max_beat_slots_per_chunk", "max_cycles_per_chunk")}
    config = LDMConfig(vocab_size=len(vocab), vocab_hash=vocabulary_hash(vocab),
                       dataset_variant=manifest["variant"], context_size=manifest["context_size"],
                       max_beats_per_cycle=manifest["max_beats_per_cycle"],
                       number_of_layers=args.layers, number_of_heads=args.heads,
                       number_of_embeddings=args.embeddings, mlp_ratio=args.mlp_ratio,
                       dropout=args.dropout, attention_layout=args.attention_layout,
                       **maxima)
    result = train(args.dataset, args.run, config, epochs=args.epochs,
                   batch_size=args.batch_size, learning_rate=args.learning_rate,
                   seed=args.seed, save_every_steps=args.save_every_steps,
                   log_every_steps=args.log_every_steps, amp=args.amp, allow_tf32=args.allow_tf32,
                   num_workers=args.num_workers, pin_memory=args.pin_memory,
                   persistent_workers=args.persistent_workers, prefetch_factor=args.prefetch_factor,
                   resume=args.resume, device=args.device)
    print(json.dumps(result, indent=2))


def _snapshot(args):
    from .hub_ldm import build_local_snapshot
    print(json.dumps(build_local_snapshot(args.run, args.output), indent=2))


def _validate(args):
    script = Path(__file__).resolve().parent.parent / "tests" / "hf_snapshot_smoke.py"
    with tempfile.TemporaryDirectory() as folder:
        command = [sys.executable, "-I", str(script), str(args.snapshot.resolve())]
        subprocess.run(command, check=True, cwd=folder)


def _experiment(args):
    from .experiment_config import RunConfig
    from .experiment_workflow import (compare_runs, inspect_run, list_runs,
                                      prepare_or_load_dataset, resume_run, start_run)

    if args.action == "list":
        print(json.dumps(list_runs(args.runs_root), indent=2))
    elif args.action == "inspect":
        print(json.dumps(inspect_run(args.run), indent=2))
    elif args.action == "compare":
        print(json.dumps(compare_runs(args.runs), indent=2))
    elif args.action == "start":
        config = RunConfig.from_dict(json.loads(args.config.read_text(encoding="utf-8")))
        dataset = prepare_or_load_dataset(config, args.ordinary, args.decorated, args.cache_root)
        run = start_run(config, dataset, args.runs_root, device=args.device)
        print(f"Run: {run}")
    elif args.action == "resume":
        print(json.dumps(resume_run(args.run, target_epochs=args.target_epochs,
                                    device=args.device), indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m ldm.cli")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--ordinary", type=Path, required=True)
    prepare.add_argument("--decorated", type=Path)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--variant", choices=("with_subd", "no_subd"), required=True)
    prepare.add_argument("--context-size", type=int, default=1024)
    prepare.add_argument("--max-beats-per-cycle", type=int, default=16)
    prepare.add_argument("--stride", type=int)
    prepare.add_argument("--decorated-weight", type=float, default=1.0)
    prepare.set_defaults(handler=_prepare)
    training = commands.add_parser("train")
    training.add_argument("--dataset", type=Path, required=True)
    training.add_argument("--run", type=Path, required=True)
    training.add_argument("--epochs", type=int, default=10)
    training.add_argument("--batch-size", type=int, default=4)
    training.add_argument("--learning-rate", type=float, default=3e-4)
    training.add_argument("--layers", type=int, default=4)
    training.add_argument("--heads", type=int, default=4)
    training.add_argument("--embeddings", type=int, default=256)
    training.add_argument("--mlp-ratio", type=int, default=4)
    training.add_argument("--dropout", type=float, default=0.1)
    training.add_argument("--attention-layout", choices=("global", "alternating_global_structured"),
                          default="alternating_global_structured")
    training.add_argument("--seed", type=int, default=1337)
    training.add_argument("--save-every-steps", type=int, default=200)
    training.add_argument("--log-every-steps", type=int, default=50)
    training.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    training.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    training.add_argument("--num-workers", type=int, default=2)
    training.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    training.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    training.add_argument("--prefetch-factor", type=int, default=2)
    training.add_argument("--device")
    training.add_argument("--resume", action="store_true")
    training.set_defaults(handler=_train)
    snapshot = commands.add_parser("snapshot")
    snapshot.add_argument("--run", type=Path, required=True)
    snapshot.add_argument("--output", type=Path, required=True)
    snapshot.set_defaults(handler=_snapshot)
    validation = commands.add_parser("validate")
    validation.add_argument("--snapshot", type=Path, required=True)
    validation.set_defaults(handler=_validate)
    experiment = commands.add_parser("experiment")
    actions = experiment.add_subparsers(dest="action", required=True)
    start = actions.add_parser("start")
    start.add_argument("--config", type=Path, required=True)
    start.add_argument("--ordinary", type=Path, required=True)
    start.add_argument("--decorated", type=Path)
    start.add_argument("--cache-root", type=Path, required=True)
    start.add_argument("--runs-root", type=Path, required=True)
    start.add_argument("--device")
    start.set_defaults(handler=_experiment)
    resume = actions.add_parser("resume")
    resume.add_argument("--run", type=Path, required=True)
    resume.add_argument("--target-epochs", type=int)
    resume.add_argument("--device")
    resume.set_defaults(handler=_experiment)
    inspect = actions.add_parser("inspect")
    inspect.add_argument("--run", type=Path, required=True)
    inspect.set_defaults(handler=_experiment)
    listed = actions.add_parser("list")
    listed.add_argument("--runs-root", type=Path, required=True)
    listed.set_defaults(handler=_experiment)
    compare = actions.add_parser("compare")
    compare.add_argument("runs", nargs="+", type=Path)
    compare.set_defaults(handler=_experiment)
    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
