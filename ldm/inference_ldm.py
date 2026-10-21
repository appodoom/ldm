"""Rhythm inference and optional embedding inspection."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .validation import validate_rules


@dataclass(frozen=True)
class InferenceConfig:
    num_cycles: int = 1
    seed: int = 1337
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    max_new_tokens: int = 2048
    cache_refresh_tokens: int = 256
    preview_tokens: int = 120

    def __post_init__(self):
        validate_rules({
            "preview_tokens": (type(self.preview_tokens) is int and self.preview_tokens >= 0,
                               "must be a nonnegative integer; 0 disables the preview"),
            "num_cycles": (self.num_cycles >= 1, "must be positive"),
            "max_new_tokens": (self.max_new_tokens >= 1, "must be positive"),
            "cache_refresh_tokens": (self.cache_refresh_tokens >= 1, "must be positive"),
            "top_k": (self.top_k >= 0, "must be nonnegative"),
            "temperature": (math.isfinite(self.temperature) and self.temperature >= 0,
                            "must be nonnegative and finite"),
            "top_p": (math.isfinite(self.top_p) and 0 < self.top_p <= 1, "must be finite in (0, 1]"),
        })


def run_inference(snapshot: str | Path, prompt: str | Path,
                  output_root: str | Path, config: InferenceConfig) -> dict:
    import torch
    from transformers import AutoModel, AutoProcessor

    snapshot, prompt, output_root = Path(snapshot), Path(prompt), Path(output_root)
    print(f"[inference] Loading model and processor from {snapshot}", flush=True)
    processor = AutoProcessor.from_pretrained(snapshot, trust_remote_code=True)
    model = AutoModel.from_pretrained(snapshot, trust_remote_code=True).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    prompt_ids = processor(derbake_path=prompt, return_tensors="pt")["input_ids"].to(device)
    print(f"[inference] Prompt: {prompt.name} | {prompt_ids.shape[-1]} tokens | "
          f"device: {device} | seed: {config.seed}", flush=True)
    print(f"[inference] Generating {config.num_cycles} valid cycles "
          f"(limit: {config.max_new_tokens} new tokens)...", flush=True)
    with torch.inference_mode():
        generated = model.generate(prompt_ids, num_cycles=config.num_cycles,
                                   max_new_tokens=config.max_new_tokens,
                                   temperature=config.temperature, top_k=config.top_k,
                                   top_p=config.top_p, seed=config.seed,
                                   cache_refresh_tokens=config.cache_refresh_tokens,
                                   return_generation_report=True)
    if config.preview_tokens:
        ids = generated.continuation_ids[0]
        count = min(config.preview_tokens, len(ids))
        atoms = processor.tokenizer.decode_atoms(ids[:count].tolist())
        suffix = " ..." if count < len(ids) else ""
        print(f"[inference] Generated-token preview ({count}/{len(ids)} tokens):\n"
              + " ".join(atoms) + suffix, flush=True)
    print("[inference] Decoding generated tokens into rhythm cycles...", flush=True)
    rhythm = processor.decode_generation(generated.continuation_ids[0],
                                         expected_num_cycles=config.num_cycles)
    output_name = f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}_{prompt.stem}_s{config.seed}"
    output_dir = output_root / output_name
    suffix = 2
    while output_dir.exists():
        output_dir = output_root / f"{output_name}_{suffix}"
        suffix += 1
    output_dir.mkdir(parents=True)
    (output_dir / "prompt.derbake").write_bytes(prompt.read_bytes())
    output_file = output_dir / "generated.derbake"
    print(f"[inference] Writing five-line .derbake: {len(rhythm.cycles)} cycles, "
          f"{rhythm.beat_count} beats | cycle lengths: {rhythm.cycle_lengths}", flush=True)
    processor.save_derbake(rhythm, output_file)
    reloaded_ids = processor(derbake_path=output_file, return_tensors="pt")["input_ids"][0]
    if not torch.equal(reloaded_ids.cpu(), generated.continuation_ids[0].cpu()):
        raise RuntimeError("Generated file failed token round trip")
    report = {"snapshot": str(snapshot), "prompt": str(prompt),
              "config": asdict(config), "generation": generated.report,
              "output": str(output_file)}
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[inference] Saved and verified: {output_file}", flush=True)
    print(f"[inference] Generation report: {generated.report}", flush=True)
    return report


def contextual_projection(snapshot: str | Path, dataset_dir: str | Path,
                          output_dir: str | Path, *, split: str = "val",
                          max_rows: int = 16, max_points: int = 2000) -> Path:
    """Save bounded contextual-token PCA coordinates for optional inspection."""
    import torch
    from transformers import AutoModel
    from .data_loading import ChunkDataset
    from .dataset_integrity import verify_dataset

    if min(max_rows, max_points) < 1:
        raise ValueError("Sampling bounds must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = AutoModel.from_pretrained(snapshot, trust_remote_code=True).eval()
    verify_dataset(dataset_dir, model_config=model.config.to_dict())
    dataset = ChunkDataset(dataset_dir, split)
    hidden, labels = [], []
    captured = []
    hook = model.core.layer_normalization.register_forward_hook(
        lambda module, inputs, output: captured.append(output.detach().cpu()))
    try:
        with torch.inference_mode():
            for row in range(min(max_rows, len(dataset))):
                batch = dataset.batch([row], torch.device("cpu"))
                captured.clear()
                model(**{key: value for key, value in batch.items() if key != "labels"})
                mask = batch["labels"][0] != -100
                indices = torch.nonzero(mask).flatten().tolist()
                for index in indices:
                    hidden.append(captured[0][0, index].numpy())
                    labels.append(model.vocab[int(batch["labels"][0, index])])
                    if len(hidden) >= max_points:
                        break
                if len(hidden) >= max_points:
                    break
    finally:
        hook.remove()
    if len(hidden) < 2:
        raise ValueError("Not enough scored tokens for contextual projection")
    matrix = np.stack(hidden)
    matrix -= matrix.mean(axis=0)
    u, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    coordinates = u[:, :2] * singular[:2]
    if coordinates.shape[1] == 1:
        coordinates = np.pad(coordinates, ((0, 0), (0, 1)))
    import csv
    target = output_dir / "contextual_pca.csv"
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["target_token", "pc1", "pc2"])
        writer.writerows((token, float(x), float(y)) for token, (x, y) in zip(labels, coordinates))
    (output_dir / "contextual_pca_config.json").write_text(json.dumps({
        "snapshot": str(snapshot), "dataset": str(dataset_dir), "split": split,
        "max_rows": max_rows, "max_points": max_points, "sampled_points": len(labels)}, indent=2) + "\n",
        encoding="utf-8")
    return target
