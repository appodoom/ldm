"""Optional inference review, sample audio and embedding inspection."""

from __future__ import annotations

import json
import math
import wave
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class InferenceConfig:
    num_cycles: int = 1
    seed: int = 1337
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    max_new_tokens: int = 2048
    cache_refresh_tokens: int = 256

    def __post_init__(self):
        if min(self.num_cycles, self.max_new_tokens, self.cache_refresh_tokens) < 1:
            raise ValueError("Cycle and token limits must be positive")
        if self.top_k < 0 or not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("Invalid sampling settings")
        if not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")


def review_generation(snapshot: str | Path, prompt: str | Path,
                      output_root: str | Path, config: InferenceConfig) -> dict:
    import torch
    from transformers import AutoModel, AutoProcessor
    from .derbake_ldm import parse_derbake

    snapshot, prompt, output_root = Path(snapshot), Path(prompt), Path(output_root)
    processor = AutoProcessor.from_pretrained(snapshot, trust_remote_code=True)
    model = AutoModel.from_pretrained(snapshot, trust_remote_code=True).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    prompt_ids = processor(derbake_path=prompt, return_tensors="pt")["input_ids"].to(device)
    with torch.inference_mode():
        generated = model.generate(prompt_ids, num_cycles=config.num_cycles,
                                   max_new_tokens=config.max_new_tokens,
                                   temperature=config.temperature, top_k=config.top_k,
                                   top_p=config.top_p, seed=config.seed,
                                   cache_refresh_tokens=config.cache_refresh_tokens,
                                   return_generation_report=True)
    rhythm = processor.decode_generation(generated.continuation_ids[0],
                                         expected_num_cycles=config.num_cycles)
    review = output_root / f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}_{prompt.stem}_s{config.seed}"
    suffix = 2
    while review.exists():
        review = output_root / f"{review.name}_{suffix}"
        suffix += 1
    review.mkdir(parents=True)
    (review / "prompt.derbake").write_bytes(prompt.read_bytes())
    processor.save_derbake(rhythm, review / "generated.derbake")
    if parse_derbake(review / "generated.derbake").rhythm != rhythm:
        raise RuntimeError("Generated file failed round trip")
    report = {"snapshot": str(snapshot), "prompt": str(prompt),
              "config": asdict(config), "generation": generated.report,
              "output": str(review / "generated.derbake")}
    (review / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def render_wav(derbake_path: str | Path, sample_dirs: dict[str, str | Path],
               output_path: str | Path, *, sample_rate: int = 22050) -> Path:
    """Render a five-line rhythm with explicit WAV sample directories."""
    from .derbake_ldm import parse_derbake

    if sample_rate < 8000:
        raise ValueError("sample_rate must be at least 8000")
    document = parse_derbake(derbake_path)
    stems = {"HIT_S": None, "HIT_D": "doums", "HIT_OTA": "taks",
             "HIT_OTI": "tiks", "HIT_PA2": "pa2s"}
    samples = {}
    for hit, folder in stems.items():
        if folder is None:
            continue
        if folder not in sample_dirs:
            raise ValueError(f"Missing sample directory for {folder}")
        choices = sorted(Path(sample_dirs[folder]).glob("*.wav"))
        if not choices:
            raise ValueError(f"No WAV samples in {sample_dirs[folder]}")
        with wave.open(str(choices[0]), "rb") as handle:
            if handle.getnchannels() != 1 or handle.getsampwidth() != 2 or handle.getframerate() != sample_rate:
                raise ValueError(f"Sample {choices[0]} must be mono 16-bit PCM at {sample_rate} Hz")
            samples[hit] = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").astype(np.float32) / 32768
    beat_cursor = 0.0
    events = []
    for beat, tempo, amplitudes in zip((beat for cycle in document.rhythm.cycles for beat in cycle),
                                       document.tempos, document.variation_amplitudes):
        seconds_per_beat = 60.0 / tempo
        for slot, (hit, amplitude) in enumerate(zip(beat.hits, amplitudes)):
            key = hit.removesuffix("_decorated")
            if key != "HIT_S" and amplitude > 0:
                events.append((int((beat_cursor + slot * seconds_per_beat / beat.subdivision) * sample_rate),
                               key, amplitude))
        beat_cursor += seconds_per_beat
    length = int((beat_cursor + 1.0) * sample_rate)
    audio = np.zeros(length, dtype=np.float32)
    for index, hit, amplitude in events:
        sample = samples[hit]
        end = min(index + len(sample), length)
        audio[index:end] += sample[:end - index] * amplitude
    peak = float(np.max(np.abs(audio))) if len(audio) else 0
    if peak > 1:
        audio /= peak
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes((audio * 32767).astype("<i2").tobytes())
    return path


def embedding_neighbors(snapshot: str | Path, tokens: list[str], *, top_n: int = 8) -> dict:
    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained(snapshot, trust_remote_code=True).eval()
    vocab = list(model.vocab)
    matrix = model.get_input_embeddings().weight.detach().float()
    normalized = torch.nn.functional.normalize(matrix, dim=1)
    scores = normalized @ normalized.T
    result = {}
    for token in tokens:
        if token not in vocab:
            raise ValueError(f"Unknown token: {token}")
        index = vocab.index(token)
        nearest = torch.topk(scores[index], min(top_n + 1, len(vocab))).indices.tolist()
        result[token] = [{"token": vocab[i], "cosine": float(scores[index, i])}
                         for i in nearest if i != index][:top_n]
    return result


def contextual_projection(snapshot: str | Path, dataset_dir: str | Path,
                          output_dir: str | Path, *, split: str = "val",
                          max_rows: int = 16, max_points: int = 2000) -> Path:
    """Save bounded contextual-token PCA coordinates for optional inspection."""
    import torch
    from transformers import AutoModel
    from .train_ldm import ChunkDataset

    if min(max_rows, max_points) < 1:
        raise ValueError("Sampling bounds must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = AutoModel.from_pretrained(snapshot, trust_remote_code=True).eval()
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
