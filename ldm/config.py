"""Sound sample selection for optional .derbake audio previews."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np


HIT_FOLDERS = {
    "HIT_D": "doums",
    "HIT_OTA": "taks",
    "HIT_OTI": "tiks",
    "HIT_PA2": "pa2s",
}


class AudioSamples:
    """Choose samples lazily from separate normal and decorated libraries."""

    def __init__(self, roots: dict[str, str | Path], sample_rate: int, seed: int):
        self.roots = {name: Path(path).expanduser() for name, path in roots.items()}
        self.sample_rate = sample_rate
        self.random = random.Random(seed)
        self.files: dict[tuple[str, str], tuple[Path, ...]] = {}
        self.audio: dict[Path, np.ndarray] = {}

    def get_audio_data(self, hit: str) -> tuple[Path, np.ndarray] | None:
        if hit == "HIT_S":
            return None
        decorated = hit.endswith("_decorated")
        base = hit.removesuffix("_decorated")
        if base not in HIT_FOLDERS:
            raise ValueError(f"Unsupported audio HIT: {hit}")
        group = "decorated_hits" if decorated else "normal_hits"
        if group not in self.roots:
            raise ValueError(f"Missing sample root: {group}")
        key = (group, base)
        if key not in self.files:
            folder = self.roots[group] / HIT_FOLDERS[base]
            if not folder.is_dir():
                raise FileNotFoundError(f"Sound folder missing for {hit}: {folder}")
            files = tuple(path for path in sorted(folder.iterdir())
                          if path.is_file() and path.suffix.lower() == ".wav")
            if not files:
                raise FileNotFoundError(f"No WAV samples for {hit} in {folder}")
            self.files[key] = files
        path = self.random.choice(self.files[key])
        if path not in self.audio:
            try:
                import soundfile as sf
            except ImportError as exc:
                raise RuntimeError('Install audio extras: python -m pip install -e ".[audio]"') from exc
            audio, source_rate = sf.read(path, dtype="float32", always_2d=False)
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            if source_rate != self.sample_rate:
                import librosa
                audio = librosa.resample(audio, orig_sr=source_rate, target_sr=self.sample_rate)
            audio = np.asarray(audio, dtype=np.float32)
            if not audio.size:
                raise ValueError(f"Empty audio sample: {path}")
            self.audio[path] = audio
        return path, self.audio[path]
