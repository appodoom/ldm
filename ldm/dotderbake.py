"""Render validated five-line .derbake rhythms with percussion samples."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import AudioSamples
from .derbake_ldm import parse_derbake


def apply_cross_fade(hit_audio: np.ndarray, fade_samples: int = 500) -> np.ndarray:
    """Apply the original cosine fade while leaving the cached sample intact."""
    audio = np.asarray(hit_audio, dtype=np.float32).copy()
    if audio.size < 2:
        return audio
    fade_length = min(fade_samples, max(1, audio.size // 4))
    phase = np.linspace(0.0, np.pi, fade_length, dtype=np.float32)
    audio[:fade_length] *= 0.5 * (1.0 - np.cos(phase))
    audio[-fade_length:] *= 0.5 * (1.0 + np.cos(phase))
    return audio


def render_wav(derbake_path: str | Path, sample_dirs: dict[str, str | Path],
               output_path: str | Path, *, sample_rate: int = 48000,
               seed: int = 1337, skeleton_gain: float = 3.0) -> Path:
    """Mix variation and skeleton events without overlaying either stream twice."""
    if sample_rate < 8000:
        raise ValueError("sample_rate must be at least 8000")
    if not np.isfinite(skeleton_gain) or skeleton_gain < 0:
        raise ValueError("skeleton_gain must be finite and nonnegative")
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError('Install audio extras: python -m pip install -e ".[audio]"') from exc

    document = parse_derbake(derbake_path, overlay_skeleton=False)
    samples = AudioSamples(sample_dirs, sample_rate, seed)
    prepared: dict[Path, np.ndarray] = {}

    def sample_for(hit: str) -> np.ndarray | None:
        selected = samples.get_audio_data(hit)
        if selected is None:
            return None
        path, audio = selected
        if path not in prepared:
            prepared[path] = apply_cross_fade(audio)
        return prepared[path]

    beat_lengths = [max(1, round(60.0 * sample_rate / tempo)) for tempo in document.tempos]
    beat_starts = [0]
    for length in beat_lengths:
        beat_starts.append(beat_starts[-1] + length)

    placements: list[tuple[int, np.ndarray, float]] = []
    output_length = beat_starts[-1]
    beats = (beat for cycle in document.rhythm.cycles for beat in cycle)
    for beat_index, (beat, amplitudes) in enumerate(zip(beats, document.variation_amplitudes)):
        for slot, (hit, amplitude) in enumerate(zip(beat.hits, amplitudes)):
            if hit == "HIT_S" or amplitude == 0:
                continue
            audio = sample_for(hit)
            position = beat_starts[beat_index] + round(slot * beat_lengths[beat_index] / beat.subdivision)
            placements.append((position, audio, amplitude))
            output_length = max(output_length, position + len(audio))

    if skeleton_gain:
        for event in document.skeleton_events:
            if event.hit == "HIT_S":
                continue
            beat_index = int(event.time_in_beats)
            fraction = event.time_in_beats - beat_index
            position = beat_starts[beat_index] + round(fraction * beat_lengths[beat_index])
            position = max(0, position + round(event.deviation))
            audio = sample_for(event.hit)
            placements.append((position, audio, skeleton_gain))
            output_length = max(output_length, position + len(audio))

    output = np.zeros(output_length, dtype=np.float32)
    for position, audio, gain in placements:
        output[position:position + len(audio)] += gain * audio
    peak = float(np.max(np.abs(output))) if output.size else 0.0
    if peak > 0.99:
        output *= 0.99 / peak

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(target, output, sample_rate, subtype="PCM_16")
    print(f"[audio] {Path(derbake_path).name} -> {target} "
          f"({document.rhythm.beat_count} beats, {len(output) / sample_rate:.2f}s)")
    return target
