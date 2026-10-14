"""Strict five-line .derbake codec shared by training and HF inference."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path


HITS = frozenset({
    "HIT_S", "HIT_D", "HIT_OTA", "HIT_OTI", "HIT_PA2",
    "HIT_D_decorated", "HIT_OTA_decorated", "HIT_PA2_decorated",
})
MAX_SUBDIVISION = 8
_HIT = re.compile(r"HIT_[A-Za-z0-9]+(?:_decorated)?(?:_SKEL)?\Z")
_POSITIVE_INT = re.compile(r"[1-9][0-9]*\Z")


@dataclass(frozen=True)
class Beat:
    subdivision: int
    hits: tuple[str, ...]

    def __post_init__(self) -> None:
        if not 1 <= self.subdivision <= MAX_SUBDIVISION:
            raise ValueError(f"Unsupported subdivision: {self.subdivision}")
        if len(self.hits) != self.subdivision or any(hit not in HITS for hit in self.hits):
            raise ValueError("Beat needs one allowed HIT per subdivision slot")


@dataclass(frozen=True)
class RhythmSequence:
    cycles: tuple[tuple[Beat, ...], ...]

    def __post_init__(self) -> None:
        if not self.cycles or any(not cycle for cycle in self.cycles):
            raise ValueError("A rhythm needs nonempty complete cycles")

    @property
    def cycle_lengths(self) -> tuple[int, ...]:
        return tuple(len(cycle) for cycle in self.cycles)

    @property
    def beat_count(self) -> int:
        return sum(self.cycle_lengths)


@dataclass(frozen=True)
class SkeletonEvent:
    time_in_beats: float
    hit: str
    deviation: float


@dataclass(frozen=True)
class DerbakeDocument:
    rhythm: RhythmSequence
    initial_tempo: float
    tempos: tuple[float, ...]
    skeleton_events: tuple[SkeletonEvent, ...]
    variation_amplitudes: tuple[tuple[float, ...], ...]


def _finite_number(text: str, *, positive: bool = False) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(f"Invalid number: {text!r}") from exc
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"Invalid number: {text!r}")
    return value


def _clean_hit(token: str) -> str:
    if not _HIT.fullmatch(token):
        raise ValueError(f"Invalid HIT: {token!r}")
    hit = token.removesuffix("_SKEL")
    if hit not in HITS:
        raise ValueError(f"Unsupported HIT: {token!r}")
    return hit


def _parse_variation(tokens: list[str]) -> tuple[list[Beat], list[tuple[float, ...]]]:
    beats: list[Beat] = []
    amplitudes: list[tuple[float, ...]] = []
    i = 0
    while i < len(tokens):
        marker = tokens[i]
        if not marker.startswith("SUBD_") or not _POSITIVE_INT.fullmatch(marker[5:]):
            raise ValueError(f"Expected SUBD_n at variation token {i}")
        subd = int(marker[5:])
        if subd > MAX_SUBDIVISION:
            raise ValueError(f"Unsupported {marker}")
        i += 1
        hits = []
        amps = []
        for _ in range(subd):
            if i + 1 >= len(tokens):
                raise ValueError(f"Incomplete {marker} beat")
            hits.append(_clean_hit(tokens[i]))
            if not tokens[i + 1].startswith("AMP_"):
                raise ValueError(f"Missing AMP after variation token {i}")
            amps.append(_finite_number(tokens[i + 1][4:]))
            if amps[-1] < 0:
                raise ValueError("Amplitude must be nonnegative")
            i += 2
        beats.append(Beat(subd, tuple(hits)))
        amplitudes.append(tuple(amps))
    return beats, amplitudes


def _parse_skeleton(tokens: list[str], beat_count: int) -> tuple[SkeletonEvent, ...]:
    if len(tokens) % 3:
        raise ValueError("Skeleton needs DELAY/HIT/DEV triples")
    events = []
    current = 0.0
    for i in range(0, len(tokens), 3):
        delay, hit, deviation = tokens[i:i + 3]
        if not delay.startswith("DELAY_") or not deviation.startswith("DEV_"):
            raise ValueError(f"Invalid skeleton triple at token {i}")
        delta = _finite_number(delay[6:])
        if delta < 0:
            raise ValueError("Skeleton delays must be nonnegative")
        current += delta
        if current >= beat_count:
            raise ValueError("Skeleton event lies outside the final cycle")
        events.append(SkeletonEvent(current, _clean_hit(hit), _finite_number(deviation[4:])))
    return tuple(events)


def parse_derbake(path: str | Path, *, max_beats_per_cycle: int | None = None) -> DerbakeDocument:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if len(lines) != 5:
        raise ValueError(f"Expected exactly five physical lines; got {len(lines)}")
    initial = _finite_number(lines[0].strip(), positive=True)
    tempos = tuple(_finite_number(value, positive=True) for value in lines[1].split())
    length_tokens = lines[2].split()
    if not length_tokens or any(not _POSITIVE_INT.fullmatch(value) for value in length_tokens):
        raise ValueError("Line 3 needs positive integer cycle lengths")
    lengths = tuple(map(int, length_tokens))
    if max_beats_per_cycle is not None and any(n > max_beats_per_cycle for n in lengths):
        raise ValueError("Cycle exceeds model capacity")
    beats, amplitudes = _parse_variation(lines[4].split())
    if sum(lengths) != len(beats) or len(tempos) != len(beats):
        raise ValueError("Tempo count, cycle lengths, and variation beats disagree")
    events = _parse_skeleton(lines[3].split(), len(beats))
    # Preserve old notebook behavior: a skeleton hit overrides its variation slot.
    normalized = [list(beat.hits) for beat in beats]
    for event in events:
        beat_index = int(event.time_in_beats)
        slot = int((event.time_in_beats - beat_index) * beats[beat_index].subdivision)
        slot = min(beats[beat_index].subdivision - 1, max(0, slot))
        normalized[beat_index][slot] = event.hit
    normalized_beats = [Beat(beat.subdivision, tuple(hits)) for beat, hits in zip(beats, normalized)]
    cycles = []
    start = 0
    for length in lengths:
        cycles.append(tuple(normalized_beats[start:start + length]))
        start += length
    return DerbakeDocument(RhythmSequence(tuple(cycles)), initial, tempos, events,
                           tuple(amplitudes))


def write_derbake(rhythm: RhythmSequence, path: str | Path) -> None:
    """Write generated rhythm; tempo/amplitude are agreed fixed output values."""
    parts = []
    for cycle in rhythm.cycles:
        for beat in cycle:
            parts.append(f"SUBD_{beat.subdivision}")
            for hit in beat.hits:
                parts.extend((hit, "AMP_1.5"))
    lines = ["120", " ".join(["120"] * rhythm.beat_count),
             " ".join(map(str, rhythm.cycle_lengths)), "", " ".join(parts)]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
