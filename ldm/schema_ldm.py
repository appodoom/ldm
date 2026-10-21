"""Bidirectional model grammars over the shared canonical rhythm."""

from __future__ import annotations

from .derbake_ldm import Beat, RhythmSequence
from .constants import HITS, SUBDIVISION_TOKENS


_SUBDIVISIONS = {token: size for size, token in enumerate(SUBDIVISION_TOKENS, 1)}


class RhythmSchema:
    def __init__(self, variant: str, max_beats_per_cycle: int):
        if variant not in {"with_subd", "no_subd"}:
            raise ValueError(f"Unknown variant: {variant}")
        if max_beats_per_cycle < 1:
            raise ValueError("max_beats_per_cycle must be positive")
        self.variant = variant
        self.max_beats_per_cycle = max_beats_per_cycle

    def encode(self, rhythm: RhythmSequence) -> list[str]:
        atoms = []
        for cycle in rhythm.cycles:
            if len(cycle) > self.max_beats_per_cycle:
                raise ValueError("Cycle exceeds model capacity")
            atoms.append("<SOC>")
            for beat in cycle:
                atoms.append("<SOB>")
                if self.variant == "with_subd":
                    atoms.append(f"SUBD_{beat.subdivision}")
                atoms.extend(beat.hits)
                atoms.append("<EOB>")
            atoms.append("<EOC>")
        return atoms

    def decode(self, atoms: list[str]) -> RhythmSequence:
        cycles = []
        i = 0
        while i < len(atoms):
            if atoms[i] != "<SOC>":
                raise ValueError(f"Expected <SOC> at atom {i}")
            i += 1
            beats = []
            while i < len(atoms) and atoms[i] != "<EOC>":
                if atoms[i] != "<SOB>":
                    raise ValueError(f"Expected <SOB> at atom {i}")
                i += 1
                subdivision = None
                if self.variant == "with_subd":
                    if i >= len(atoms) or atoms[i] not in _SUBDIVISIONS:
                        raise ValueError("Valid explicit subdivision is missing")
                    subdivision = _SUBDIVISIONS[atoms[i]]
                    i += 1
                hits = []
                while i < len(atoms) and atoms[i] in HITS:
                    hits.append(atoms[i])
                    i += 1
                if i >= len(atoms) or atoms[i] != "<EOB>":
                    raise ValueError("Beat is missing <EOB> or contains an invalid atom")
                i += 1
                beats.append(Beat(subdivision if subdivision is not None else len(hits), tuple(hits)))
                if len(beats) > self.max_beats_per_cycle:
                    raise ValueError("Cycle exceeds model capacity")
            if i >= len(atoms) or not beats:
                raise ValueError("Cycle is incomplete or empty")
            i += 1  # <EOC>
            cycles.append(tuple(beats))
        return RhythmSequence(tuple(cycles))

    def validate_candidate(self, atoms: list[str]) -> None:
        rhythm = self.decode(atoms)
        if len(rhythm.cycles) != 1:
            raise ValueError("Candidate must contain exactly one cycle")
