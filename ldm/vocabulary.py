"""Immutable atomic token inventory and fingerprints."""

from __future__ import annotations

import hashlib
import json

from .derbake_ldm import HITS, MAX_SUBDIVISION

STRUCTURAL = ("<PAD>", "<SOC>", "<EOC>", "<SOB>", "<EOB>")
HIT_ORDER = (
    "HIT_S", "HIT_D", "HIT_OTA", "HIT_OTI", "HIT_PA2",
    "HIT_D_decorated", "HIT_OTA_decorated", "HIT_PA2_decorated",
)
assert set(HIT_ORDER) == HITS


def atomic_vocabulary(variant: str) -> tuple[str, ...]:
    if variant not in {"with_subd", "no_subd"}:
        raise ValueError(f"Unknown variant: {variant}")
    subd = tuple(f"SUBD_{i}" for i in range(1, MAX_SUBDIVISION + 1)) if variant == "with_subd" else ()
    return STRUCTURAL + subd + HIT_ORDER


def vocabulary_hash(tokens: list[str] | tuple[str, ...]) -> str:
    if len(set(tokens)) != len(tokens):
        raise ValueError("Duplicate vocabulary atom")
    raw = json.dumps(list(tokens), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class AtomicVocabulary:
    def __init__(self, tokens: list[str] | tuple[str, ...]):
        self.tokens = tuple(tokens)
        self.hash = vocabulary_hash(self.tokens)
        self.to_id = {atom: i for i, atom in enumerate(self.tokens)}

    def encode(self, atoms: list[str]) -> list[int]:
        try:
            return [self.to_id[atom] for atom in atoms]
        except KeyError as exc:
            raise ValueError(f"Unknown atomic token: {exc.args[0]}") from exc

    def decode(self, ids: list[int]) -> list[str]:
        try:
            return [self.tokens[int(i)] for i in ids]
        except IndexError as exc:
            raise ValueError("Token ID outside vocabulary") from exc
