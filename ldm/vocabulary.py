"""Immutable atomic token inventory and fingerprints."""

from __future__ import annotations

import hashlib
import json

from .constants import HIT_ORDER, STRUCTURAL, SUBDIVISION_TOKENS

def atomic_vocabulary(variant: str) -> tuple[str, ...]:
    if variant not in {"with_subd", "no_subd"}:
        raise ValueError(f"Unknown variant: {variant}")
    subd = SUBDIVISION_TOKENS if variant == "with_subd" else ()
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
        result = []
        for raw_id in ids:
            index = int(raw_id)
            if index < 0 or index >= len(self.tokens):
                raise ValueError("Token ID outside vocabulary")
            result.append(self.tokens[index])
        return result
