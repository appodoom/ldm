"""Verified, variant-independent normalized source arrays shared by preparation runs."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np

from .constants import HIT_ORDER, MAX_SUBDIVISION, NORMALIZATION_RULE, PREPARATION_VERSION
from .dataset_integrity import _sha256
from .derbake_ldm import Beat, parse_derbake


class NormalizedSource:
    def __init__(self, folder: Path, recipe: dict):
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if manifest["recipe"] != recipe:
            raise ValueError(f"Normalized cache recipe differs: {folder}")
        expected = {"cycles": np.dtype("int64"), "beats": np.dtype("int64"),
                    "hits": np.dtype("uint8"), "skeleton": np.dtype("bool")}
        if set(manifest["arrays"]) != set(expected):
            raise ValueError(f"Normalized cache inventory differs: {folder}")
        for name, dtype in expected.items():
            path = folder / f"{name}.npy"
            info = manifest["arrays"][name]
            if _sha256(path) != info["sha256"]:
                raise ValueError(f"Normalized cache content check failed: {path}")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.dtype != dtype or list(array.shape) != info["shape"] or array.ndim != 1:
                raise ValueError(f"Invalid normalized cache array: {path}")
            setattr(self, name, array)
        if (len(self.cycles) < 2 or len(self.beats) < 2 or self.cycles[0] != 0 or self.beats[0] != 0
                or self.cycles[-1] != len(self.beats) - 1 or self.beats[-1] != len(self.hits)
                or len(self.skeleton) != len(self.hits) or np.any(np.diff(self.cycles) <= 0)
                or np.any(np.diff(self.beats) < 1) or np.any(np.diff(self.beats) > MAX_SUBDIVISION)
                or np.any(self.hits >= len(HIT_ORDER))):
            raise ValueError(f"Invalid normalized source boundaries: {folder}")

    def iter_cycles(self, start: int, end: int):
        """Materialize at most one cycle of Beat objects at a time."""
        for cycle in range(start, end):
            yield tuple(Beat(int(self.beats[b + 1] - self.beats[b]),
                             tuple(HIT_ORDER[int(hit)] for hit in self.hits[self.beats[b]:self.beats[b + 1]]))
                        for b in range(int(self.cycles[cycle]), int(self.cycles[cycle + 1])))

    def atom_mask(self, start: int, end: int, variant: str) -> list[bool]:
        mask = []
        for cycle in range(start, end):
            mask.append(False)  # <SOC>
            for beat in range(int(self.cycles[cycle]), int(self.cycles[cycle + 1])):
                mask.extend([False] * (2 if variant == "with_subd" else 1))  # SOB, optional SUBD
                mask.extend(self.skeleton[self.beats[beat]:self.beats[beat + 1]].tolist())
                mask.append(False)  # <EOB>
            mask.append(False)  # <EOC>
        return mask


def load_normalized_source(source: Path, cache_root: Path, source_hash: str) -> NormalizedSource:
    recipe = {"source_hash": source_hash, "hits": list(HIT_ORDER),
              "preparation_version": PREPARATION_VERSION, "normalization_rule": NORMALIZATION_RULE}
    key = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    cache_root.mkdir(parents=True, exist_ok=True)
    folder = cache_root / key
    if not folder.exists():
        document = parse_derbake(source)
        if _sha256(source) != source_hash:
            raise ValueError(f"Source changed during normalization: {source}")
        cycles = np.r_[np.int64(0), np.cumsum(document.rhythm.cycle_lengths, dtype=np.int64)]
        flat_beats = [beat for cycle in document.rhythm.cycles for beat in cycle]
        beats = np.r_[np.int64(0), np.cumsum([beat.subdivision for beat in flat_beats], dtype=np.int64)]
        hit_ids = {hit: index for index, hit in enumerate(HIT_ORDER)}
        hits = np.fromiter((hit_ids[hit] for beat in flat_beats for hit in beat.hits),
                           dtype=np.uint8, count=int(beats[-1]))
        skeleton = np.zeros(len(hits), dtype=bool)
        for event in document.skeleton_events:
            beat = int(event.time_in_beats)
            subdivision = flat_beats[beat].subdivision
            slot = min(subdivision - 1, max(0, int((event.time_in_beats - beat) * subdivision)))
            skeleton[beats[beat] + slot] = True
        del document, flat_beats
        # Only complete directories become visible; interrupted writes are never reused.
        with tempfile.TemporaryDirectory(prefix=".normalizing-", dir=cache_root) as staging:
            temporary = Path(staging) / "payload"
            temporary.mkdir()
            inventory = {}
            for name, array in (("cycles", cycles), ("beats", beats), ("hits", hits), ("skeleton", skeleton)):
                path = temporary / f"{name}.npy"
                np.save(path, array, allow_pickle=False)
                inventory[name] = {"sha256": _sha256(path), "shape": list(array.shape)}
            (temporary / "manifest.json").write_text(json.dumps({"recipe": recipe, "arrays": inventory}),
                                                    encoding="utf-8")
            try:
                temporary.rename(folder)
            except OSError:
                if not folder.exists():
                    raise
    else:
        print(f"[prepare] Reusing normalized source {source.name}", flush=True)
    return NormalizedSource(folder, recipe)
