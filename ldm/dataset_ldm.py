"""Cycle-aligned per-file dataset preparation and structured chunk arrays."""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import hashlib
import json
import numpy as np
from .derbake_ldm import RhythmSequence, parse_derbake
from .schema_ldm import RhythmSchema
from .vocabulary import AtomicVocabulary, atomic_vocabulary
from .splits import split_cycle_ranges
from .rhythm_ldm import (STRUCTURED_ROUTING_FIELDS, count_chunks, get_chunk_starts,
    build_rhythm_lookup_arrays, compute_rhythm_ids_for_sequence,
    compute_hierarchy_ids_for_sequence, build_structured_routing_for_window,
    build_target_ownership_masks_for_sequence)

def write_split_chunks(
    split_name: str,
    sequences: List[List[int]],
    out_dir: Path,
    window_len: int,
    stride: int,
    pad: bool,
    pad_id: int,
    rhythm_info: Dict[str, np.ndarray],
    max_beats_per_cycle: int,
    create_target_masks: bool = False,
    source_kinds: List[int] | None = None,
    dtype=np.int32,
) -> Dict[str, int]:
    """Write token, rhythm, ownership, and structured-routing arrays."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{split_name}.npy"
    kind_path = out_dir / f"{split_name}_source_kind.npy"
    beat_idx_path = out_dir / f"{split_name}_beat_idx.npy"
    beat_pos_path = out_dir / f"{split_name}_beat_pos.npy"
    score_mask_path = out_dir / f"{split_name}_score_mask.npy"
    hit_count_mask_path = out_dir / f"{split_name}_hit_count_mask.npy"
    routing_paths = {
        field: out_dir / f"{split_name}_{field}.npy"
        for field in STRUCTURED_ROUTING_FIELDS
    }

    parsed_files = []
    total_chunks = 0
    files_with_chunks = 0
    if source_kinds is None:
        source_kinds = [0] * len(sequences)
    if len(source_kinds) != len(sequences):
        raise ValueError("Each sequence needs a source kind")
    for ids in sequences:
        num_chunks = count_chunks(len(ids), window_len, stride, pad)
        parsed_files.append((ids, num_chunks))
        total_chunks += num_chunks
        files_with_chunks += int(num_chunks > 0)

    if total_chunks == 0:
        empty_int = np.empty((0, window_len), dtype=np.int32)
        empty_bool = np.empty((0, window_len), dtype=bool)
        np.save(out_path, np.empty((0, window_len), dtype=dtype))
        np.save(kind_path, np.empty(0, dtype=np.uint8))
        np.save(beat_idx_path, empty_int)
        np.save(beat_pos_path, empty_int)
        for field, path in routing_paths.items():
            np.save(path, empty_bool if field.endswith("_mask") else empty_int)
        if create_target_masks:
            empty_mask = np.empty((0, window_len - 1), dtype=bool)
            np.save(score_mask_path, empty_mask)
            np.save(hit_count_mask_path, empty_mask)
        return {
            "chunks": 0,
            "sequences_seen": len(sequences),
            "sequences_used": 0,
            "unique_targets": 0,
            "unique_hit_count_events": 0,
            "target_masks": bool(create_target_masks),
            "max_tokens_per_beat": 0,
            "max_beat_slots_per_chunk": 0,
            "max_cycles_per_chunk": 0,
        }

    shape = (total_chunks, window_len)
    arr = np.lib.format.open_memmap(out_path, mode="w+", dtype=dtype, shape=shape)
    kind_arr = np.lib.format.open_memmap(kind_path, mode="w+", dtype=np.uint8,
                                         shape=(total_chunks,))
    beat_idx_arr = np.lib.format.open_memmap(
        beat_idx_path, mode="w+", dtype=np.int32, shape=shape
    )
    beat_pos_arr = np.lib.format.open_memmap(
        beat_pos_path, mode="w+", dtype=np.int32, shape=shape
    )
    routing_arrays = {
        field: np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=bool if field.endswith("_mask") else np.int32,
            shape=shape,
        )
        for field, path in routing_paths.items()
    }

    score_mask_arr = None
    hit_count_mask_arr = None
    if create_target_masks:
        mask_shape = (total_chunks, window_len - 1)
        score_mask_arr = np.lib.format.open_memmap(
            score_mask_path, mode="w+", dtype=bool, shape=mask_shape
        )
        hit_count_mask_arr = np.lib.format.open_memmap(
            hit_count_mask_path, mode="w+", dtype=bool, shape=mask_shape
        )

    row = 0
    unique_targets = 0
    unique_hit_count_events = 0
    maxima = {
        "max_tokens_per_beat": 0,
        "max_beat_slots_per_chunk": 0,
        "max_cycles_per_chunk": 0,
    }

    for (ids, expected_chunks), source_kind in zip(parsed_files, source_kinds):
        if expected_chunks == 0:
            continue
        seq_len = len(ids)
        sequence_beat_idx, sequence_beat_pos = compute_rhythm_ids_for_sequence(
            ids, rhythm_info, max_beats_per_cycle
        )
        hierarchy = compute_hierarchy_ids_for_sequence(ids, rhythm_info)
        chunk_starts = get_chunk_starts(seq_len, window_len, stride, pad)
        if len(chunk_starts) != expected_chunks:
            raise RuntimeError("Chunk count changed between sizing and writing")

        file_score_masks = None
        file_hit_count_masks = None
        if create_target_masks:
            file_score_masks, file_hit_count_masks, mask_stats = (
                build_target_ownership_masks_for_sequence(
                    ids, chunk_starts, window_len, rhythm_info["subd_sizes"],
                    rhythm_info["hit_atom_counts"],
                )
            )
            unique_targets += mask_stats["unique_targets"]
            unique_hit_count_events += mask_stats["unique_hit_count_events"]

        for file_row, start in enumerate(chunk_starts):
            kind_arr[row] = source_kind
            real_end = min(start + window_len, seq_len)
            real_len = real_end - start
            chunk = np.full(window_len, pad_id, dtype=dtype)
            chunk[:real_len] = np.asarray(ids[start:real_end], dtype=dtype)
            arr[row] = chunk

            beat_idx_chunk = np.zeros(window_len, dtype=np.int32)
            beat_pos_chunk = np.zeros(window_len, dtype=np.int32)
            beat_idx_chunk[:real_len] = sequence_beat_idx[start:real_end]
            beat_pos_chunk[:real_len] = sequence_beat_pos[start:real_end]
            beat_idx_arr[row] = beat_idx_chunk
            beat_pos_arr[row] = beat_pos_chunk

            routing, routing_stats = build_structured_routing_for_window(
                hierarchy, start, real_end, window_len
            )
            for field in STRUCTURED_ROUTING_FIELDS:
                routing_arrays[field][row] = routing[field]
            for key in maxima:
                maxima[key] = max(maxima[key], routing_stats[key])

            if create_target_masks:
                score_mask_arr[row] = file_score_masks[file_row]
                hit_count_mask_arr[row] = file_hit_count_masks[file_row]
            row += 1

    if row != total_chunks:
        raise RuntimeError(
            f"{split_name}: wrote {row} chunks but expected {total_chunks} chunks"
        )
    arr.flush()
    kind_arr.flush()
    beat_idx_arr.flush()
    beat_pos_arr.flush()
    for routing_array in routing_arrays.values():
        routing_array.flush()
    if create_target_masks:
        score_mask_arr.flush()
        hit_count_mask_arr.flush()

    return {
        "chunks": total_chunks,
        "sequences_seen": len(sequences),
        "sequences_used": files_with_chunks,
        "unique_targets": unique_targets,
        "unique_hit_count_events": unique_hit_count_events,
        "target_masks": bool(create_target_masks),
        **maxima,
    }


def prepare_dataset(
    ordinary_dir: str | Path, output_dir: str | Path, *, variant: str,
    context_size: int, max_beats_per_cycle: int,
    decorated_dir: str | Path | None = None, stride: int | None = None,
    decorated_weight: float = 1.0,
) -> dict:
    """Create split-specific arrays, keeping paired versions on identical cycles."""
    ordinary_dir = Path(ordinary_dir)
    output_dir = Path(output_dir)
    files = sorted(ordinary_dir.glob("*.derbake"))
    if not files:
        raise ValueError(f"No .derbake files in {ordinary_dir}")
    if context_size < 2:
        raise ValueError("context_size must be at least 2")
    stride = max(1, context_size // 2) if stride is None else stride
    if not 1 <= stride <= context_size:
        raise ValueError("stride must be between 1 and context_size")
    decorated = Path(decorated_dir) if decorated_dir is not None else None
    if not np.isfinite(decorated_weight) or decorated_weight <= 0:
        raise ValueError("decorated_weight must be positive and finite")
    if decorated is not None and {p.name for p in files} != {p.name for p in decorated.glob("*.derbake")}:
        raise ValueError("Ordinary and decorated filenames must match exactly")
    schema = RhythmSchema(variant, max_beats_per_cycle)
    vocab = AtomicVocabulary(atomic_vocabulary(variant))
    sequences: dict[str, list[list[int]]] = {"train": [], "val": [], "test": []}
    source_kinds: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    split_files = []
    digest = hashlib.sha256()
    for source in files:
        documents = [parse_derbake(source, max_beats_per_cycle=max_beats_per_cycle)]
        digest.update(source.name.encode("utf-8"))
        digest.update(hashlib.sha256(source.read_bytes()).digest())
        if decorated is not None:
            pair = decorated / source.name
            documents.append(parse_derbake(pair, max_beats_per_cycle=max_beats_per_cycle))
            digest.update(hashlib.sha256(pair.read_bytes()).digest())
            if documents[1].rhythm.cycle_lengths != documents[0].rhythm.cycle_lengths:
                raise ValueError(f"Paired cycles differ in {source.name}")
        ranges = split_cycle_ranges(len(documents[0].rhythm.cycles))
        split_files.append({"file": source.name, "cycles": len(documents[0].rhythm.cycles),
                            "ranges": {key: list(value) for key, value in ranges.items()}})
        for source_kind, document in enumerate(documents):
            for split_name, (start, end) in ranges.items():
                piece = RhythmSequence(document.rhythm.cycles[start:end])
                sequences[split_name].append(vocab.encode(schema.encode(piece)))
                source_kinds[split_name].append(source_kind)
    output_dir.mkdir(parents=True, exist_ok=True)
    rhythm_info = build_rhythm_lookup_arrays(list(vocab.tokens), max_beat_positions=8)
    stats = {}
    for split_name, items in sequences.items():
        stats[split_name] = write_split_chunks(
            split_name, items, output_dir, context_size + 1, stride, True,
            vocab.to_id["<PAD>"], rhythm_info, max_beats_per_cycle,
            create_target_masks=True,
            source_kinds=source_kinds[split_name],
        )
    manifest = {"format_version": 2, "preparation_version": 1,
                "normalization_version": "skeleton_overlay_v1",
                "split_version": "per_file_contiguous_cycles_v1",
                "variant": variant, "vocab_hash": vocab.hash,
                "ordinary_dir": str(ordinary_dir),
                "decorated_dir": str(decorated) if decorated else None,
                "context_size": context_size, "stride": stride,
                "max_beats_per_cycle": max_beats_per_cycle,
                "source_hash": digest.hexdigest(), "files": split_files, "stats": stats}
    manifest["decorated_weight"] = decorated_weight if decorated else None
    (output_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                                       encoding="utf-8")
    return manifest
