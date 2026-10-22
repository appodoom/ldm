"""Cycle-aligned per-file dataset preparation and structured chunk arrays."""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import hashlib
from time import perf_counter
import numpy as np
from .derbake_ldm import RhythmSequence, parse_derbake
from .constants import MAX_SUBDIVISION, NORMAL_DATA, DATA_DECORATIONS, PREPARATION_VERSION, NORMALIZATION_RULE, SPLIT_RULE
from .schema_ldm import RhythmSchema
from .vocabulary import AtomicVocabulary, atomic_vocabulary
from .splits import split_cycle_ranges
from .validation import validate_rules
from .rhythm_ldm import (STRUCTURED_ROUTING_FIELDS, count_chunks, get_chunk_starts,
    build_rhythm_lookup_arrays, compute_rhythm_ids_for_sequence,
    compute_hierarchy_ids_for_sequence, build_structured_routing_for_window,
    build_target_ownership_masks_for_sequence)


def skeleton_atom_mask(document, start_cycle: int, end_cycle: int, variant: str) -> list[bool]:
    """Mark normalized HIT atoms sourced from line-four skeleton events."""
    lengths = document.rhythm.cycle_lengths
    first_beat = sum(lengths[:start_cycle])
    last_beat = sum(lengths[:end_cycle])
    event_slots = set()
    flat_beats = [beat for cycle in document.rhythm.cycles for beat in cycle]
    for event in document.skeleton_events:
        beat_index = int(event.time_in_beats)
        if first_beat <= beat_index < last_beat:
            beat = flat_beats[beat_index]
            slot = int((event.time_in_beats - beat_index) * beat.subdivision)
            event_slots.add((beat_index, min(beat.subdivision - 1, max(0, slot))))
    mask = []
    beat_index = first_beat
    for cycle in document.rhythm.cycles[start_cycle:end_cycle]:
        mask.append(False)  # <SOC>
        for beat in cycle:
            mask.append(False)  # <SOB>
            if variant == "with_subd":
                mask.append(False)
            for slot in range(beat.subdivision):
                mask.append((beat_index, slot) in event_slots)
            mask.append(False)  # <EOB>
            beat_index += 1
        mask.append(False)  # <EOC>
    return mask

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
    skeleton_targets: List[List[bool]] | None = None,
    source_file_ids: List[int] | None = None,
    dtype=np.int32,
) -> Dict[str, int]:
    """Write token, rhythm, ownership, and structured-routing arrays."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{split_name}.npy"
    file_path = out_dir / f"{split_name}_source_file.npy"
    beat_idx_path = out_dir / f"{split_name}_beat_idx.npy"
    beat_pos_path = out_dir / f"{split_name}_beat_pos.npy"
    score_mask_path = out_dir / f"{split_name}_score_mask.npy"
    hit_count_mask_path = out_dir / f"{split_name}_hit_count_mask.npy"
    skeleton_mask_path = out_dir / f"{split_name}_skeleton_mask.npy"
    routing_paths = {
        field: out_dir / f"{split_name}_{field}.npy"
        for field in STRUCTURED_ROUTING_FIELDS
    }

    parsed_files = []
    total_chunks = 0
    files_with_chunks = 0
    validate_rules({
        "source_file_ids": (source_file_ids is None or len(source_file_ids) == len(sequences),
                            "Each sequence needs a source file ID"),
        "skeleton_targets": (skeleton_targets is None or (len(skeleton_targets) == len(sequences) and
            all(len(mask) == len(ids) for mask, ids in zip(skeleton_targets, sequences))),
            "Skeleton target masks must match sequences"),
    })
    for ids in sequences:
        num_chunks = count_chunks(len(ids), window_len, stride, pad)
        parsed_files.append((ids, num_chunks))
        total_chunks += num_chunks
        files_with_chunks += int(num_chunks > 0)

    if total_chunks == 0:
        empty_int = np.empty((0, window_len), dtype=np.int32)
        empty_bool = np.empty((0, window_len), dtype=bool)
        np.save(out_path, np.empty((0, window_len), dtype=dtype))
        if source_file_ids is not None:
            np.save(file_path, np.empty(0, dtype=np.int32))
        np.save(beat_idx_path, empty_int)
        np.save(beat_pos_path, empty_int)
        for field, path in routing_paths.items():
            np.save(path, empty_bool if field.endswith("_mask") else empty_int)
        if create_target_masks:
            empty_mask = np.empty((0, window_len - 1), dtype=bool)
            np.save(score_mask_path, empty_mask)
            np.save(hit_count_mask_path, empty_mask)
            if skeleton_targets is not None:
                np.save(skeleton_mask_path, empty_mask)
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
    file_arr = (np.lib.format.open_memmap(file_path, mode="w+", dtype=np.int32,
                                         shape=(total_chunks,)) if source_file_ids is not None else None)
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
    skeleton_mask_arr = None
    if create_target_masks:
        mask_shape = (total_chunks, window_len - 1)
        score_mask_arr = np.lib.format.open_memmap(
            score_mask_path, mode="w+", dtype=bool, shape=mask_shape
        )
        hit_count_mask_arr = np.lib.format.open_memmap(
            hit_count_mask_path, mode="w+", dtype=bool, shape=mask_shape
        )
        if skeleton_targets is not None:
            skeleton_mask_arr = np.lib.format.open_memmap(
                skeleton_mask_path, mode="w+", dtype=bool, shape=mask_shape
            )

    row = 0
    unique_targets = 0
    unique_hit_count_events = 0
    maxima = {
        "max_tokens_per_beat": 0,
        "max_beat_slots_per_chunk": 0,
        "max_cycles_per_chunk": 0,
    }

    split_started = perf_counter()
    print(f"[prepare] {split_name}: writing {total_chunks:,} chunks", flush=True)
    for file_index, (ids, expected_chunks) in enumerate(parsed_files):
        if expected_chunks == 0:
            continue
        seq_len = len(ids)
        print(f"[prepare] {split_name}: sequence {file_index + 1}/{len(parsed_files)}, "
              f"{seq_len:,} tokens, {expected_chunks:,} chunks; building metadata", flush=True)
        source_mask = (np.asarray(skeleton_targets[file_index], dtype=bool)
                       if skeleton_targets is not None else None)
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
            if file_arr is not None:
                file_arr[row] = source_file_ids[file_index]
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
                if skeleton_mask_arr is not None:
                    skeleton_window = np.zeros(window_len - 1, dtype=bool)
                    segment = source_mask[start + 1:min(start + window_len, seq_len)]
                    skeleton_window[:len(segment)] = segment
                    skeleton_mask_arr[row] = file_score_masks[file_row] & skeleton_window
            row += 1
            if row % 500 == 0 or row == total_chunks:
                print(f"[prepare] {split_name}: {row:,}/{total_chunks:,} chunks "
                      f"({100 * row / total_chunks:.1f}%), {perf_counter() - split_started:.1f}s", flush=True)

    if row != total_chunks:
        raise RuntimeError(
            f"{split_name}: wrote {row} chunks but expected {total_chunks} chunks"
        )
    arr.flush()
    if file_arr is not None:
        file_arr.flush()
    beat_idx_arr.flush()
    beat_pos_arr.flush()
    for routing_array in routing_arrays.values():
        routing_array.flush()
    if create_target_masks:
        score_mask_arr.flush()
        hit_count_mask_arr.flush()
        if skeleton_mask_arr is not None:
            skeleton_mask_arr.flush()

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
    source_dir: str | Path, output_dir: str | Path, *, variant: str,
    context_size: int, max_beats_per_cycle: int,
    decoration: str = NORMAL_DATA, stride: int | None = None,
) -> dict:
    """Prepare one independent source, splitting complete cycles within each file."""
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    files = sorted(source_dir.glob("*.derbake"))
    stride = max(1, context_size // 2) if stride is None else stride
    validate_rules({
        "source_dir": (bool(files), f"No .derbake files in {source_dir}"),
        "decoration": (decoration in DATA_DECORATIONS, f"Unknown decoration mode: {decoration}"),
        "context_size": (context_size >= 2, "must be at least 2"),
        "stride": (1 <= stride <= context_size, "must be between 1 and context_size"),
    })
    schema = RhythmSchema(variant, max_beats_per_cycle)
    vocab = AtomicVocabulary(atomic_vocabulary(variant))
    sequences: dict[str, list[list[int]]] = {"train": [], "val": [], "test": []}
    skeleton_masks: dict[str, list[list[bool]]] = {"train": [], "val": [], "test": []}
    file_ids: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    split_files = []
    digest = hashlib.sha256()
    preparation_started = perf_counter()
    for source_index, source in enumerate(files):
        file_started = perf_counter()
        print(f"[prepare] {variant}: parsing file {source_index + 1}/{len(files)} "
              f"{source.name} ({source.stat().st_size / 2**20:.1f} MiB)", flush=True)
        document = parse_derbake(source, max_beats_per_cycle=max_beats_per_cycle)
        digest.update(source.name.encode("utf-8"))
        digest.update(hashlib.sha256(source.read_bytes()).digest())
        ranges = split_cycle_ranges(len(document.rhythm.cycles))
        split_files.append({"file": source.name, "cycles": len(document.rhythm.cycles),
                            "ranges": {key: list(value) for key, value in ranges.items()}})
        for split_name, (start, end) in ranges.items():
            piece = RhythmSequence(document.rhythm.cycles[start:end])
            atoms = schema.encode(piece)
            mask = skeleton_atom_mask(document, start, end, variant)
            if len(mask) != len(atoms):
                raise RuntimeError("Skeleton provenance differs from encoded atoms")
            sequences[split_name].append(vocab.encode(atoms))
            skeleton_masks[split_name].append(mask)
            file_ids[split_name].append(source_index)
        print(f"[prepare] {source.name}: {len(document.rhythm.cycles):,} cycles encoded "
              f"in {perf_counter() - file_started:.1f}s", flush=True)
        del document, piece, atoms, mask
    output_dir.mkdir(parents=True, exist_ok=True)
    rhythm_info = build_rhythm_lookup_arrays(list(vocab.tokens), max_beat_positions=MAX_SUBDIVISION)
    stats = {}
    for split_name, items in sequences.items():
        stats[split_name] = write_split_chunks(
            split_name, items, output_dir, context_size + 1, stride, True,
            vocab.to_id["<PAD>"], rhythm_info, max_beats_per_cycle,
            create_target_masks=True, skeleton_targets=skeleton_masks[split_name],
            source_file_ids=file_ids[split_name],
        )
    manifest = {"preparation_version": PREPARATION_VERSION,
                "normalization_rule": NORMALIZATION_RULE,
                "split_rule": SPLIT_RULE,
                "variant": variant, "decoration": decoration, "vocab_hash": vocab.hash,
                "source_dir": str(source_dir),
                "context_size": context_size, "stride": stride,
                "max_beats_per_cycle": max_beats_per_cycle,
                "source_hash": digest.hexdigest(), "files": split_files, "stats": stats}
    from .dataset_integrity import seal_dataset
    print("[prepare] Verifying and hashing prepared arrays...", flush=True)
    seal_dataset(output_dir, manifest)
    print(f"[prepare] {variant} complete in {perf_counter() - preparation_started:.1f}s", flush=True)
    return manifest
