"""Rhythm metadata and structured routing; extracted from the research notebook."""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union
import re
import numpy as np
try:
    import torch
except ImportError:  # Dataset preparation only needs NumPy.
    torch = None


_HIT = re.compile(r"HIT_[A-Za-z0-9]+(?:_decorated)?\Z")
_SUBD = re.compile(r"SUBD_([1-9][0-9]*)\Z")


def count_hit_atoms(token: str) -> int:
    """Atomic vocabulary: one HIT token occupies exactly one position."""
    return int(_HIT.fullmatch(token) is not None)


def parse_subdivision_token(token: str) -> Optional[int]:
    match = _SUBD.fullmatch(token)
    return int(match.group(1)) if match else None

def get_chunk_starts(
    seq_len: int,
    window_len: int,
    stride: int,
    pad: bool,
) -> List[int]:
    """
    Return every chunk start, including a final right-aligned tail chunk.
    """
    if window_len <= 0:
        raise ValueError(f"window_len must be positive. Got {window_len}")
    if stride <= 0:
        raise ValueError(f"stride must be positive. Got {stride}")
    if seq_len <= 0:
        return []
    if seq_len < window_len:
        return [0] if pad else []
    last_start = seq_len - window_len
    starts = list(range(0, last_start + 1, stride))
    if starts[-1] != last_start:
        starts.append(last_start)

    return starts


def iter_chunk_specs(
    seq_len: int,
    window_len: int,
    stride: int,
    pad: bool = True,
):
    """Yield chunk bounds and padding metadata from the canonical chunk layout."""
    for start in get_chunk_starts(seq_len, window_len, stride, pad):
        end = min(start + window_len, seq_len)
        pad_count = max(0, window_len - (end - start))
        yield start, end, pad_count, pad_count > 0


def count_chunks(
    seq_len: int,
    window_len: int,
    stride: int,
    pad: bool,
) -> int:
    return len(get_chunk_starts(seq_len, window_len, stride, pad))


def build_rhythm_lookup_arrays(
    vocab: List[str],
    max_beat_positions: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    vocab_size = len(vocab)

    is_sob = np.zeros(vocab_size, dtype=bool)
    is_eob = np.zeros(vocab_size, dtype=bool)
    is_soc = np.zeros(vocab_size, dtype=bool)
    is_eoc = np.zeros(vocab_size, dtype=bool)
    is_hit = np.zeros(vocab_size, dtype=bool)
    hit_atom_counts = np.zeros(vocab_size, dtype=np.int32)
    is_subd = np.zeros(vocab_size, dtype=bool)
    subd_sizes = np.zeros(vocab_size, dtype=np.int32)
    is_delim = np.zeros(vocab_size, dtype=bool)

    max_subd = 1

    for idx, tok in enumerate(vocab):
        is_sob[idx] = tok == "<SOB>"
        is_eob[idx] = tok == "<EOB>"
        is_soc[idx] = tok == "<SOC>"
        is_eoc[idx] = tok == "<EOC>"

        hit_atom_counts[idx] = count_hit_atoms(tok)
        is_hit[idx] = hit_atom_counts[idx] > 0

        subd_size = parse_subdivision_token(tok)
        if subd_size is not None:
            is_subd[idx] = True
            subd_sizes[idx] = subd_size
            max_subd = max(max_subd, subd_size)

        if tok in {"<SOC>", "<EOC>", "<SOB>", "<EOB>"} or (
            tok.startswith("<") and tok.endswith(">")
        ):
            is_delim[idx] = True

    if max_beat_positions is None:
        max_beat_positions = max_subd
    max_beat_positions = max(1, int(max_beat_positions))

    return {
        "is_sob": is_sob,
        "is_eob": is_eob,
        "is_soc": is_soc,
        "is_eoc": is_eoc,
        "is_hit": is_hit,
        "hit_atom_counts": hit_atom_counts,
        "is_subd": is_subd,
        "subd_sizes": subd_sizes,
        "is_delim": is_delim,
        "max_beat_positions": max_beat_positions,
    }


class RhythmStateTracker:
    """Incrementally assign rhythm IDs in complete-sequence coordinates."""

    def __init__(
        self,
        rhythm_info: Dict[str, np.ndarray],
        max_beats_per_cycle: int,
    ):
        if max_beats_per_cycle <= 0:
            raise ValueError("max_beats_per_cycle must be positive")

        self.rhythm_info = rhythm_info
        self.max_beats_per_cycle = int(max_beats_per_cycle)
        self.max_beat_positions = max(
            1, int(rhythm_info["max_beat_positions"])
        )
        self.in_cycle = False
        self.beat = 0
        self.started = False
        self.cur = 0
        self.in_beat = False

    def append(self, token_id: int) -> Tuple[int, int]:
        """Consume one token and return its beat-index and beat-position IDs."""
        token_id = int(token_id)
        vocab_size = len(self.rhythm_info["is_sob"])
        if token_id < 0 or token_id >= vocab_size:
            raise IndexError(
                f"token_id {token_id} is outside vocabulary size {vocab_size}"
            )

        sob_t = bool(self.rhythm_info["is_sob"][token_id])
        soc_t = bool(self.rhythm_info["is_soc"][token_id])
        eoc_t = bool(self.rhythm_info["is_eoc"][token_id])
        hit_count = int(self.rhythm_info["hit_atom_counts"][token_id])
        subd_t = bool(self.rhythm_info["is_subd"][token_id])
        delim_t = bool(self.rhythm_info["is_delim"][token_id])

        start_cycle = soc_t or ((not self.in_cycle) and sob_t)
        if start_cycle:
            self.in_cycle = True
        if eoc_t:
            self.in_cycle = False
        if start_cycle:
            self.beat = 0
            self.started = False

        inc = sob_t and self.in_cycle
        if inc and self.started:
            self.beat += 1
        if inc:
            self.started = True

        beat_idx = self.beat if self.in_cycle else 0
        if beat_idx >= self.max_beats_per_cycle:
            raise ValueError(
                f"Observed beat index {beat_idx}, but metadata allows only "
                f"0..{self.max_beats_per_cycle - 1}"
            )

        reset = sob_t or subd_t
        end_or_boundary = delim_t and (not sob_t)
        if reset:
            self.in_beat = True
        if end_or_boundary:
            self.in_beat = False
        if reset or end_or_boundary:
            self.cur = 0

        beat_pos = self.cur if self.in_beat else 0
        if self.in_beat:
            self.cur += hit_count

        beat_pos = min(beat_pos, self.max_beat_positions - 1)
        return int(beat_idx), int(beat_pos)

    def extend(self, sequence_ids: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Consume a sequence once and return aligned rhythm arrays."""
        beat_idx_ids = np.empty(len(sequence_ids), dtype=np.int32)
        beat_pos_ids = np.empty(len(sequence_ids), dtype=np.int32)
        for position, token_id in enumerate(sequence_ids):
            beat_idx, beat_pos = self.append(token_id)
            beat_idx_ids[position] = beat_idx
            beat_pos_ids[position] = beat_pos
        return beat_idx_ids, beat_pos_ids


def compute_rhythm_ids_for_sequence(
    sequence_ids: List[int],
    rhythm_info: Dict[str, np.ndarray],
    max_beats_per_cycle: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute rhythm IDs once in complete source-sequence coordinates.

    Chunking or context slicing must happen after this pass so the same
    original token always keeps the same rhythm labels.
    """
    tracker = RhythmStateTracker(
        rhythm_info=rhythm_info,
        max_beats_per_cycle=max_beats_per_cycle,
    )
    return tracker.extend(sequence_ids)


def build_target_ownership_masks_for_sequence(
    sequence_ids: List[int],
    chunk_starts: List[int],
    window_len: int,
    subd_sizes: np.ndarray,
    hit_atom_counts: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Assign each original target and complete SUBD event to one chunk.

    Chunks may overlap so the model keeps their full context. The masks decide
    which occurrence contributes to training loss or evaluation metrics.
    """
    sequence_ids = np.asarray(sequence_ids, dtype=np.int64)
    seq_len = int(sequence_ids.size)
    context_len = window_len - 1
    if context_len <= 0:
        raise ValueError("window_len must be at least 2 for next-token evaluation")

    score_masks = np.zeros((len(chunk_starts), context_len), dtype=bool)
    hit_count_masks = np.zeros_like(score_masks)

    scored_targets = np.zeros(seq_len, dtype=bool)
    for chunk_row, start in enumerate(chunk_starts):
        target_positions = np.arange(
            start + 1,
            min(start + window_len, seq_len),
            dtype=np.int64,
        )
        if target_positions.size == 0:
            continue
        take = ~scored_targets[target_positions]
        local_target_positions = target_positions[take] - (start + 1)
        score_masks[chunk_row, local_target_positions] = True
        scored_targets[target_positions[take]] = True

    missing_targets = np.flatnonzero(~scored_targets[1:]) + 1
    if missing_targets.size:
        raise RuntimeError(
            f"Chunk layout failed to cover {missing_targets.size} next-token targets"
        )

    complete_hit_events = 0
    for source_position, token_id in enumerate(sequence_ids):
        expected_hits = int(subd_sizes[token_id])
        if expected_hits <= 0:
            continue
        hit_total = 0
        event_end = source_position + 1
        while event_end < seq_len and hit_total < expected_hits:
            hit_count = int(hit_atom_counts[sequence_ids[event_end]])
            if hit_count == 0:
                break
            hit_total += hit_count
            event_end += 1
        if hit_total != expected_hits or event_end >= seq_len or hit_atom_counts[sequence_ids[event_end]]:
            raise ValueError(f"Invalid or incomplete SUBD event at token position {source_position}")
        complete_hit_events += 1
        hit_token_count = event_end - source_position - 1
        assigned = False
        for chunk_row, start in enumerate(chunk_starts):
            local_position = source_position - start
            if local_position < 0 or local_position >= context_len:
                continue
            if local_position + hit_token_count >= context_len:
                continue
            hit_count_masks[chunk_row, local_position] = True
            assigned = True
            break
        if not assigned:
            raise RuntimeError(
                "A complete SUBD event does not fit in any evaluation chunk. "
                "Increase the window size or reduce the maximum subdivision."
            )

    stats = {
        "unique_targets": int(score_masks.sum()),
        "unique_hit_count_events": int(hit_count_masks.sum()),
        "complete_hit_count_events": complete_hit_events,
    }
    if stats["unique_targets"] != max(0, seq_len - 1):
        raise RuntimeError("Evaluation target-mask coverage is inconsistent")
    if stats["unique_hit_count_events"] != complete_hit_events:
        raise RuntimeError("Evaluation hit-event mask coverage is inconsistent")

    return score_masks, hit_count_masks, stats

STRUCTURED_ROUTING_FIELDS = (
    "beat_slot",
    "cycle_slot",
    "previous_completed_beat_slot",
    "previous_completed_cycle_slot",
    "complete_eob_mask",
    "complete_eoc_mask",
    "local_beat_position",
)


def compute_hierarchy_ids_for_sequence(
    sequence_ids: List[int],
    rhythm_info: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """Build stable beat/cycle identities before any overlapping chunking.

    IDs and local beat positions are computed in complete-file coordinates so
    an overlapping token receives identical hierarchy metadata in every chunk.
    Local routing positions count model tokens; beat-position IDs count HIT atoms.
    """
    length = len(sequence_ids)
    beat_uid = np.full(length, -1, dtype=np.int32)
    cycle_uid = np.full(length, -1, dtype=np.int32)
    local_beat_position = np.full(length, -1, dtype=np.int32)
    beat_start = np.zeros(length, dtype=bool)
    beat_end = np.zeros(length, dtype=bool)
    cycle_start = np.zeros(length, dtype=bool)
    cycle_end = np.zeros(length, dtype=bool)

    next_beat_uid = 0
    next_cycle_uid = 0
    active_beat = -1
    active_cycle = -1
    beat_position = 0

    for position, raw_token_id in enumerate(sequence_ids):
        token_id = int(raw_token_id)
        is_soc = bool(rhythm_info["is_soc"][token_id])
        is_eoc = bool(rhythm_info["is_eoc"][token_id])
        is_sob = bool(rhythm_info["is_sob"][token_id])
        is_eob = bool(rhythm_info["is_eob"][token_id])

        if is_soc:
            active_cycle = next_cycle_uid
            next_cycle_uid += 1
            cycle_start[position] = True
            active_beat = -1
        elif is_sob and active_cycle < 0:
            # Tolerate legacy sequences that begin directly with <SOB>.
            active_cycle = next_cycle_uid
            next_cycle_uid += 1
            cycle_start[position] = True

        if is_sob:
            active_beat = next_beat_uid
            next_beat_uid += 1
            beat_position = 0
            beat_start[position] = True

        if active_cycle >= 0:
            cycle_uid[position] = active_cycle
        if active_beat >= 0:
            beat_uid[position] = active_beat
            local_beat_position[position] = beat_position
            beat_position += 1

        if is_eob and active_beat >= 0:
            beat_end[position] = True
            active_beat = -1
            beat_position = 0
        if is_eoc and active_cycle >= 0:
            cycle_end[position] = True
            active_beat = -1
            active_cycle = -1
            beat_position = 0

    boundary_positions = {
        name: {int(uids[pos]): int(pos) for pos in np.flatnonzero(mask)}
        for name, uids, mask in (
            ("beat_start", beat_uid, beat_start),
            ("beat_end", beat_uid, beat_end),
            ("cycle_start", cycle_uid, cycle_start),
            ("cycle_end", cycle_uid, cycle_end),
        )
    }
    return {
        "boundary_positions": boundary_positions,
        "beat_uid": beat_uid,
        "cycle_uid": cycle_uid,
        "local_beat_position": local_beat_position,
        "beat_start": beat_start,
        "beat_end": beat_end,
        "cycle_start": cycle_start,
        "cycle_end": cycle_end,
    }


def build_structured_routing_for_window(
    hierarchy: Dict[str, Any],
    start: int,
    real_end: int,
    window_len: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """Create one padded, causal routing record for a source-file window.

    Complete summary masks require both boundaries to be visible. The previous
    memory slots are assigned before processing the current boundary, so an
    <EOB>/<EOC> summary first becomes visible to the following token.
    """
    if not (0 <= start <= real_end):
        raise ValueError("Invalid structured-routing source interval")
    real_len = real_end - start
    if real_len > window_len:
        raise ValueError("Structured-routing interval exceeds window_len")

    result = {
        "beat_slot": np.full(window_len, -1, dtype=np.int32),
        "cycle_slot": np.full(window_len, -1, dtype=np.int32),
        "previous_completed_beat_slot": np.full(window_len, -1, dtype=np.int32),
        "previous_completed_cycle_slot": np.full(window_len, -1, dtype=np.int32),
        "complete_eob_mask": np.zeros(window_len, dtype=bool),
        "complete_eoc_mask": np.zeros(window_len, dtype=bool),
        "local_beat_position": np.full(window_len, -1, dtype=np.int32),
    }
    if real_len == 0:
        return result, {
            "max_tokens_per_beat": 0,
            "max_beat_slots_per_chunk": 0,
            "max_cycles_per_chunk": 0,
        }

    beat_ids = hierarchy["beat_uid"][start:real_end]
    cycle_ids = hierarchy["cycle_uid"][start:real_end]
    local_positions = hierarchy["local_beat_position"][start:real_end]

    visible_beats = [int(uid) for uid in np.unique(beat_ids) if uid >= 0]
    visible_cycles = [int(uid) for uid in np.unique(cycle_ids) if uid >= 0]
    beat_to_slot = {uid: slot for slot, uid in enumerate(visible_beats)}
    cycle_to_slot = {uid: slot for slot, uid in enumerate(visible_cycles)}

    for local_index in range(real_len):
        beat_id = int(beat_ids[local_index])
        cycle_id = int(cycle_ids[local_index])
        if beat_id >= 0:
            result["beat_slot"][local_index] = beat_to_slot[beat_id]
            result["local_beat_position"][local_index] = int(
                local_positions[local_index]
            )
        if cycle_id >= 0:
            result["cycle_slot"][local_index] = cycle_to_slot[cycle_id]

    boundaries = hierarchy["boundary_positions"]

    for beat_id, slot in beat_to_slot.items():
        beat_start = boundaries["beat_start"].get(beat_id, -1)
        beat_end = boundaries["beat_end"].get(beat_id, -1)
        if start <= beat_start <= beat_end < real_end:
            result["complete_eob_mask"][beat_end - start] = True
    for cycle_id, slot in cycle_to_slot.items():
        cycle_start = boundaries["cycle_start"].get(cycle_id, -1)
        cycle_end = boundaries["cycle_end"].get(cycle_id, -1)
        if start <= cycle_start <= cycle_end < real_end:
            result["complete_eoc_mask"][cycle_end - start] = True

    latest_beat_slot = -1
    latest_beat_cycle_slot = -1
    latest_cycle_slot = -1
    for local_index in range(real_len):
        cycle_slot = int(result["cycle_slot"][local_index])
        source_position = start + local_index
        if hierarchy["cycle_start"][source_position]:
            latest_beat_slot = -1
            latest_beat_cycle_slot = cycle_slot

        result["previous_completed_beat_slot"][local_index] = (
            latest_beat_slot if latest_beat_cycle_slot == cycle_slot else -1
        )
        result["previous_completed_cycle_slot"][local_index] = latest_cycle_slot

        if result["complete_eob_mask"][local_index]:
            latest_beat_slot = int(result["beat_slot"][local_index])
            latest_beat_cycle_slot = cycle_slot
        if result["complete_eoc_mask"][local_index]:
            latest_cycle_slot = cycle_slot
            latest_beat_slot = -1
            latest_beat_cycle_slot = -1

    max_local_position = int(local_positions.max(initial=-1))
    return result, {
        "max_tokens_per_beat": max_local_position + 1,
        "max_beat_slots_per_chunk": len(visible_beats),
        "max_cycles_per_chunk": len(visible_cycles),
    }

class StructuredGenerationTracker:
    """CPU-only hierarchy state for one-token structured cached decoding.

    This state is intentionally maintained outside CUDA: sampled token IDs are
    already CPU scalars, so no GPU synchronization is introduced by boundaries.
    """

    def __init__(self, rhythm_info: Dict[str, np.ndarray]):
        self.rhythm_info = rhythm_info
        self.in_cycle = False
        self.in_beat = False
        self.cycle_ordinal = -1
        self.next_cycle_ordinal = 0
        self.beat_index_in_cycle = 0
        self.next_beat_index = 0
        self.local_beat_position = 0
        self.current_beat_publishable = False
        self.current_cycle_publishable = False

    def append(
        self, token_id: int, visible_to_cache: bool = True
    ) -> Dict[str, Union[bool, int]]:
        token_id = int(token_id)
        starts_cycle = bool(self.rhythm_info["is_soc"][token_id])
        starts_beat = bool(self.rhythm_info["is_sob"][token_id])
        ends_beat = bool(self.rhythm_info["is_eob"][token_id])
        ends_cycle = bool(self.rhythm_info["is_eoc"][token_id])

        if starts_cycle or (starts_beat and not self.in_cycle):
            starts_cycle = True
            self.in_cycle = True
            self.in_beat = False
            self.cycle_ordinal = self.next_cycle_ordinal
            self.next_cycle_ordinal += 1
            self.next_beat_index = 0
            self.beat_index_in_cycle = 0
            self.local_beat_position = 0
            self.current_cycle_publishable = bool(visible_to_cache)
            self.current_beat_publishable = False

        if starts_beat:
            self.in_beat = True
            self.beat_index_in_cycle = self.next_beat_index
            self.next_beat_index += 1
            self.local_beat_position = 0
            self.current_beat_publishable = bool(visible_to_cache)

        token_in_beat = self.in_beat
        step = {
            "starts_beat": starts_beat,
            "ends_beat": (
                ends_beat and token_in_beat and self.current_beat_publishable
            ),
            "closes_beat": ends_beat and token_in_beat,
            "starts_cycle": starts_cycle,
            "ends_cycle": (
                ends_cycle and self.in_cycle and self.current_cycle_publishable
            ),
            "closes_cycle": ends_cycle and self.in_cycle,
            "in_beat": token_in_beat,
            "local_beat_position": self.local_beat_position if token_in_beat else 0,
            "beat_index_in_cycle": self.beat_index_in_cycle,
            "cycle_ordinal": max(0, self.cycle_ordinal),
        }

        if token_in_beat:
            self.local_beat_position += 1
        if step["closes_beat"]:
            self.in_beat = False
            self.local_beat_position = 0
            self.current_beat_publishable = False
        if step["closes_cycle"]:
            self.in_cycle = False
            self.in_beat = False
            self.local_beat_position = 0
            self.current_beat_publishable = False
            self.current_cycle_publishable = False
        return step


def structured_step_to_tensors(
    step: Dict[str, Union[bool, int]],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    if torch is None:
        raise ImportError("PyTorch is required for generation")
    return {
        key: torch.tensor(
            [[value]],
            dtype=torch.bool if isinstance(value, bool) else torch.long,
            device=device,
        )
        for key, value in step.items()
    }
