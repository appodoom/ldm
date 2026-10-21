"""Autoregressive sampling with notebook-equivalent global/structured KV caches."""
from __future__ import annotations
from typing import Optional, List
import numpy as np
import torch
from .model_core import GPT
from .rhythm_ldm import (build_rhythm_lookup_arrays, RhythmStateTracker,
    compute_hierarchy_ids_for_sequence, build_structured_routing_for_window,
    StructuredGenerationTracker, structured_step_to_tensors)
from .schema_ldm import RhythmSchema
from .cycle_collector import CycleCollector
from .validation import validate_rules

def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or k >= logits.numel():
        return logits
    threshold = torch.topk(logits, k).values[-1]
    return logits.masked_fill(logits < threshold, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    if p <= 0.0 or p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    remove = cumulative > p
    remove[1:] = remove[:-1].clone()
    remove[0] = False
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))

    filtered = torch.full_like(logits, float("-inf"))
    filtered[sorted_indices] = sorted_logits
    return filtered


def sample_next_id(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    generator: Optional[torch.Generator],
) -> int:
    if temperature <= 0:
        return int(torch.argmax(logits).item())

    filtered = top_k_filter(logits / temperature, top_k)
    filtered = top_p_filter(filtered, top_p)
    probabilities = torch.softmax(filtered, dim=-1)

    if not torch.isfinite(probabilities).all() or probabilities.sum() <= 0:
        return int(torch.argmax(logits).item())

    return int(
        torch.multinomial(
            probabilities,
            num_samples=1,
            generator=generator,
        ).item()
    )


@torch.inference_mode()
def generate_continuation_ids(
    model: GPT,
    vocab: List[str],
    prompt_ids: List[int],
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    seed: Optional[int],
    num_cycles: int,
    schema: RhythmSchema,
    cache_refresh_tokens: int = 256,
) -> tuple[List[int], dict]:
    """Generate with one-token updates for both global and structured caches."""
    validate_rules({
        "max_new_tokens": (max_new_tokens >= 1, "must be positive"),
        "prompt_ids": (bool(prompt_ids), "cannot be empty"),
        "cache_refresh_tokens": (cache_refresh_tokens >= 1, "must be positive"),
    })

    device = next(model.parameters()).device
    model.eval()
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    all_ids = list(prompt_ids)
    pad_id = vocab.index("<PAD>")
    # Keep sampled tokens unchanged for the model and collector. Orphan or
    # excess beat markers are neutral only in metadata, including cache rebuilds.
    hierarchy_ids = list(prompt_ids)
    collector = CycleCollector(schema, num_cycles)
    rhythm_info = build_rhythm_lookup_arrays(
        vocab, max_beat_positions=model.max_beat_positions
    )
    rhythm_tracker = RhythmStateTracker(
        rhythm_info=rhythm_info,
        max_beats_per_cycle=model.max_beats_per_cycle,
    )
    prompt_beat_idx, prompt_beat_pos = rhythm_tracker.extend(all_ids)
    all_beat_idx = prompt_beat_idx.tolist()
    all_beat_pos = prompt_beat_pos.tolist()

    def build_prefill_context(context_start: int):
        context_ids = all_ids[context_start:]
        context_beat_idx = all_beat_idx[context_start:]
        context_beat_pos = all_beat_pos[context_start:]
        if not (
            len(context_ids) == len(context_beat_idx) == len(context_beat_pos)
        ):
            raise ValueError("Token and rhythm context lengths must match")
        token_tensor = torch.tensor(
            context_ids, dtype=torch.long, device=device
        ).unsqueeze(0)
        beat_idx_tensor = torch.tensor(
            context_beat_idx, dtype=torch.long, device=device
        ).unsqueeze(0)
        beat_pos_tensor = torch.tensor(
            context_beat_pos, dtype=torch.long, device=device
        ).unsqueeze(0)

        routing_tensor = None
        structured_tracker = None
        if model.has_structured_layers:
            hierarchy = compute_hierarchy_ids_for_sequence(hierarchy_ids, rhythm_info)
            routing, _ = build_structured_routing_for_window(
                hierarchy=hierarchy,
                start=context_start,
                real_end=len(all_ids),
                window_len=len(context_ids),
            )
            routing_tensor = {
                field: torch.from_numpy(values).unsqueeze(0).to(device)
                for field, values in routing.items()
            }

            # Replay from the visible partial cycle's true <SOC> (if any). This
            # reconstructs exact beat/local coordinates without adding hidden
            # pre-context keys to either attention cache.
            replay_start = context_start
            if context_start < len(all_ids):
                active_cycle_uid = int(hierarchy["cycle_uid"][context_start])
                if active_cycle_uid >= 0:
                    matching_starts = np.flatnonzero(
                        hierarchy["cycle_start"]
                        & (hierarchy["cycle_uid"] == active_cycle_uid)
                    )
                    if matching_starts.size:
                        replay_start = int(matching_starts[0])
            structured_tracker = StructuredGenerationTracker(rhythm_info)
            for token_id in hierarchy_ids[replay_start:context_start]:
                structured_tracker.append(token_id, visible_to_cache=False)
            for token_id in hierarchy_ids[context_start:]:
                structured_tracker.append(token_id, visible_to_cache=True)

        return (
            token_tensor,
            beat_idx_tensor,
            beat_pos_tensor,
            routing_tensor,
            structured_tracker,
        )

    context_start = max(0, len(all_ids) - model.context_size)
    (
        x,
        beat_idx_tensor,
        beat_pos_tensor,
        routing_tensor,
        structured_tracker,
    ) = build_prefill_context(context_start)
    logits, kv_caches = model.prefill_with_kv_cache(
        x,
        beat_idx_ids=beat_idx_tensor,
        beat_pos_ids=beat_pos_tensor,
        structured_routing=routing_tensor,
    )
    next_token_logits = logits[0, -1]

    for generation_step in range(max_new_tokens):
        next_id = sample_next_id(
            logits=next_token_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generator=generator,
        )
        hierarchy_id = next_id
        if rhythm_info["is_sob"][next_id] and (
            not rhythm_tracker.in_cycle or
            (rhythm_tracker.started and rhythm_tracker.beat + 1 >= model.max_beats_per_cycle)
        ):
            hierarchy_id = pad_id
        hierarchy_ids.append(hierarchy_id)
        all_ids.append(next_id)
        done = collector.append(next_id, vocab[next_id])
        next_beat_idx, next_beat_pos = rhythm_tracker.append(hierarchy_id)
        all_beat_idx.append(next_beat_idx)
        all_beat_pos.append(next_beat_pos)
        structured_step = (
            structured_step_to_tensors(structured_tracker.append(hierarchy_id), device)
            if structured_tracker is not None
            else None
        )

        if done:
            break
        if generation_step + 1 >= max_new_tokens:
            break

        global_cache = next(
            wrapped["state"] for wrapped in kv_caches if wrapped["kind"] == "global"
        )
        cache_len = int(global_cache[2])
        if cache_len < model.context_size:
            decode_logits, kv_caches = model.decode_one_with_kv_cache(
                torch.tensor([[next_id]], dtype=torch.long, device=device),
                kv_caches=kv_caches,
                beat_idx_ids=torch.tensor(
                    [[next_beat_idx]], dtype=torch.long, device=device
                ),
                beat_pos_ids=torch.tensor(
                    [[next_beat_pos]], dtype=torch.long, device=device
                ),
                structured_step=structured_step,
            )
            next_token_logits = decode_logits[0, -1]
        else:
            # Rebuild global and structured caches together. Replaying metadata
            # is CPU-only; the model still performs one prefill, not per-token
            # complete-history recomputation.
            remaining = max_new_tokens - generation_step - 1
            reserve = min(
                cache_refresh_tokens,
                max(1, remaining),
                model.context_size - 1,
            )
            refresh_context_len = model.context_size - reserve
            context_start = max(0, len(all_ids) - refresh_context_len)
            (
                x,
                beat_idx_tensor,
                beat_pos_tensor,
                routing_tensor,
                structured_tracker,
            ) = build_prefill_context(context_start)
            del kv_caches
            logits, kv_caches = model.prefill_with_kv_cache(
                x,
                beat_idx_ids=beat_idx_tensor,
                beat_pos_ids=beat_pos_tensor,
                structured_routing=routing_tensor,
            )
            next_token_logits = logits[0, -1]

    return collector.finish()
