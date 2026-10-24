"""Attention correctness checks and optional CUDA benchmarks for atomic LDM models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .derbake_ldm import Beat, RhythmSequence
from .schema_ldm import RhythmSchema
from .validation import validate_rules
from .rhythm_ldm import (build_rhythm_lookup_arrays, compute_hierarchy_ids_for_sequence,
                         compute_rhythm_ids_for_sequence, build_structured_routing_for_window,
                         StructuredGenerationTracker, structured_step_to_tensors)


def _fixture(core, vocab, variant):
    schema = RhythmSchema(variant, core.max_beats_per_cycle)
    subdivision = 2 if core.max_beat_positions >= 2 and (not core.has_structured_layers or
                      core.max_tokens_per_beat >= (5 if variant == 'with_subd' else 4)) else 1
    for cycles, beats in ((2, 2), (2, 1), (1, 1)):
        if beats > core.max_beats_per_cycle:
            continue
        if core.has_structured_layers and (cycles > core.max_cycles_per_chunk or
                                           cycles * beats > core.max_beat_slots_per_chunk):
            continue
        rhythm = RhythmSequence(tuple(tuple(Beat(subdivision, tuple('HIT_D' if (i + j) % 2 == 0 else 'HIT_OTA'
                                            for j in range(subdivision)))
                                            for i in range(beats)) for _ in range(cycles)))
        ids = [vocab.index(atom) for atom in schema.encode(rhythm)]
        if len(ids) <= core.context_size:
            return ids, {'cycles': cycles, 'beats_per_cycle': beats, 'subdivision': subdivision}
    return ids[:core.context_size], {'cycles': 0, 'beats_per_cycle': 1, 'partial_cycle': True}


def _window(core, vocab, ids, start=0, end=None):
    end = len(ids) if end is None else end
    device = next(core.parameters()).device
    lookup = build_rhythm_lookup_arrays(vocab, core.max_beat_positions)
    beat, position = compute_rhythm_ids_for_sequence(ids, lookup, core.max_beats_per_cycle)
    x = torch.tensor([ids[start:end]], dtype=torch.long, device=device)
    beat = torch.as_tensor(beat[start:end].copy(), device=device).long().unsqueeze(0)
    position = torch.as_tensor(position[start:end].copy(), device=device).long().unsqueeze(0)
    routing, tracker = None, None
    if core.has_structured_layers:
        hierarchy = compute_hierarchy_ids_for_sequence(ids, lookup)
        arrays, _ = build_structured_routing_for_window(hierarchy, start, end, end - start)
        routing = {name: torch.from_numpy(array).unsqueeze(0).to(device) for name, array in arrays.items()}
        replay = start
        uid = int(hierarchy['cycle_uid'][start])
        if uid >= 0:
            starts = np.flatnonzero(hierarchy['cycle_start'] & (hierarchy['cycle_uid'] == uid))
            if starts.size:
                replay = int(starts[0])
        tracker = StructuredGenerationTracker(lookup)
        for token in ids[replay:start]:
            tracker.append(token, visible_to_cache=False)
    return x, beat, position, routing, tracker


def padding_utilization(core, routing) -> dict:
    if routing is None:
        return {}
    batch = routing['beat_slot'].shape[0]
    return {'inside_beat': float((routing['beat_slot'] >= 0).sum()) /
                max(1, batch * core.max_beat_slots_per_chunk * core.max_tokens_per_beat),
            'beat_memory': float(routing['complete_eob_mask'].sum()) /
                max(1, batch * core.max_cycles_per_chunk * core.max_beats_per_cycle),
            'cycle_memory': float(routing['complete_eoc_mask'].sum()) /
                max(1, batch * core.max_cycles_per_chunk)}


@torch.no_grad()
def _cached_rollout(core, vocab, ids, start=0, prefix=1, check=True):
    x, beat, position, routing, tracker = _window(core, vocab, ids, start)
    reference = core(x, beat, position, routing) if check else None
    _, _, _, prefix_routing, _ = _window(core, vocab, ids, start, start + prefix)
    logits, caches = core.prefill_with_kv_cache(x[:, :prefix], beat[:, :prefix],
                                               position[:, :prefix], prefix_routing)
    if check:
        torch.testing.assert_close(logits, reference[:, :prefix], atol=2e-5, rtol=2e-5)
    if tracker is not None:
        for token in ids[start:start + prefix]:
            tracker.append(token)
    for offset in range(prefix, x.shape[1]):
        step = structured_step_to_tensors(tracker.append(ids[start + offset]), x.device) if tracker else None
        logits, caches = core.decode_one_with_kv_cache(x[:, offset:offset + 1], caches,
            beat[:, offset:offset + 1], position[:, offset:offset + 1], step)
        if check:
            torch.testing.assert_close(logits, reference[:, offset:offset + 1], atol=2e-5, rtol=2e-5)
    return x.shape[1] - prefix


@torch.no_grad()
def validate_core(core, vocab: list[str], variant: str) -> dict:
    """Check full/prefill/decode agreement, partial windows, causality, and stream masks."""
    from .model_core import cuda_precision_policy
    device = next(core.parameters()).device
    # Full and incremental matmuls have different shapes. TF32/autocast can
    # introduce rounding differences larger than these strict correctness tolerances.
    with cuda_precision_policy(device, allow_tf32=False), torch.autocast(device.type, enabled=False):
        training = core.training
        core.eval()
        try:
            ids, coverage = _fixture(core, vocab, variant)
            x, beat, position, routing, _ = _window(core, vocab, ids)
            reference = core(x, beat, position, routing)
            windows = sorted({0, 1, min(3, len(ids) - 2)})
            for start in windows:
                for prefix in sorted({1, max(1, (len(ids) - start) // 2)}):
                    _cached_rollout(core, vocab, ids, start, prefix)
            payloads = [i for i, token in enumerate(ids) if vocab[token].startswith('HIT_')]
            if payloads:
                payload = payloads[-1]
                changed = x.clone()
                changed[0, payload] = vocab.index('HIT_OTA' if vocab[ids[payload]] != 'HIT_OTA' else 'HIT_D')
                future = core(changed, beat, position, routing)
                torch.testing.assert_close(reference[:, :payload], future[:, :payload], atol=2e-5, rtol=2e-5)
            if len(ids) < core.context_size:
                # Future padding must not alter any real-token prediction.
                pad_ids = ids + [vocab.index('<PAD>')] * min(3, core.context_size - len(ids))
                px, pb, pp, pr, _ = _window(core, vocab, pad_ids)
                padded = core(px, pb, pp, pr)
                torch.testing.assert_close(reference, padded[:, :len(ids)], atol=2e-5, rtol=2e-5)
            checked_streams = 0
            for block in core.blocks:
                if not hasattr(block, 'use_inside_beat_stream'):
                    continue
                for index, name in enumerate(('use_inside_beat_stream', 'use_beat_memory_stream', 'use_cycle_memory_stream')):
                    original = getattr(block, name)
                    try:
                        setattr(block, name, False)
                        core(x, beat, position, routing)
                        if float(block.last_gate_usage[index]) != 0.0:
                            raise AssertionError(f'Disabled {name} has nonzero gate weight')
                        checked_streams += 1
                    finally:
                        setattr(block, name, original)
            return {'passed': True, 'coverage': coverage, 'cache_windows': windows,
                    'future_causality': bool(payloads), 'padding_invariance': len(ids) < core.context_size,
                    'disabled_stream_checks': checked_streams,
                    'padding_utilization': padding_utilization(core, routing)}
        finally:
            core.train(training)


def _write_report(output: Path, report: dict):
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


def run_model_checks(snapshot: str | Path, output_dir: str | Path, *, device: str = 'cpu') -> dict:
    from transformers import AutoModel
    print(f'[validation] Loading {snapshot} on {device}', flush=True)
    model = AutoModel.from_pretrained(snapshot, trust_remote_code=True).to(device).eval()
    report = validate_core(model.core, list(model.vocab), model.config.dataset_variant)
    report.update(snapshot=str(snapshot), config=model.config.to_dict(), torch_version=torch.__version__, device=device)
    _write_report(Path(output_dir) / 'model_checks.json', report)
    print('[validation] Cache equivalence, causality, padding, and stream isolation passed', flush=True)
    return report


def profile_cuda(snapshot: str | Path, dataset_dir: str | Path, output_dir: str | Path, *,
                 batch_size: int = 1, warmup_steps: int = 3, measured_steps: int = 10,
                 amp: bool = True, allow_tf32: bool = True) -> dict:
    """Benchmark gradients without optimizer updates and deterministic cached decoding."""
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA profiling requires a GPU runtime')
    if batch_size < 1 or warmup_steps < 0 or measured_steps < 1:
        raise ValueError('Invalid batch size or profiling step counts')
    from transformers import AutoModel
    from .data_loading import ChunkDataset
    from .dataset_integrity import verify_dataset
    from .model_core import cuda_precision_policy
    with cuda_precision_policy(torch.device('cuda'), allow_tf32=allow_tf32):
        model = AutoModel.from_pretrained(snapshot, trust_remote_code=True)
        fingerprint = verify_dataset(dataset_dir, model_config=model.config.to_dict())
        model.cuda()
        from .model_core import configure_global_flash
        attention_backend = configure_global_flash(model.core, amp=amp)
        dataset = ChunkDataset(dataset_dir, 'train', model_config=model.config)
        rows = list(range(min(batch_size, len(dataset))))
        if not rows:
            raise ValueError('No training chunks to profile')
        batch = dataset.batch(rows, torch.device('cuda'))
        params = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
        times = []
        stream_times = {}
        print(f'[profile] {torch.cuda.get_device_name(0)}: {warmup_steps} warmup, {measured_steps} measured steps', flush=True)
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            model.train()
            for step in range(warmup_steps + measured_steps):
                if step == warmup_steps:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                with torch.autocast('cuda', enabled=amp):
                    loss = model(**batch).loss
                gradients = torch.autograd.grad(loss, params, allow_unused=True)
                end.record()
                end.synchronize()
                if step >= warmup_steps:
                    times.append(start.elapsed_time(end))
                del gradients, loss
            peak = torch.cuda.max_memory_allocated() / 1024 ** 3
            # Measure stream scopes separately so CUDA event instrumentation does
            # not inflate the end-to-end structured/global comparison.
            for block in model.core.blocks:
                if hasattr(block, 'profile_stream_events'):
                    block.profile_stream_events = True
            if model.core.has_structured_layers:
                with torch.no_grad(), torch.autocast('cuda', enabled=amp):
                    for _ in range(measured_steps):
                        model(**{key: value for key, value in batch.items() if key != 'labels'})
                        torch.cuda.synchronize()
                        for layer, block in enumerate(model.core.blocks, 1):
                            if hasattr(block, 'profiled_stream_times_ms'):
                                for name, duration in block.profiled_stream_times_ms().items():
                                    stream_times.setdefault(f'layer_{layer}/{name}', []).append(duration)
            for block in model.core.blocks:
                if hasattr(block, 'profile_stream_events'):
                    block.profile_stream_events = False
            model.eval()
            ids, coverage = _fixture(model.core, list(model.vocab), model.config.dataset_variant)
            for _ in range(warmup_steps):
                _cached_rollout(model.core, list(model.vocab), ids, check=False)
            generation_times = []
            for _ in range(measured_steps):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                decoded = _cached_rollout(model.core, list(model.vocab), ids, check=False)
                end.record()
                end.synchronize()
                generation_times.append(start.elapsed_time(end))
        report = {'profile_version': 2, 'attention_backend': attention_backend, 'snapshot': str(snapshot), 'config': model.config.to_dict(),
                  'dataset_fingerprint': fingerprint,
                  'device_name': torch.cuda.get_device_name(0), 'torch_version': torch.__version__,
                  'cuda_version': torch.version.cuda, 'batch_size': len(rows),
                  'device_capability': list(torch.cuda.get_device_capability()),
                  'allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                  'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
                  'matmul_precision': torch.get_float32_matmul_precision(),
                  'sdpa_backends': {'global_policy': attention_backend['global_policy'], 'flash_available': torch.backends.cuda.is_flash_attention_available(),
                      'flash_enabled': torch.backends.cuda.flash_sdp_enabled(),
                      'math_enabled': torch.backends.cuda.math_sdp_enabled(),
                      'memory_efficient_enabled': torch.backends.cuda.mem_efficient_sdp_enabled()},
                  'context_size': int(batch['input_ids'].shape[1]), 'amp': amp,
                  'warmup_steps': warmup_steps, 'measured_steps': measured_steps,
                  'mean_step_ms': float(np.mean(times)), 'step_ms': times,
                  'peak_allocated_gib': peak,
                  'stream_forward_mean_ms': {name: float(np.mean(values)) for name, values in stream_times.items()},
                  'scored_tokens_per_second': float((batch['labels'] != -100).sum()) / (np.mean(times) / 1000),
                  'cached_generation': {'mode': 'teacher_forced_prefill_and_decode', 'coverage': coverage,
                      'amp': False, 'dtype': 'torch.float32', 'sdpa_policy': 'automatic',
                      'decoded_tokens': decoded, 'mean_ms': float(np.mean(generation_times)),
                      'tokens_per_second': decoded / (np.mean(generation_times) / 1000)},
                  'padding_utilization': padding_utilization(model.core, batch['structured_routing'])}
        _write_report(Path(output_dir) / 'cuda_profile.json', report)
        print(f"[profile] {report['mean_step_ms']:.2f} ms/step; {peak:.3f} GiB peak", flush=True)
        return report


def compare_profiles(baseline: dict, candidate: dict) -> dict:
    """Only apply the speed/memory gate to matching hardware, data, and model sizes."""
    keys = ('profile_version', 'device_name', 'torch_version', 'cuda_version', 'dataset_fingerprint',
            'batch_size', 'context_size', 'amp', 'warmup_steps', 'measured_steps',
            'device_capability', 'allow_tf32', 'cudnn_allow_tf32', 'matmul_precision', 'sdpa_backends')
    required = (*keys, 'config', 'mean_step_ms', 'peak_allocated_gib')
    validate_rules({'profiles': (all(key in report for report in (baseline, candidate) for key in required),
                                'Profiles are incomplete; create both with the current profiler')})
    model_keys = ('number_of_layers', 'number_of_heads', 'number_of_embeddings', 'mlp_ratio',
                  'dropout', 'position_encoding', 'rope_theta', 'rope_fraction', 'tie_weights',
                  'use_type_embeddings', 'concat_type_embeddings', 'dataset_variant', 'vocab_hash',
                  'activation_name', 'dataset_decoration', 'max_beat_positions', 'max_beats_per_cycle', 'max_tokens_per_beat',
                  'max_beat_slots_per_chunk', 'max_cycles_per_chunk', 'attention_layout')
    validate_rules({'config': (all(key in report['config'] for report in (baseline, candidate) for key in model_keys),
                              'Profiles contain incomplete model configuration')})
    validate_rules({
        **{key: (baseline[key] == candidate[key], f'Profiles are not comparable: {key}') for key in keys},
        **{f'config.{key}': (baseline['config'][key] == candidate['config'][key],
                            f'Profiles are not comparable: config.{key}')
           for key in model_keys if key != 'attention_layout'},
        'baseline': (baseline['config']['attention_layout'] == 'global', 'baseline must use global attention'),
        'candidate': (candidate['config']['attention_layout'] == 'alternating_global_structured',
                      'candidate must use structured attention'),
        'measurements': (all(np.isfinite(report[key]) and report[key] > 0
                             for report in (baseline, candidate) for key in ('mean_step_ms', 'peak_allocated_gib')),
                         'Profile timing and memory must be finite and positive'),
    })
    step = candidate['mean_step_ms'] / baseline['mean_step_ms']
    memory = candidate['peak_allocated_gib'] / max(baseline['peak_allocated_gib'], 1e-12)
    return {'structured_to_global_step_time': step, 'structured_to_global_peak_memory': memory,
            'not_slower_than_global': step <= 1.0, 'not_more_memory_than_global': memory <= 1.0,
            'optimization_gate_passed': step <= 1.0 and memory <= 1.0}
