"""Content verification for every prepared training and evaluation artifact."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .rhythm_ldm import STRUCTURED_ROUTING_FIELDS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _expected_arrays(manifest: dict) -> dict:
    expected = {}
    for split in ('train', 'val', 'test'):
        rows = manifest['stats'][split]['chunks']
        width = manifest['context_size'] + 1
        shapes = {'': ((rows, width), 'int32'),
                  '_source_kind': ((rows,), 'uint8'),
                  '_source_file': ((rows,), 'int32'),
                  '_beat_idx': ((rows, width), 'int32'),
                  '_beat_pos': ((rows, width), 'int32')}
        for name in ('score_mask', 'hit_count_mask', 'skeleton_mask'):
            shapes[f'_{name}'] = ((rows, width - 1), 'bool')
        for name in STRUCTURED_ROUTING_FIELDS:
            shapes[f'_{name}'] = ((rows, width), 'bool' if name.endswith('_mask') else 'int32')
        expected.update({f'{split}{suffix}.npy': value for suffix, value in shapes.items()})
    return expected


def _inventory(folder: Path, manifest: dict) -> dict:
    expected = _expected_arrays(manifest)
    actual = {path.name for path in folder.glob('*.npy')}
    if actual != set(expected):
        raise ValueError(f'Prepared arrays differ: missing={sorted(set(expected) - actual)}, '
                         f'unexpected={sorted(actual - set(expected))}')
    result = {}
    for name, (shape, dtype) in sorted(expected.items()):
        path = folder / name
        array = np.load(path, mmap_mode='r', allow_pickle=False)
        if array.shape != shape or array.dtype != np.dtype(dtype):
            raise ValueError(f'Invalid prepared array {name}: {array.shape}/{array.dtype}; '
                             f'expected {shape}/{dtype}')
        result[name] = {'sha256': _sha256(path), 'bytes': path.stat().st_size,
                        'shape': list(shape), 'dtype': str(array.dtype)}
    return result


def seal_dataset(folder: str | Path, manifest: dict) -> None:
    """Record the completed payload only after all arrays have been written."""
    folder = Path(folder)
    manifest['integrity_version'] = 1
    manifest['artifacts'] = _inventory(folder, manifest)
    target = folder / 'dataset_manifest.json'
    temporary = target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    temporary.replace(target)


def dataset_fingerprint(folder: str | Path) -> str:
    manifest = json.loads((Path(folder) / 'dataset_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('integrity_version') != 1 or 'artifacts' not in manifest:
        raise ValueError('Prepared dataset has no content inventory. Reprepare data for new runs; '
                         'resuming an older run requires a verified dataset migration')
    # Local paths do not change the identity of identical, verified data.
    stable = {key: value for key, value in manifest.items()
              if key not in {'ordinary_dir', 'decorated_dir'}}
    return hashlib.sha256(json.dumps(stable, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def verify_dataset(folder: str | Path, *, model_config: dict | None = None) -> str:
    folder = Path(folder)
    fingerprint = dataset_fingerprint(folder)
    manifest = json.loads((folder / 'dataset_manifest.json').read_text(encoding='utf-8'))
    if model_config is not None:
        fields = {'variant': 'dataset_variant', 'vocab_hash': 'vocab_hash',
                  'context_size': 'context_size', 'max_beats_per_cycle': 'max_beats_per_cycle'}
        differences = [key for key, model_key in fields.items()
                       if manifest[key] != model_config.get(model_key)]
        if model_config.get('attention_layout') == 'alternating_global_structured':
            for key in ('max_tokens_per_beat', 'max_beat_slots_per_chunk', 'max_cycles_per_chunk'):
                required = max(stats[key] for stats in manifest['stats'].values())
                if required > model_config.get(key, 0):
                    differences.append(key)
        if differences:
            raise ValueError(f"Prepared dataset differs from model: {', '.join(differences)}")
    actual = _inventory(folder, manifest)
    if actual != manifest['artifacts']:
        changed = sorted(name for name in actual if actual[name] != manifest['artifacts'].get(name))
        raise ValueError(f'Prepared dataset content check failed: {changed}')
    return fingerprint
