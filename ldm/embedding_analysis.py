"""Static token representations, neighbor tables, and subdivision cluster statistics."""

from __future__ import annotations

import csv
import json
from enum import Enum
from pathlib import Path

import numpy as np

from .constants import DECORATED_SUFFIX


class EmbeddingRepresentation(str, Enum):
    TOKEN_ONLY = 'token_only'
    BASE_INPUT = 'base_input'
    OUTPUT_HEAD = 'output_head'


def _group(token: str) -> str:
    if token.startswith('SUBD_'):
        return 'subdivision'
    if token.endswith(DECORATED_SUFFIX):
        return 'decorated_hit'
    if token.startswith('HIT_'):
        return 'hit'
    return 'structure'


def embedding_statistics(tokens: list[str], matrix: np.ndarray, top_k: int = 8) -> dict:
    """Compute neighbor tables without Torch, excluding each token's own vector."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or len(matrix) != len(tokens) or len(tokens) < 2 or top_k < 1:
        raise ValueError('Provide one vector per token, at least two tokens, and positive top_k')
    if len(set(tokens)) != len(tokens) or not np.isfinite(matrix).all():
        raise ValueError('Tokens must be unique and vectors finite')
    normalized = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    cosine = normalized @ normalized.T
    neighbors, summaries = [], []
    subd = np.array([token.startswith('SUBD_') for token in tokens])
    k = min(top_k, len(tokens) - 1)
    for i, token in enumerate(tokens):
        indices = [int(j) for j in np.argsort(-cosine[i], kind='stable') if j != i][:k]
        neighbors.extend({'token': token, 'token_group': _group(token), 'rank': rank,
                          'neighbor': tokens[j], 'neighbor_group': _group(tokens[j]),
                          'cosine': float(cosine[i, j])}
                         for rank, j in enumerate(indices, 1))
        if subd[i]:
            others = subd.copy()
            others[i] = False
            summaries.append({'token': token, 'neighbor_purity': float(subd[indices].mean()),
                              'mean_subd_cosine': float(cosine[i, others].mean()) if others.any() else None,
                              'mean_non_subd_cosine': float(cosine[i, ~subd].mean()) if (~subd).any() else None})
    pairs = subd[:, None] & subd[None, :] & ~np.eye(len(tokens), dtype=bool)
    cross = subd[:, None] & ~subd[None, :]
    metrics = {'subdivision_analysis_applicable': bool(subd.any()), 'top_k': k,
               'mean_subd_to_subd': float(cosine[pairs].mean()) if pairs.any() else None,
               'mean_subd_to_non_subd': float(cosine[cross].mean()) if cross.any() else None,
               'max_subd_to_non_subd': float(cosine[cross].max()) if cross.any() else None,
               'overall_purity': float(np.mean([row['neighbor_purity'] for row in summaries])) if summaries else None}
    return {'neighbors': neighbors, 'subdivision_summary': summaries, 'metrics': metrics}


def representation_matrix(model, mode: EmbeddingRepresentation):
    """Match the model's actual token/type addition or concatenation projection."""
    if not isinstance(mode, EmbeddingRepresentation):
        raise TypeError('Use an EmbeddingRepresentation option')
    import torch
    core = model.core
    if mode is EmbeddingRepresentation.OUTPUT_HEAD:
        return core.head.weight.detach().float()
    token = core.token_embeddings.weight.detach()
    if mode is EmbeddingRepresentation.TOKEN_ONLY or not core.use_type_embeddings:
        return token.float()
    types = core.type_embeddings(core.token_type_ids)
    if core.concat_type_embeddings:
        return core.type_concat_projection(torch.cat((token, types), dim=-1)).detach().float()
    return (token + types).detach().float()


def analyze_static_embeddings(snapshot: str | Path, output_dir: str | Path, *,
                              representations: list[EmbeddingRepresentation] | tuple = (EmbeddingRepresentation.BASE_INPUT,),
                              top_k: int = 8) -> list[dict]:
    import torch
    from transformers import AutoModel
    if not representations or any(not isinstance(mode, EmbeddingRepresentation) for mode in representations):
        raise ValueError('Choose at least one EmbeddingRepresentation')
    model = AutoModel.from_pretrained(snapshot, trust_remote_code=True).eval()
    reports = []
    for representation in dict.fromkeys(representations):
        with torch.inference_mode():
            matrix = representation_matrix(model, representation).cpu().numpy()
        result = embedding_statistics(list(model.vocab), matrix, top_k)
        target = Path(output_dir) / representation.value
        target.mkdir(parents=True, exist_ok=True)
        tables = {'neighbors.csv': result['neighbors'],
                  'subd_neighbors.csv': [row for row in result['neighbors'] if row['token_group'] == 'subdivision'],
                  'subd_summary.csv': result['subdivision_summary']}
        for name, rows in tables.items():
            fields = list(rows[0]) if rows else (list(result['neighbors'][0]) if 'neighbors' in name else
                      ['token', 'neighbor_purity', 'mean_subd_cosine', 'mean_non_subd_cosine'])
            with (target / name).open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        manifest = {'snapshot': str(snapshot), 'representation': representation.value,
                    'vocab_hash': model.config.vocab_hash, 'metrics': result['metrics']}
        (target / 'metrics.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        print(f'[embeddings] {representation.value}: {target}', flush=True)
        reports.append({'output_dir': str(target), **manifest})
    return reports
