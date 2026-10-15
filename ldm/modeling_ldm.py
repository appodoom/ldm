"""Transformers wrapper around the one canonical LDM attention model."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import torch
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

from .configuration_ldm import LDMConfig
from .cycle_collector import IncompleteGenerationError
from .derbake_ldm import RhythmSequence  # HF caches only direct relative imports.
from .generation_ldm import generate_continuation_ids
from .model_core import GPT
from .rhythm_ldm import STRUCTURED_ROUTING_FIELDS  # Needed by model_core/generation_ldm.
from .schema_ldm import RhythmSchema
from .vocabulary import atomic_vocabulary, vocabulary_hash


@dataclass
class LDMGenerationOutput:
    continuation_ids: torch.Tensor
    report: dict


class LDMModel(PreTrainedModel):
    config_class = LDMConfig
    base_model_prefix = "core"
    main_input_name = "input_ids"
    _tied_weights_keys = ["core.head.weight"]

    def __init__(self, config: LDMConfig):
        super().__init__(config)
        vocab = atomic_vocabulary(config.dataset_variant)
        if config.vocab_size not in {0, len(vocab)}:
            raise ValueError("Config vocabulary size differs from schema")
        if config.vocab_hash and config.vocab_hash != vocabulary_hash(vocab):
            raise ValueError("Config vocabulary hash differs from schema")
        config.vocab_size = len(vocab)
        config.vocab_hash = vocabulary_hash(vocab)
        self.vocab = vocab
        self.core = GPT(
            vocab=list(vocab), context_size=config.context_size,
            number_of_layers=config.number_of_layers, number_of_heads=config.number_of_heads,
            number_of_embeddings=config.number_of_embeddings, mlp_ratio=config.mlp_ratio,
            dropout=config.dropout, tie_weights=config.tie_weights,
            use_type_embeddings=config.use_type_embeddings,
            concat_type_embeddings=config.concat_type_embeddings,
            activation_name=config.activation_name, position_encoding=config.position_encoding,
            rope_theta=config.rope_theta, rope_fraction=config.rope_fraction,
            max_beat_positions=config.max_beat_positions,
            max_beats_per_cycle=config.max_beats_per_cycle,
            attention_layout=config.attention_layout,
            use_inside_beat_stream=config.use_inside_beat_stream,
            use_beat_memory_stream=config.use_beat_memory_stream,
            use_cycle_memory_stream=config.use_cycle_memory_stream,
            max_tokens_per_beat=config.max_tokens_per_beat,
            max_beat_slots_per_chunk=config.max_beat_slots_per_chunk,
            max_cycles_per_chunk=config.max_cycles_per_chunk,
        )
        self.post_init()

    def get_input_embeddings(self):
        return self.core.token_embeddings

    def set_input_embeddings(self, value):
        self.core.token_embeddings = value

    def get_output_embeddings(self):
        return self.core.head

    def set_output_embeddings(self, value):
        self.core.head = value

    def forward(self, input_ids, labels=None, beat_idx_ids=None, beat_pos_ids=None,
                structured_routing=None, **kwargs):
        logits = self.core(input_ids, beat_idx_ids=beat_idx_ids,
                           beat_pos_ids=beat_pos_ids, structured_routing=structured_routing)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
                                   ignore_index=-100)
        return CausalLMOutput(loss=loss, logits=logits)

    def generate(self, input_ids: torch.Tensor, *, num_cycles: int, max_new_tokens: int = 4096,
                 temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0,
                 seed: int | None = None, return_generation_report: bool = False,
                 cache_refresh_tokens: int = 256):
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("LDM generation supports batch size 1")
        if input_ids.shape[1] < 1 or input_ids.shape[1] > self.config.context_size:
            raise ValueError("Prompt length exceeds model context")
        schema = RhythmSchema(self.config.dataset_variant, self.config.max_beats_per_cycle)
        prompt_ids = input_ids[0].tolist()
        if any(index < 0 or index >= len(self.vocab) for index in prompt_ids):
            raise ValueError("Prompt contains a token ID outside the vocabulary")
        try:
            schema.decode([self.vocab[index] for index in prompt_ids])
        except (IndexError, ValueError) as exc:
            raise ValueError("Prompt must contain only complete valid cycles") from exc
        try:
            ids, report = generate_continuation_ids(
                self.core, list(self.vocab), prompt_ids, max_new_tokens,
                temperature, top_k, top_p, seed, num_cycles, schema,
                cache_refresh_tokens=cache_refresh_tokens,
            )
        except IncompleteGenerationError as exc:
            raise IncompleteGenerationError(exc.report) from exc
        if report["discarded_cycles"] or report["orphan_tokens"]:
            warnings.warn(f"LDM skipped {report['discarded_cycles']} invalid cycles and "
                          f"{report['orphan_tokens']} orphan tokens; inspect generation report",
                          stacklevel=2)
        result = torch.tensor([ids], dtype=torch.long, device=input_ids.device)
        if return_generation_report:
            return LDMGenerationOutput(result, report)
        return result
