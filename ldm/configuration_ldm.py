"""Model-only configuration; training settings stay in run metadata."""

from __future__ import annotations

from transformers import PretrainedConfig


class LDMConfig(PretrainedConfig):
    model_type = "ldm_rhythm"

    def __init__(
        self, vocab_size: int = 0, vocab_hash: str = "", dataset_variant: str = "with_subd",
        context_size: int = 2048, max_beats_per_cycle: int = 16,
        number_of_layers: int = 4, number_of_heads: int = 4,
        number_of_embeddings: int = 256, mlp_ratio: int = 4,
        dropout: float = 0.1, tie_weights: bool = True,
        use_type_embeddings: bool = True, concat_type_embeddings: bool = False,
        activation_name: str = "tanh", position_encoding: str = "rope",
        rope_theta: float = 10000.0, rope_fraction: float = 1.0,
        max_beat_positions: int = 8, attention_layout: str = "alternating_global_structured",
        use_inside_beat_stream: bool = True, use_beat_memory_stream: bool = True,
        use_cycle_memory_stream: bool = True, max_tokens_per_beat: int = 16,
        max_beat_slots_per_chunk: int = 512, max_cycles_per_chunk: int = 256,
        derbake_format_version: int = 2, token_schema_version: int = 1, **kwargs,
    ):
        kwargs.pop("tie_word_embeddings", None)
        super().__init__(tie_word_embeddings=tie_weights, **kwargs)
        for name, value in locals().copy().items():
            if name not in {"self", "kwargs", "__class__"}:
                setattr(self, name, value)
        if dataset_variant not in {"with_subd", "no_subd"}:
            raise ValueError("Unsupported dataset variant")
        if vocab_size < 0 or context_size < 2 or max_beats_per_cycle < 1:
            raise ValueError("Invalid model capacities")
