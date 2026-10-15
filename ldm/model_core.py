"""Canonical LDM attention/model math, shared by training and HF loading.

Extracted from the notebook without changing the attention equations.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import math
import torch
from torch import nn
from torch.nn import functional as F
from .rhythm_ldm import STRUCTURED_ROUTING_FIELDS, count_hit_atoms, parse_subdivision_token

def build_token_to_id(vocab_list: List[str]) -> Dict[str, int]:
    token_to_id = {tok: i for i, tok in enumerate(vocab_list)}

    if len(token_to_id) != len(vocab_list):
        raise ValueError("vocab_list has duplicate tokens")

    return token_to_id


def _build_id_to_token(token_to_id: Dict[str, int]) -> list:
    vocab_size = len(token_to_id)
    id_to_token = ["<UNK>"] * vocab_size
    for tok, idx in token_to_id.items():
        if 0 <= idx < vocab_size:
            id_to_token[idx] = tok
    return id_to_token


# Positional Embedding
def build_sinusoidal_table(max_len: int, dim: int) -> torch.Tensor:
    """
    Returns [max_len, dim] fixed sinusoidal positional encodings (on CPU).
    When you .to(device) the model, this buffer follows automatically.
    """
    pe = torch.zeros(max_len, dim, dtype=torch.float32)
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(
        1
    )  # [max_len, 1]
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
    )

    pe[:, 0::2] = torch.sin(position * div_term)
    if dim % 2 == 0:
        pe[:, 1::2] = torch.cos(position * div_term)
    else:
        pe[:, 1::2] = torch.cos(div_term[: (dim // 2)] * position)
    return pe


# Precompute useful information about each token in the vocabulary before training starts.
def build_token_metadata_buffers(
    token_to_id: Dict[str, int],
    max_beat_positions: Optional[int] = None,
):
    """
    Types:
      0 = DELIM  (<SOC>, <EOC>, <SOB>, <EOB> and generally <...>)
      1 = SUBD   (SUBD_4, SUBD_8, etc.)
      2 = HIT    (HIT_*)
      3 = OTHER
    """
    vocab_size = len(token_to_id)
    id_to_token = _build_id_to_token(token_to_id)

    TYPE_DELIM = 0
    TYPE_SUBD = 1
    TYPE_HIT = 2
    TYPE_OTHER = 3
    n_types = 4

    token_type_ids = torch.full((vocab_size,), TYPE_OTHER, dtype=torch.long)
    is_hit = torch.zeros((vocab_size,), dtype=torch.bool)
    hit_atom_counts = torch.zeros((vocab_size,), dtype=torch.long)
    is_sob = torch.zeros((vocab_size,), dtype=torch.bool)

    max_subd = 1

    for idx, tok in enumerate(id_to_token):
        if tok == "<SOB>":
            is_sob[idx] = True

        if tok in {"<SOC>", "<EOC>", "<SOB>", "<EOB>"} or (
            tok.startswith("<") and tok.endswith(">")
        ):
            token_type_ids[idx] = TYPE_DELIM
            continue

        subd = parse_subdivision_token(tok)
        if subd is not None:
            token_type_ids[idx] = TYPE_SUBD
            if subd > max_subd:
                max_subd = subd
            continue

        hit_count = count_hit_atoms(tok)
        if hit_count:
            token_type_ids[idx] = TYPE_HIT
            is_hit[idx] = True
            hit_atom_counts[idx] = hit_count
            continue

        token_type_ids[idx] = TYPE_OTHER

    if max_beat_positions is not None:
        max_beat_positions = int(max_beat_positions)
        if max_beat_positions < 1:
            raise ValueError("max_beat_positions must be positive")
    # Keep embeddings large enough for any allowed subdivision.
    max_beat_positions = max(max_subd, max_beat_positions or 1)

    return {
        "n_types": n_types,
        "TYPE_SUBD": TYPE_SUBD,
        "token_type_ids": token_type_ids,
        "is_hit": is_hit,
        "hit_atom_counts": hit_atom_counts,
        "is_sob": is_sob,
        "max_beat_positions": max_beat_positions,
    }

# Attention implementations used by both global and structured layouts.
VALID_ATTENTION_LAYOUTS = frozenset({"global", "alternating_global_structured"})


def make_activation(name: str) -> nn.Module:
    name = name.lower().strip()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name in {"silu", "swish"}:
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    if name in {"leaky_relu", "lrelu"}:
        return nn.LeakyReLU(negative_slope=0.1)
    raise ValueError(f"Unknown activation: {name}")


def resolve_attention_layout(attention_layout: str, number_of_layers: int) -> List[str]:
    layout = str(attention_layout).strip().lower()
    if layout not in VALID_ATTENTION_LAYOUTS:
        raise ValueError(
            f"attention_layout must be one of {sorted(VALID_ATTENTION_LAYOUTS)}, got {layout!r}"
        )
    if number_of_layers <= 0:
        raise ValueError("number_of_layers must be positive")
    if layout == "global":
        return ["global"] * number_of_layers
    # One-based odd layers are global; one-based even layers are structured.
    return ["global" if index % 2 == 0 else "structured" for index in range(number_of_layers)]


def apply_rotary_embedding(
    tensor: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    """Apply pairwise RoPE to the leading dimensions of [..., head_dim]."""
    rotary_dim = cosine.size(-1)
    if rotary_dim <= 0 or rotary_dim > tensor.size(-1) or rotary_dim % 2 != 0:
        raise ValueError(
            f"Invalid rotary_dim={rotary_dim} for head_dim={tensor.size(-1)}"
        )
    cosine = cosine.to(device=tensor.device, dtype=tensor.dtype)
    sine = sine.to(device=tensor.device, dtype=tensor.dtype)
    rotary = tensor[..., :rotary_dim]
    passthrough = tensor[..., rotary_dim:]
    pairs = rotary.reshape(*rotary.shape[:-1], rotary_dim // 2, 2)
    rotated = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)
    rotary = rotary * cosine + rotated * sine
    return torch.cat((rotary, passthrough), dim=-1) if passthrough.size(-1) else rotary


def rope_cos_sin_for_positions(
    positions: torch.Tensor,
    rope_inv_freq: Optional[torch.Tensor],
    dtype: torch.dtype,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Return broadcastable RoPE tensors for an arbitrary [B, T] coordinate."""
    if rope_inv_freq is None:
        return None, None
    frequencies = positions.to(torch.float32).unsqueeze(-1) * rope_inv_freq.float()
    cosine = frequencies.cos().repeat_interleave(2, dim=-1).unsqueeze(1)
    sine = frequencies.sin().repeat_interleave(2, dim=-1).unsqueeze(1)
    if torch.is_autocast_enabled(positions.device.type):
        dtype = torch.get_autocast_dtype(positions.device.type)
    return cosine.to(dtype=dtype), sine.to(dtype=dtype)


class Transformer_Attention_Block(nn.Module):
    """Pre-norm causal global attention with exact incremental KV caching."""

    def __init__(
        self,
        number_embeddings: int,
        number_heads: int,
        mlp_ratio: int,
        dropout: float,
        activation_name: str = "gelu",
    ):
        super().__init__()
        if number_embeddings % number_heads != 0:
            raise ValueError("number_embeddings must be divisible by number_heads")
        self.layer_norm_1 = nn.LayerNorm(number_embeddings)
        self.number_heads = number_heads
        self.head_dim = number_embeddings // number_heads
        self.attention_layer = nn.MultiheadAttention(
            embed_dim=number_embeddings,
            num_heads=number_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm_2 = nn.LayerNorm(number_embeddings)
        hidden = mlp_ratio * number_embeddings
        self.multi_layer_perceptron = nn.Sequential(
            nn.Linear(number_embeddings, hidden),
            make_activation(activation_name),
            nn.Linear(hidden, number_embeddings),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def _project_qkv(self, x: torch.Tensor):
        batch_size, sequence_length, _ = x.shape
        qkv = F.linear(
            x,
            self.attention_layer.in_proj_weight,
            self.attention_layer.in_proj_bias,
        )
        return tuple(
            part.view(batch_size, sequence_length, self.number_heads, self.head_dim)
            .transpose(1, 2)
            for part in qkv.chunk(3, dim=-1)
        )

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pre_attention = self.layer_norm_1(x)
        batch_size, sequence_length, embed_dim = pre_attention.shape
        query, key, value = self._project_qkv(pre_attention)
        if (rope_cos is None) != (rope_sin is None):
            raise ValueError("rope_cos and rope_sin must be provided together")
        if rope_cos is not None:
            query = apply_rotary_embedding(query, rope_cos, rope_sin)
            key = apply_rotary_embedding(key, rope_cos, rope_sin)

        kwargs = {
            "dropout_p": self.attention_layer.dropout if self.training else 0.0,
            "is_causal": True,
        }
        attention = F.scaled_dot_product_attention(query, key, value, **kwargs)
        attention = attention.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, embed_dim
        )
        x = x + self.dropout(self.attention_layer.out_proj(attention))
        return x + self.multi_layer_perceptron(self.layer_norm_2(x))

    def forward_with_kv_cache(
        self,
        x: torch.Tensor,
        kv_cache=None,
        max_cache_len: Optional[int] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ):
        if self.training:
            raise RuntimeError("KV-cached attention requires model.eval()")
        pre_attention = self.layer_norm_1(x)
        batch_size, query_len, embed_dim = pre_attention.shape
        query, key, value = self._project_qkv(pre_attention)
        if (rope_cos is None) != (rope_sin is None):
            raise ValueError("rope_cos and rope_sin must be provided together")
        if rope_cos is not None:
            query = apply_rotary_embedding(query, rope_cos, rope_sin)
            key = apply_rotary_embedding(key, rope_cos, rope_sin)

        if kv_cache is None:
            max_cache_len = query_len if max_cache_len is None else int(max_cache_len)
            if query_len > max_cache_len:
                raise ValueError("Prefill exceeds the allocated global KV cache")
            key_cache = key.new_empty(batch_size, self.number_heads, max_cache_len, self.head_dim)
            value_cache = value.new_empty(
                batch_size, self.number_heads, max_cache_len, self.head_dim
            )
            key_cache[:, :, :query_len].copy_(key)
            value_cache[:, :, :query_len].copy_(value)
            cache_len = query_len
            key_for_attention, value_for_attention = key, value
            is_causal = query_len > 1
        else:
            key_cache, value_cache, cache_len = kv_cache
            if query_len != 1:
                raise ValueError("Cached global decode accepts one token")
            if cache_len >= key_cache.size(2):
                raise ValueError("Global KV cache is full; refresh the context window")
            key_cache[:, :, cache_len : cache_len + 1].copy_(key)
            value_cache[:, :, cache_len : cache_len + 1].copy_(value)
            cache_len += 1
            key_for_attention = key_cache[:, :, :cache_len]
            value_for_attention = value_cache[:, :, :cache_len]
            is_causal = False

        attention = F.scaled_dot_product_attention(
            query, key_for_attention, value_for_attention, dropout_p=0.0, is_causal=is_causal
        )
        attention = attention.transpose(1, 2).contiguous().view(
            batch_size, query_len, embed_dim
        )
        x = x + self.dropout(self.attention_layer.out_proj(attention))
        x = x + self.multi_layer_perceptron(self.layer_norm_2(x))
        return x, (key_cache, value_cache, cache_len)


class Structured_Attention_Block(nn.Module):
    """Vectorized local/beat/cycle replacement block using shared Q/K/V."""

    def __init__(
        self,
        number_embeddings: int,
        number_heads: int,
        mlp_ratio: int,
        dropout: float,
        activation_name: str,
        max_tokens_per_beat: int,
        max_beat_slots_per_chunk: int,
        max_beats_per_cycle: int,
        max_cycles_per_chunk: int,
        use_inside_beat_stream: bool,
        use_beat_memory_stream: bool,
        use_cycle_memory_stream: bool,
    ):
        super().__init__()
        if number_embeddings % number_heads != 0:
            raise ValueError("number_embeddings must be divisible by number_heads")
        for name, value in {
            "max_tokens_per_beat": max_tokens_per_beat,
            "max_beat_slots_per_chunk": max_beat_slots_per_chunk,
            "max_beats_per_cycle": max_beats_per_cycle,
            "max_cycles_per_chunk": max_cycles_per_chunk,
        }.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive for structured attention")

        self.number_embeddings = int(number_embeddings)
        self.number_heads = int(number_heads)
        self.head_dim = number_embeddings // number_heads
        self.max_tokens_per_beat = int(max_tokens_per_beat)
        self.max_beat_slots_per_chunk = int(max_beat_slots_per_chunk)
        self.max_beats_per_cycle = int(max_beats_per_cycle)
        self.max_cycles_per_chunk = int(max_cycles_per_chunk)
        self.use_inside_beat_stream = bool(use_inside_beat_stream)
        self.use_beat_memory_stream = bool(use_beat_memory_stream)
        self.use_cycle_memory_stream = bool(use_cycle_memory_stream)

        self.layer_norm_1 = nn.LayerNorm(number_embeddings)
        self.qkv_projection = nn.Linear(number_embeddings, 3 * number_embeddings)
        self.stream_gate = nn.Linear(number_embeddings, 4)
        self.output_projection = nn.Linear(number_embeddings, number_embeddings)
        self.attention_dropout = float(dropout)
        self.residual_dropout = nn.Dropout(dropout)
        self.layer_norm_2 = nn.LayerNorm(number_embeddings)
        hidden = mlp_ratio * number_embeddings
        self.multi_layer_perceptron = nn.Sequential(
            nn.Linear(number_embeddings, hidden),
            make_activation(activation_name),
            nn.Linear(hidden, number_embeddings),
            nn.Dropout(dropout),
        )
        self.last_gate_usage = None
        self.profile_stream_events = False
        self._pending_stream_events = {}

    def _begin_stream_timer(self, name: str, tensor: torch.Tensor):
        if not self.profile_stream_events or not tensor.is_cuda:
            return None
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        return name, start, end

    def _finish_stream_timer(self, timer) -> None:
        if timer is None:
            return
        name, start, end = timer
        end.record()
        self._pending_stream_events[name] = (start, end)

    def profiled_stream_times_ms(self) -> Dict[str, float]:
        return {
            name: start.elapsed_time(end)
            for name, (start, end) in self._pending_stream_events.items()
        }

    def _project_qkv(self, normalized: torch.Tensor):
        batch_size, sequence_length, _ = normalized.shape
        return tuple(
            part.view(batch_size, sequence_length, self.number_heads, self.head_dim)
            .transpose(1, 2)
            for part in self.qkv_projection(normalized).chunk(3, dim=-1)
        )

    @staticmethod
    def _rotate(
        tensor: torch.Tensor,
        positions: torch.Tensor,
        rope_inv_freq: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if positions.ndim == 1:
            positions = positions.unsqueeze(1)
        cosine, sine = rope_cos_sin_for_positions(
            positions.clamp_min(0), rope_inv_freq, tensor.dtype
        )
        return tensor if cosine is None else apply_rotary_embedding(tensor, cosine, sine)

    def _pack_qkv(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        group_slot: torch.Tensor,
        item_position: torch.Tensor,
        valid: torch.Tensor,
        groups_per_batch: int,
        items_per_group: int,
    ):
        """Fixed-shape scatter-add packing; no dynamic Python loops or nonzero()."""
        batch_size, heads, sequence_length, head_dim = query.shape
        valid = (
            valid
            & (group_slot >= 0)
            & (group_slot < groups_per_batch)
            & (item_position >= 0)
            & (item_position < items_per_group)
        )
        batch_index = torch.arange(batch_size, device=query.device).unsqueeze(1)
        flat_index = (
            (batch_index * groups_per_batch + group_slot.clamp(0, groups_per_batch - 1))
            * items_per_group
            + item_position.clamp(0, items_per_group - 1)
        ).reshape(-1)
        flat_size = batch_size * groups_per_batch * items_per_group
        multiplier = valid.reshape(-1, 1, 1).to(query.dtype)

        def pack(tensor: torch.Tensor):
            source = tensor.transpose(1, 2).reshape(-1, heads, head_dim) * multiplier
            packed = tensor.new_zeros(flat_size, heads, head_dim)
            packed.scatter_add_(
                0, flat_index[:, None, None].expand(-1, heads, head_dim), source
            )
            return packed.view(
                batch_size * groups_per_batch, items_per_group, heads, head_dim
            ).transpose(1, 2)

        counts = torch.zeros(flat_size, device=query.device, dtype=torch.int32)
        counts.scatter_add_(0, flat_index, valid.reshape(-1).to(torch.int32))
        packed_valid = counts.view(batch_size * groups_per_batch, items_per_group) > 0
        return pack(query), pack(key), pack(value), packed_valid, valid

    @staticmethod
    def _packed_causal_sdpa(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        valid: torch.Tensor,
        dropout_p: float,
    ) -> torch.Tensor:
        length = query.size(2)
        causal = torch.ones(length, length, dtype=torch.bool, device=query.device).tril()
        query_valid = valid[:, None, :, None]
        key_valid = valid[:, None, None, :]
        mask = query_valid & key_valid & causal.view(1, 1, length, length)
        # Give padded query rows a harmless zero-valued diagonal instead of an
        # all-masked softmax row. Their outputs are explicitly zeroed below.
        diagonal = torch.eye(length, dtype=torch.bool, device=query.device).view(
            1, 1, length, length
        )
        mask = mask | ((~query_valid) & diagonal)
        output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=mask, dropout_p=dropout_p, is_causal=False
        )
        return output * valid[:, None, :, None].to(output.dtype)

    def _unpack_tokens(
        self,
        packed: torch.Tensor,
        group_slot: torch.Tensor,
        item_position: torch.Tensor,
        valid: torch.Tensor,
        groups_per_batch: int,
        items_per_group: int,
    ) -> torch.Tensor:
        """Gather one packed item per token without expanding the packed length."""
        batch_size, sequence_length = group_slot.shape
        batch_index = torch.arange(batch_size, device=packed.device).unsqueeze(1)
        flat_group = (
            batch_index * groups_per_batch
            + group_slot.clamp(0, groups_per_batch - 1)
        )
        # [B*G, H, L, D] -> [B*G, L, H, D], followed by paired
        # advanced indices. The result is directly [B, T, H, D]; importantly,
        # no [B, T, H, L, D] intermediate is ever materialized.
        packed_by_item = packed.permute(0, 2, 1, 3)
        selected = packed_by_item[
            flat_group,
            item_position.clamp(0, items_per_group - 1),
        ]
        selected = selected.reshape(
            batch_size, sequence_length, self.number_embeddings
        )
        return selected * valid.unsqueeze(-1).to(selected.dtype)

    def _scatter_context_table(
        self,
        token_context: torch.Tensor,
        slots: torch.Tensor,
        valid: torch.Tensor,
        table_size: int,
    ) -> torch.Tensor:
        batch_size, sequence_length, embed_dim = token_context.shape
        batch_index = torch.arange(batch_size, device=token_context.device).unsqueeze(1)
        flat_index = (
            batch_index * table_size + slots.clamp(0, table_size - 1)
        ).reshape(-1)
        source = token_context.reshape(-1, embed_dim) * valid.reshape(-1, 1).to(
            token_context.dtype
        )
        table = token_context.new_zeros(batch_size * table_size, embed_dim)
        table.scatter_add_(0, flat_index[:, None].expand(-1, embed_dim), source)
        return table.view(batch_size, table_size, embed_dim)

    @staticmethod
    def _gather_context_table(
        table: torch.Tensor,
        slots: torch.Tensor,
        available: torch.Tensor,
    ) -> torch.Tensor:
        index = slots.clamp(0, table.size(1) - 1).unsqueeze(-1).expand(-1, -1, table.size(2))
        gathered = torch.gather(table, 1, index)
        return gathered * available.unsqueeze(-1).to(gathered.dtype)

    def _compute_streams(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        beat_idx_ids: torch.Tensor,
        routing: Dict[str, torch.Tensor],
        rope_inv_freq: Optional[torch.Tensor],
    ):
        batch_size, _, sequence_length, _ = query.shape
        zero_context = query.new_zeros(batch_size, sequence_length, self.number_embeddings)
        false_mask = torch.zeros(
            batch_size, sequence_length, dtype=torch.bool, device=query.device
        )
        dropout_p = self.attention_dropout if self.training else 0.0
        auxiliary = {}
        self._pending_stream_events = {}

        local_context, local_available = zero_context, false_mask
        if self.use_inside_beat_stream:
            stream_timer = self._begin_stream_timer("inside_beat", query)
            local_position = routing["local_beat_position"]
            local_valid = (routing["beat_slot"] >= 0) & (local_position >= 0)
            local_q = self._rotate(query, local_position, rope_inv_freq)
            local_k = self._rotate(key, local_position, rope_inv_freq)
            packed_q, packed_k, packed_v, packed_valid, local_valid = self._pack_qkv(
                local_q,
                local_k,
                value,
                routing["beat_slot"],
                local_position,
                local_valid,
                self.max_beat_slots_per_chunk,
                self.max_tokens_per_beat,
            )
            packed_output = self._packed_causal_sdpa(
                packed_q, packed_k, packed_v, packed_valid, dropout_p
            )
            local_context = self._unpack_tokens(
                packed_output,
                routing["beat_slot"],
                local_position,
                local_valid,
                self.max_beat_slots_per_chunk,
                self.max_tokens_per_beat,
            )
            local_available = local_valid
            auxiliary.update(
                local_k=packed_k,
                local_v=packed_v,
                local_valid=packed_valid,
            )
            self._finish_stream_timer(stream_timer)

        beat_context, beat_available = zero_context, false_mask
        beat_table = query.new_zeros(
            batch_size, self.max_beat_slots_per_chunk, self.number_embeddings
        )
        if self.use_beat_memory_stream:
            stream_timer = self._begin_stream_timer("beat_memory", query)
            endpoint_valid = routing["complete_eob_mask"]
            beat_q = self._rotate(query, beat_idx_ids, rope_inv_freq)
            beat_k = self._rotate(key, beat_idx_ids, rope_inv_freq)
            packed_q, packed_k, packed_v, packed_valid, endpoint_valid = self._pack_qkv(
                beat_q,
                beat_k,
                value,
                routing["cycle_slot"],
                beat_idx_ids,
                endpoint_valid,
                self.max_cycles_per_chunk,
                self.max_beats_per_cycle,
            )
            packed_output = self._packed_causal_sdpa(
                packed_q, packed_k, packed_v, packed_valid, dropout_p
            )
            endpoint_context = self._unpack_tokens(
                packed_output,
                routing["cycle_slot"],
                beat_idx_ids,
                endpoint_valid,
                self.max_cycles_per_chunk,
                self.max_beats_per_cycle,
            )
            beat_table = self._scatter_context_table(
                endpoint_context,
                routing["beat_slot"],
                endpoint_valid,
                self.max_beat_slots_per_chunk,
            )
            previous_slot = routing["previous_completed_beat_slot"]
            beat_available = (previous_slot >= 0) & (
                previous_slot < self.max_beat_slots_per_chunk
            )
            beat_context = self._gather_context_table(
                beat_table, previous_slot, beat_available
            )
            auxiliary.update(
                beat_k=packed_k,
                beat_v=packed_v,
                beat_valid=packed_valid,
                beat_table=beat_table,
            )
            self._finish_stream_timer(stream_timer)

        cycle_context, cycle_available = zero_context, false_mask
        cycle_table = query.new_zeros(
            batch_size, self.max_cycles_per_chunk, self.number_embeddings
        )
        if self.use_cycle_memory_stream:
            stream_timer = self._begin_stream_timer("cycle_memory", query)
            endpoint_valid = routing["complete_eoc_mask"]
            cycle_position = routing["cycle_slot"]
            cycle_q = self._rotate(query, cycle_position, rope_inv_freq)
            cycle_k = self._rotate(key, cycle_position, rope_inv_freq)
            zero_groups = torch.zeros_like(cycle_position)
            packed_q, packed_k, packed_v, packed_valid, endpoint_valid = self._pack_qkv(
                cycle_q,
                cycle_k,
                value,
                zero_groups,
                cycle_position,
                endpoint_valid,
                1,
                self.max_cycles_per_chunk,
            )
            packed_output = self._packed_causal_sdpa(
                packed_q, packed_k, packed_v, packed_valid, dropout_p
            )
            endpoint_context = self._unpack_tokens(
                packed_output,
                zero_groups,
                cycle_position,
                endpoint_valid,
                1,
                self.max_cycles_per_chunk,
            )
            cycle_table = self._scatter_context_table(
                endpoint_context,
                cycle_position,
                endpoint_valid,
                self.max_cycles_per_chunk,
            )
            previous_slot = routing["previous_completed_cycle_slot"]
            cycle_available = (previous_slot >= 0) & (
                previous_slot < self.max_cycles_per_chunk
            )
            cycle_context = self._gather_context_table(
                cycle_table, previous_slot, cycle_available
            )
            auxiliary.update(
                cycle_k=packed_k,
                cycle_v=packed_v,
                cycle_valid=packed_valid,
                cycle_table=cycle_table,
            )
            self._finish_stream_timer(stream_timer)

        auxiliary["beat_table"] = beat_table
        auxiliary["cycle_table"] = cycle_table
        contexts = (local_context, beat_context, cycle_context, zero_context)
        availability = torch.stack(
            (
                local_available,
                beat_available,
                cycle_available,
                torch.ones_like(false_mask),
            ),
            dim=-1,
        )
        return contexts, availability, auxiliary

    def _fuse_streams(
        self,
        normalized: torch.Tensor,
        contexts,
        availability: torch.Tensor,
    ) -> torch.Tensor:
        gate_logits = self.stream_gate(normalized).float()
        gate_logits = gate_logits.masked_fill(~availability, torch.finfo(torch.float32).min)
        weights = torch.softmax(gate_logits, dim=-1).to(normalized.dtype)
        stacked = torch.stack(contexts, dim=-2)
        fused = (stacked * weights.unsqueeze(-1)).sum(dim=-2)
        self.last_gate_usage = weights.detach().mean(dim=(0, 1))
        return fused

    def _build_prefill_cache(
        self,
        auxiliary: Dict[str, torch.Tensor],
        routing: Dict[str, torch.Tensor],
        in_beat_after: torch.Tensor,
        in_cycle_after: torch.Tensor,
    ) -> Dict[str, Any]:
        batch_size, sequence_length = routing["beat_slot"].shape
        device = routing["beat_slot"].device
        dtype_source = next(iter(auxiliary.values()))

        local_k = dtype_source.new_zeros(
            batch_size, self.number_heads, self.max_tokens_per_beat, self.head_dim
        )
        local_v = torch.zeros_like(local_k)
        local_valid = torch.zeros(
            batch_size, self.max_tokens_per_beat, dtype=torch.bool, device=device
        )
        if self.use_inside_beat_stream:
            all_k = auxiliary["local_k"].reshape(
                batch_size,
                self.max_beat_slots_per_chunk,
                self.number_heads,
                self.max_tokens_per_beat,
                self.head_dim,
            )
            all_v = auxiliary["local_v"].reshape_as(all_k)
            all_valid = auxiliary["local_valid"].view(
                batch_size, self.max_beat_slots_per_chunk, self.max_tokens_per_beat
            )
            active_slot = routing["beat_slot"][:, -1].clamp(
                0, self.max_beat_slots_per_chunk - 1
            )
            batch_index = torch.arange(batch_size, device=device)
            local_k = all_k[batch_index, active_slot] * in_beat_after[:, None, None, None]
            local_v = all_v[batch_index, active_slot] * in_beat_after[:, None, None, None]
            local_valid = all_valid[batch_index, active_slot] & in_beat_after[:, None]

        beat_k = dtype_source.new_zeros(
            batch_size, self.number_heads, self.max_beats_per_cycle, self.head_dim
        )
        beat_v = torch.zeros_like(beat_k)
        beat_valid = torch.zeros(
            batch_size, self.max_beats_per_cycle, dtype=torch.bool, device=device
        )
        if self.use_beat_memory_stream:
            all_k = auxiliary["beat_k"].reshape(
                batch_size,
                self.max_cycles_per_chunk,
                self.number_heads,
                self.max_beats_per_cycle,
                self.head_dim,
            )
            all_v = auxiliary["beat_v"].reshape_as(all_k)
            all_valid = auxiliary["beat_valid"].view(
                batch_size, self.max_cycles_per_chunk, self.max_beats_per_cycle
            )
            active_cycle = routing["cycle_slot"][:, -1].clamp(
                0, self.max_cycles_per_chunk - 1
            )
            batch_index = torch.arange(batch_size, device=device)
            beat_k = all_k[batch_index, active_cycle] * in_cycle_after[:, None, None, None]
            beat_v = all_v[batch_index, active_cycle] * in_cycle_after[:, None, None, None]
            beat_valid = all_valid[batch_index, active_cycle] & in_cycle_after[:, None]

        cycle_k = dtype_source.new_zeros(
            batch_size, self.number_heads, self.max_cycles_per_chunk, self.head_dim
        )
        cycle_v = torch.zeros_like(cycle_k)
        cycle_valid = torch.zeros(
            batch_size, self.max_cycles_per_chunk, dtype=torch.bool, device=device
        )
        if self.use_cycle_memory_stream:
            cycle_k = auxiliary["cycle_k"]
            cycle_v = auxiliary["cycle_v"]
            cycle_valid = auxiliary["cycle_valid"]

        positions = torch.arange(sequence_length, device=device).view(1, -1)
        eob_score = torch.where(
            routing["complete_eob_mask"] & (
                routing["cycle_slot"] == routing["cycle_slot"][:, -1:]
            ),
            positions,
            torch.full_like(positions, -1),
        )
        last_eob_index = eob_score.argmax(dim=1, keepdim=True)
        has_eob = (
            (eob_score.max(dim=1).values >= 0)
            & in_cycle_after
            & self.use_beat_memory_stream
        )
        last_beat_slot = torch.gather(routing["beat_slot"], 1, last_eob_index)
        latest_beat_context = self._gather_context_table(
            auxiliary["beat_table"], last_beat_slot, has_eob[:, None]
        )

        eoc_score = torch.where(
            routing["complete_eoc_mask"],
            positions,
            torch.full_like(positions, -1),
        )
        last_eoc_index = eoc_score.argmax(dim=1, keepdim=True)
        has_eoc = (eoc_score.max(dim=1).values >= 0) & self.use_cycle_memory_stream
        last_cycle_slot = torch.gather(routing["cycle_slot"], 1, last_eoc_index)
        latest_cycle_context = self._gather_context_table(
            auxiliary["cycle_table"], last_cycle_slot, has_eoc[:, None]
        )

        return {
            "local_k": local_k,
            "local_v": local_v,
            "local_valid": local_valid,
            "beat_k": beat_k,
            "beat_v": beat_v,
            "beat_valid": beat_valid,
            "cycle_k": cycle_k,
            "cycle_v": cycle_v,
            "cycle_valid": cycle_valid,
            "latest_beat_context": latest_beat_context,
            "latest_beat_valid": has_eob[:, None],
            "latest_cycle_context": latest_cycle_context,
            "latest_cycle_valid": has_eoc[:, None],
            "rhythm_boundary_state": {
                "in_beat": in_beat_after,
                "in_cycle": in_cycle_after,
            },
        }

    def forward(
        self,
        x: torch.Tensor,
        beat_idx_ids: torch.Tensor,
        routing: Dict[str, torch.Tensor],
        rope_inv_freq: Optional[torch.Tensor],
        return_cache: bool = False,
        in_beat_after: Optional[torch.Tensor] = None,
        in_cycle_after: Optional[torch.Tensor] = None,
    ):
        normalized = self.layer_norm_1(x)
        query, key, value = self._project_qkv(normalized)
        contexts, availability, auxiliary = self._compute_streams(
            query, key, value, beat_idx_ids, routing, rope_inv_freq
        )
        fused = self._fuse_streams(normalized, contexts, availability)
        x = x + self.residual_dropout(self.output_projection(fused))
        x = x + self.multi_layer_perceptron(self.layer_norm_2(x))
        if not return_cache:
            return x
        if in_beat_after is None or in_cycle_after is None:
            raise ValueError("Prefill cache construction requires final boundary state")
        cache = self._build_prefill_cache(
            auxiliary, routing, in_beat_after, in_cycle_after
        )
        return x, cache

    @staticmethod
    def _single_query_sdpa(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        any_valid = valid.any(dim=1, keepdim=True)
        fallback = torch.zeros_like(valid)
        fallback[:, 0] = True
        safe_valid = valid | ((~any_valid) & fallback)
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=safe_valid[:, None, None, :],
            dropout_p=0.0,
            is_causal=False,
        )
        return output * any_valid[:, None, :, None].to(output.dtype)

    @staticmethod
    def _cache_write(
        cache_tensor: torch.Tensor,
        value: torch.Tensor,
        position: torch.Tensor,
        write_mask: torch.Tensor,
    ) -> torch.Tensor:
        length = cache_tensor.size(2)
        position = position.reshape(cache_tensor.size(0))
        write_mask = write_mask.reshape(cache_tensor.size(0))
        one_hot = F.one_hot(position.clamp(0, length - 1), num_classes=length).to(torch.bool)
        one_hot = one_hot & write_mask[:, None]
        return torch.where(
            one_hot[:, None, :, None],
            value.expand(-1, -1, length, -1),
            cache_tensor,
        )

    def forward_with_structured_cache(
        self,
        x: torch.Tensor,
        cache: Dict[str, Any],
        step: Dict[str, torch.Tensor],
        rope_inv_freq: Optional[torch.Tensor],
    ):
        """Process one new token and update fixed structured caches exactly once."""
        if self.training:
            raise RuntimeError("Structured cached decode requires model.eval()")
        if x.size(1) != 1:
            raise ValueError("Structured cached decode accepts exactly one token")
        normalized = self.layer_norm_1(x)
        query, key, value = self._project_qkv(normalized)
        batch_size = x.size(0)
        device = x.device
        zero_context = x.new_zeros(batch_size, 1, self.number_embeddings)
        false_available = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)

        starts_beat = step["starts_beat"].to(torch.bool).reshape(batch_size)
        starts_cycle = step["starts_cycle"].to(torch.bool).reshape(batch_size)
        ends_beat = step["ends_beat"].to(torch.bool).reshape(batch_size)
        closes_beat = step["closes_beat"].to(torch.bool).reshape(batch_size)
        ends_cycle = step["ends_cycle"].to(torch.bool).reshape(batch_size)
        closes_cycle = step["closes_cycle"].to(torch.bool).reshape(batch_size)
        in_beat = step["in_beat"].to(torch.bool).reshape(batch_size)
        reset_local = starts_beat | starts_cycle
        reset_beat = starts_cycle
        cache["local_k"] = cache["local_k"] * (~reset_local)[:, None, None, None]
        cache["local_v"] = cache["local_v"] * (~reset_local)[:, None, None, None]
        cache["local_valid"] = cache["local_valid"] & (~reset_local)[:, None]
        cache["beat_k"] = cache["beat_k"] * (~reset_beat)[:, None, None, None]
        cache["beat_v"] = cache["beat_v"] * (~reset_beat)[:, None, None, None]
        cache["beat_valid"] = cache["beat_valid"] & (~reset_beat)[:, None]
        cache["latest_beat_context"] = cache["latest_beat_context"] * (
            ~reset_beat
        )[:, None, None]
        cache["latest_beat_valid"] = cache["latest_beat_valid"] & (~reset_beat)[:, None]

        local_context, local_available = zero_context, false_available
        if self.use_inside_beat_stream:
            local_position = step["local_beat_position"].to(torch.long).reshape(batch_size)
            local_q = self._rotate(query, local_position, rope_inv_freq)
            local_k = self._rotate(key, local_position, rope_inv_freq)
            cache["local_k"] = self._cache_write(
                cache["local_k"], local_k, local_position, in_beat
            )
            cache["local_v"] = self._cache_write(
                cache["local_v"], value, local_position, in_beat
            )
            local_one_hot = F.one_hot(
                local_position.clamp(0, self.max_tokens_per_beat - 1),
                num_classes=self.max_tokens_per_beat,
            ).to(torch.bool)
            cache["local_valid"] = cache["local_valid"] | (
                local_one_hot & in_beat[:, None]
            )
            local_heads = self._single_query_sdpa(
                local_q, cache["local_k"], cache["local_v"], cache["local_valid"]
            )
            local_context = local_heads.transpose(1, 2).reshape(
                batch_size, 1, self.number_embeddings
            )
            local_available = in_beat[:, None]

        beat_context = (
            cache["latest_beat_context"]
            if self.use_beat_memory_stream
            else zero_context
        )
        beat_available = (
            cache["latest_beat_valid"]
            if self.use_beat_memory_stream
            else false_available
        )
        next_beat_context = beat_context
        next_beat_valid = beat_available
        if self.use_beat_memory_stream:
            beat_position = step["beat_index_in_cycle"].to(torch.long).reshape(batch_size)
            beat_q = self._rotate(query, beat_position, rope_inv_freq)
            beat_k = self._rotate(key, beat_position, rope_inv_freq)
            cache["beat_k"] = self._cache_write(
                cache["beat_k"], beat_k, beat_position, ends_beat
            )
            cache["beat_v"] = self._cache_write(
                cache["beat_v"], value, beat_position, ends_beat
            )
            beat_one_hot = F.one_hot(
                beat_position.clamp(0, self.max_beats_per_cycle - 1),
                num_classes=self.max_beats_per_cycle,
            ).to(torch.bool)
            cache["beat_valid"] = cache["beat_valid"] | (
                beat_one_hot & ends_beat[:, None]
            )
            encoded = self._single_query_sdpa(
                beat_q, cache["beat_k"], cache["beat_v"], cache["beat_valid"]
            ).transpose(1, 2).reshape(batch_size, 1, self.number_embeddings)
            next_beat_context = torch.where(
                ends_beat[:, None, None], encoded, beat_context
            )
            next_beat_valid = beat_available | ends_beat[:, None]

        cycle_context = (
            cache["latest_cycle_context"]
            if self.use_cycle_memory_stream
            else zero_context
        )
        cycle_available = (
            cache["latest_cycle_valid"]
            if self.use_cycle_memory_stream
            else false_available
        )
        next_cycle_context = cycle_context
        next_cycle_valid = cycle_available
        if self.use_cycle_memory_stream:
            cycle_position = step["cycle_ordinal"].to(torch.long).reshape(batch_size)
            cycle_q = self._rotate(query, cycle_position, rope_inv_freq)
            cycle_k = self._rotate(key, cycle_position, rope_inv_freq)
            cache["cycle_k"] = self._cache_write(
                cache["cycle_k"], cycle_k, cycle_position, ends_cycle
            )
            cache["cycle_v"] = self._cache_write(
                cache["cycle_v"], value, cycle_position, ends_cycle
            )
            cycle_one_hot = F.one_hot(
                cycle_position.clamp(0, self.max_cycles_per_chunk - 1),
                num_classes=self.max_cycles_per_chunk,
            ).to(torch.bool)
            cache["cycle_valid"] = cache["cycle_valid"] | (
                cycle_one_hot & ends_cycle[:, None]
            )
            encoded = self._single_query_sdpa(
                cycle_q, cache["cycle_k"], cache["cycle_v"], cache["cycle_valid"]
            ).transpose(1, 2).reshape(batch_size, 1, self.number_embeddings)
            next_cycle_context = torch.where(
                ends_cycle[:, None, None], encoded, cycle_context
            )
            next_cycle_valid = cycle_available | ends_cycle[:, None]

        availability = torch.stack(
            (
                local_available.squeeze(-1),
                beat_available.squeeze(-1),
                cycle_available.squeeze(-1),
                torch.ones(batch_size, dtype=torch.bool, device=device),
            ),
            dim=-1,
        ).unsqueeze(1)
        fused = self._fuse_streams(
            normalized,
            (local_context, beat_context, cycle_context, zero_context),
            availability,
        )
        x = x + self.residual_dropout(self.output_projection(fused))
        x = x + self.multi_layer_perceptron(self.layer_norm_2(x))

        cache["latest_beat_context"] = next_beat_context
        cache["latest_beat_valid"] = next_beat_valid
        cache["latest_cycle_context"] = next_cycle_context
        cache["latest_cycle_valid"] = next_cycle_valid
        clear_after_beat = closes_beat | closes_cycle
        cache["local_k"] = cache["local_k"] * (~clear_after_beat)[:, None, None, None]
        cache["local_v"] = cache["local_v"] * (~clear_after_beat)[:, None, None, None]
        cache["local_valid"] = cache["local_valid"] & (~clear_after_beat)[:, None]
        clear_after_cycle = closes_cycle
        cache["beat_k"] = cache["beat_k"] * (~clear_after_cycle)[:, None, None, None]
        cache["beat_v"] = cache["beat_v"] * (~clear_after_cycle)[:, None, None, None]
        cache["beat_valid"] = cache["beat_valid"] & (~clear_after_cycle)[:, None]
        cache["latest_beat_context"] = cache["latest_beat_context"] * (
            ~clear_after_cycle
        )[:, None, None]
        cache["latest_beat_valid"] = cache["latest_beat_valid"] & (
            ~clear_after_cycle
        )[:, None]
        cache["rhythm_boundary_state"] = {
            key: value.detach() for key, value in step.items()
        }
        return x, cache


class GPT(nn.Module):
    def __init__(
        self,
        vocab: List[str],
        context_size: int,
        number_of_layers: int,
        number_of_heads: int,
        number_of_embeddings: int,
        mlp_ratio: int,
        dropout: float,
        tie_weights: bool = True,
        use_type_embeddings: bool = True,
        concat_type_embeddings: bool = False,
        activation_name: str = "gelu",
        position_encoding: str = "sinusoidal",
        rope_theta: float = 10_000.0,
        rope_fraction: float = 1.0,
        max_beat_positions: Optional[int] = None,
        max_beats_per_cycle: Optional[int] = None,
        attention_layout: str = "global",
        use_inside_beat_stream: bool = True,
        use_beat_memory_stream: bool = True,
        use_cycle_memory_stream: bool = True,
        max_tokens_per_beat: Optional[int] = None,
        max_beat_slots_per_chunk: Optional[int] = None,
        max_cycles_per_chunk: Optional[int] = None,
    ):
        super().__init__()
        token_to_id = build_token_to_id(vocab)
        self.vocab_size = len(token_to_id)
        self.context_size = int(context_size)
        self.position_encoding = str(position_encoding).strip().lower()
        if self.position_encoding not in {"sinusoidal", "rope"}:
            raise ValueError("position_encoding must be 'sinusoidal' or 'rope'")
        self.rope_theta = float(rope_theta)
        self.rope_fraction = float(rope_fraction)
        if not math.isfinite(self.rope_theta) or self.rope_theta <= 0:
            raise ValueError("rope_theta must be finite and positive")
        if not math.isfinite(self.rope_fraction) or not 0 < self.rope_fraction <= 1:
            raise ValueError("rope_fraction must be in (0, 1]")
        if number_of_embeddings % number_of_heads != 0:
            raise ValueError("number_of_embeddings must be divisible by number_of_heads")
        head_dim = number_of_embeddings // number_of_heads
        rotary_dim = int(head_dim * self.rope_fraction)
        rotary_dim -= rotary_dim % 2
        if self.position_encoding == "rope" and rotary_dim < 2:
            raise ValueError("rope_fraction leaves no even rotary dimensions")
        self.rotary_dim = rotary_dim if self.position_encoding == "rope" else 0

        self.attention_layout = str(attention_layout).strip().lower()
        self.resolved_attention_layout = resolve_attention_layout(
            self.attention_layout, number_of_layers
        )
        self.has_structured_layers = "structured" in self.resolved_attention_layout
        self.max_tokens_per_beat = int(max_tokens_per_beat or 0)
        self.max_beat_slots_per_chunk = int(max_beat_slots_per_chunk or 0)
        self.max_cycles_per_chunk = int(max_cycles_per_chunk or 0)
        if self.has_structured_layers:
            if self.position_encoding != "rope":
                raise ValueError(
                    "alternating_global_structured requires position_encoding='rope' "
                    "for the stream-specific coordinate systems"
                )

        self.use_type_embeddings = bool(use_type_embeddings)
        self.concat_type_embeddings = bool(concat_type_embeddings) and self.use_type_embeddings
        meta = build_token_metadata_buffers(
            token_to_id, max_beat_positions=max_beat_positions
        )
        self.n_types = meta["n_types"]
        self.max_beat_positions = meta["max_beat_positions"]
        self.TYPE_SUBD = meta["TYPE_SUBD"]
        self.register_buffer("token_type_ids", meta["token_type_ids"], persistent=True)
        self.register_buffer("is_hit_vocab", meta["is_hit"], persistent=True)
        self.register_buffer("hit_atom_counts_vocab", meta["hit_atom_counts"], persistent=True)
        self.register_buffer("is_sob_vocab", meta["is_sob"], persistent=True)

        self.token_embeddings = nn.Embedding(self.vocab_size, number_of_embeddings)
        if self.use_type_embeddings:
            self.type_embeddings = nn.Embedding(self.n_types, number_of_embeddings)
            self.type_concat_projection = (
                nn.Linear(2 * number_of_embeddings, number_of_embeddings)
                if self.concat_type_embeddings
                else None
            )
        else:
            self.type_embeddings = None
            self.type_concat_projection = None

        self.beat_positional_embeddings = nn.Embedding(
            self.max_beat_positions, number_of_embeddings
        )
        self.id_soc = token_to_id.get("<SOC>")
        self.id_eoc = token_to_id.get("<EOC>")
        self.id_sob = token_to_id.get("<SOB>")
        self.id_eob = token_to_id.get("<EOB>")
        if max_beats_per_cycle is None or int(max_beats_per_cycle) <= 0:
            raise ValueError("max_beats_per_cycle is required and must be positive")
        self.max_beats_per_cycle = int(max_beats_per_cycle)
        self.beat_idx_embeddings = nn.Embedding(
            self.max_beats_per_cycle, number_of_embeddings
        )

        if self.position_encoding == "sinusoidal":
            self.register_buffer(
                "fixed_sinusoidal_pe",
                build_sinusoidal_table(context_size, number_of_embeddings),
                persistent=False,
            )
            self.register_buffer("rope_inv_freq", None, persistent=False)
        else:
            inv_freq = 1.0 / (
                self.rope_theta
                ** (
                    torch.arange(0, self.rotary_dim, 2, dtype=torch.float32)
                    / self.rotary_dim
                )
            )
            self.register_buffer("fixed_sinusoidal_pe", None, persistent=False)
            self.register_buffer("rope_inv_freq", inv_freq, persistent=False)

        self.dropout = nn.Dropout(dropout)
        block_list = []
        for block_kind in self.resolved_attention_layout:
            if block_kind == "global":
                block = Transformer_Attention_Block(
                    number_embeddings=number_of_embeddings,
                    number_heads=number_of_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    activation_name=activation_name,
                )
            else:
                block = Structured_Attention_Block(
                    number_embeddings=number_of_embeddings,
                    number_heads=number_of_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    activation_name=activation_name,
                    max_tokens_per_beat=int(max_tokens_per_beat or 0),
                    max_beat_slots_per_chunk=int(max_beat_slots_per_chunk or 0),
                    max_beats_per_cycle=self.max_beats_per_cycle,
                    max_cycles_per_chunk=int(max_cycles_per_chunk or 0),
                    use_inside_beat_stream=use_inside_beat_stream,
                    use_beat_memory_stream=use_beat_memory_stream,
                    use_cycle_memory_stream=use_cycle_memory_stream,
                )
            block_list.append(block)
        self.blocks = nn.ModuleList(block_list)
        self.layer_normalization = nn.LayerNorm(number_of_embeddings)
        self.head = nn.Linear(number_of_embeddings, self.vocab_size, bias=False)
        if tie_weights:
            self.head.weight = self.token_embeddings.weight

    def _rope_cos_sin(self, position_offset, sequence_length, device, dtype):
        if self.position_encoding != "rope":
            return None, None
        if position_offset < 0 or position_offset + sequence_length > self.context_size:
            raise ValueError("Global RoPE position exceeds context_size")
        positions = torch.arange(
            position_offset,
            position_offset + sequence_length,
            device=device,
            dtype=torch.long,
        ).unsqueeze(0)
        return rope_cos_sin_for_positions(positions, self.rope_inv_freq, dtype)

    def _compute_rhythm_ids(self, batch_of_token_ids, type_ids=None):
        B, T = batch_of_token_ids.shape
        device = batch_of_token_ids.device
        if type_ids is None:
            type_ids = self.token_type_ids[batch_of_token_ids]
        is_sob = self.is_sob_vocab[batch_of_token_ids]
        is_hit = self.is_hit_vocab[batch_of_token_ids]
        hit_atom_counts = self.hit_atom_counts_vocab[batch_of_token_ids]
        is_soc = (
            batch_of_token_ids == self.id_soc
            if self.id_soc is not None
            else torch.zeros((B, T), dtype=torch.bool, device=device)
        )
        is_eoc = (
            batch_of_token_ids == self.id_eoc
            if self.id_eoc is not None
            else torch.zeros((B, T), dtype=torch.bool, device=device)
        )
        is_subd = type_ids == self.TYPE_SUBD
        is_delim = type_ids == 0
        beat_idx_ids = torch.empty((B, T), dtype=torch.long, device=device)
        beat_pos_ids = torch.empty((B, T), dtype=torch.long, device=device)
        zeros = torch.zeros(B, dtype=torch.long, device=device)
        in_cycle = torch.zeros(B, dtype=torch.bool, device=device)
        beat = torch.zeros(B, dtype=torch.long, device=device)
        started = torch.zeros(B, dtype=torch.bool, device=device)
        current_position = torch.zeros(B, dtype=torch.long, device=device)
        in_beat = torch.zeros(B, dtype=torch.bool, device=device)
        for position in range(T):
            sob = is_sob[:, position]
            soc = is_soc[:, position]
            eoc = is_eoc[:, position]
            start_cycle = soc | ((~in_cycle) & sob)
            in_cycle = torch.where(start_cycle, True, in_cycle)
            in_cycle = torch.where(eoc, False, in_cycle)
            beat = torch.where(start_cycle, zeros, beat)
            started = torch.where(start_cycle, False, started)
            increment = sob & in_cycle
            beat = torch.where(increment & started, beat + 1, beat)
            started = torch.where(increment, True, started)
            beat_idx_ids[:, position] = torch.where(in_cycle, beat, zeros)
            reset = sob | is_subd[:, position]
            boundary = is_delim[:, position] & (~sob)
            in_beat = torch.where(reset, True, in_beat)
            in_beat = torch.where(boundary, False, in_beat)
            current_position = torch.where(reset | boundary, zeros, current_position)
            beat_pos_ids[:, position] = torch.where(in_beat, current_position, zeros)
            current_position = current_position + torch.where(
                in_beat, hit_atom_counts[:, position], zeros
            )
        beat_idx_ids.clamp_(0, self.max_beats_per_cycle - 1)
        beat_pos_ids.clamp_(0, self.max_beat_positions - 1)
        return beat_idx_ids, beat_pos_ids, is_hit

    def _embed_tokens(
        self,
        batch_of_token_ids,
        beat_idx_ids=None,
        beat_pos_ids=None,
        position_offset=0,
    ):
        _, sequence_length = batch_of_token_ids.shape
        if position_offset < 0 or position_offset + sequence_length > self.context_size:
            raise ValueError("Embedding position exceeds context_size")
        token_embedding = self.token_embeddings(batch_of_token_ids)
        type_ids = self.token_type_ids[batch_of_token_ids]
        if self.use_type_embeddings:
            type_embedding = self.type_embeddings(type_ids)
            base = (
                self.type_concat_projection(torch.cat((token_embedding, type_embedding), dim=-1))
                if self.concat_type_embeddings
                else token_embedding + type_embedding
            )
        else:
            base = token_embedding
        if beat_idx_ids is None or beat_pos_ids is None:
            beat_idx_ids, beat_pos_ids, _ = self._compute_rhythm_ids(
                batch_of_token_ids, type_ids
            )
        beat_idx_ids = beat_idx_ids.to(batch_of_token_ids.device, torch.long, non_blocking=True)
        beat_pos_ids = beat_pos_ids.to(batch_of_token_ids.device, torch.long, non_blocking=True)
        beat_idx_ids = beat_idx_ids.clamp(0, self.max_beats_per_cycle - 1)
        beat_pos_ids = beat_pos_ids.clamp(0, self.max_beat_positions - 1)
        is_hit = self.is_hit_vocab[batch_of_token_ids]
        positional = self.beat_positional_embeddings(beat_pos_ids) * is_hit.unsqueeze(-1)
        if self.position_encoding == "sinusoidal":
            positional = positional + self.fixed_sinusoidal_pe[
                position_offset : position_offset + sequence_length
            ].unsqueeze(0)
        return self.dropout(base + positional + self.beat_idx_embeddings(beat_idx_ids))

    def _prepare_routing(self, routing, device):
        if routing is None:
            return None
        missing = [field for field in STRUCTURED_ROUTING_FIELDS if field not in routing]
        if missing:
            raise KeyError(f"Structured routing is missing fields: {missing}")
        prepared = {
            field: routing[field].to(
                device=device,
                dtype=torch.bool if field.endswith("_mask") else torch.long,
                non_blocking=True,
            )
            for field in STRUCTURED_ROUTING_FIELDS
        }
        for field, limit in (
            ("beat_slot", self.max_beat_slots_per_chunk),
            ("cycle_slot", self.max_cycles_per_chunk),
            ("local_beat_position", self.max_tokens_per_beat),
        ):
            if torch.any(prepared[field] >= limit):
                raise ValueError(f"Structured {field} exceeds model capacity {limit}")
        return prepared

    def forward(
        self,
        batch_of_token_ids,
        beat_idx_ids=None,
        beat_pos_ids=None,
        structured_routing=None,
    ):
        _, sequence_length = batch_of_token_ids.shape
        if sequence_length > self.context_size:
            raise ValueError("Sequence length exceeds context_size")
        if beat_idx_ids is None or beat_pos_ids is None:
            beat_idx_ids, beat_pos_ids, _ = self._compute_rhythm_ids(batch_of_token_ids)
        routing = (
            self._prepare_routing(structured_routing, batch_of_token_ids.device)
            if self.has_structured_layers
            else None
        )
        if self.has_structured_layers and routing is None:
            raise ValueError(
                "alternating_global_structured requires structured routing arrays"
            )
        x = self._embed_tokens(batch_of_token_ids, beat_idx_ids, beat_pos_ids, 0)
        global_cos, global_sin = self._rope_cos_sin(0, sequence_length, x.device, x.dtype)
        beat_idx_ids = beat_idx_ids.to(x.device, torch.long, non_blocking=True)
        for kind, block in zip(self.resolved_attention_layout, self.blocks):
            if kind == "global":
                x = block(x, rope_cos=global_cos, rope_sin=global_sin)
            else:
                x = block(x, beat_idx_ids, routing, self.rope_inv_freq)
        return self.head(self.layer_normalization(x))

    def prefill_with_kv_cache(
        self,
        batch_of_token_ids,
        beat_idx_ids=None,
        beat_pos_ids=None,
        structured_routing=None,
    ):
        _, sequence_length = batch_of_token_ids.shape
        if not 0 < sequence_length <= self.context_size:
            raise ValueError("Invalid prefill length")
        if beat_idx_ids is None or beat_pos_ids is None:
            beat_idx_ids, beat_pos_ids, _ = self._compute_rhythm_ids(batch_of_token_ids)
        routing = (
            self._prepare_routing(structured_routing, batch_of_token_ids.device)
            if self.has_structured_layers
            else None
        )
        if self.has_structured_layers and routing is None:
            raise ValueError("Structured prefill requires routing arrays")
        x = self._embed_tokens(batch_of_token_ids, beat_idx_ids, beat_pos_ids, 0)
        global_cos, global_sin = self._rope_cos_sin(0, sequence_length, x.device, x.dtype)
        beat_idx_ids = beat_idx_ids.to(x.device, torch.long, non_blocking=True)
        in_beat_after = routing["beat_slot"][:, -1] >= 0 if routing is not None else None
        in_cycle_after = routing["cycle_slot"][:, -1] >= 0 if routing is not None else None
        if routing is not None:
            if self.id_eob is not None:
                in_beat_after &= batch_of_token_ids[:, -1] != self.id_eob
            if self.id_eoc is not None:
                in_cycle_after &= batch_of_token_ids[:, -1] != self.id_eoc

        caches = []
        for kind, block in zip(self.resolved_attention_layout, self.blocks):
            if kind == "global":
                x, cache = block.forward_with_kv_cache(
                    x,
                    kv_cache=None,
                    max_cache_len=self.context_size,
                    rope_cos=global_cos,
                    rope_sin=global_sin,
                )
            else:
                x, cache = block(
                    x,
                    beat_idx_ids,
                    routing,
                    self.rope_inv_freq,
                    return_cache=True,
                    in_beat_after=in_beat_after,
                    in_cycle_after=in_cycle_after,
                )
            caches.append({"kind": kind, "state": cache})
        return self.head(self.layer_normalization(x)), caches

    def decode_one_with_kv_cache(
        self,
        batch_of_token_ids,
        kv_caches,
        beat_idx_ids,
        beat_pos_ids,
        structured_step=None,
    ):
        if batch_of_token_ids.shape[1] != 1:
            raise ValueError("decode_one_with_kv_cache expects [B, 1]")
        if len(kv_caches) != len(self.blocks):
            raise ValueError("KV cache layer count does not match model")
        global_cache = next(cache["state"] for cache in kv_caches if cache["kind"] == "global")
        position = int(global_cache[2])
        if position >= self.context_size:
            raise ValueError("KV cache is full; refresh both cache types")
        if self.has_structured_layers and structured_step is None:
            raise ValueError("Structured cached decode requires structured_step metadata")
        if structured_step is not None:
            structured_step = {
                key: value.to(batch_of_token_ids.device, non_blocking=True)
                for key, value in structured_step.items()
            }
            for field, limit in (
                ("cycle_ordinal", self.max_cycles_per_chunk),
                ("beat_index_in_cycle", self.max_beats_per_cycle),
                ("local_beat_position", self.max_tokens_per_beat),
            ):
                if torch.any(structured_step[field] >= limit):
                    raise ValueError(f"Structured {field} exceeds model capacity {limit}")
        x = self._embed_tokens(
            batch_of_token_ids, beat_idx_ids, beat_pos_ids, position
        )
        global_cos, global_sin = self._rope_cos_sin(position, 1, x.device, x.dtype)
        updated = []
        for kind, block, wrapped_cache in zip(
            self.resolved_attention_layout, self.blocks, kv_caches
        ):
            if kind == "global":
                x, state = block.forward_with_kv_cache(
                    x,
                    kv_cache=wrapped_cache["state"],
                    rope_cos=global_cos,
                    rope_sin=global_sin,
                )
            else:
                x, state = block.forward_with_structured_cache(
                    x, wrapped_cache["state"], structured_step, self.rope_inv_freq
                )
            updated.append({"kind": kind, "state": state})
        return self.head(self.layer_normalization(x)), updated
