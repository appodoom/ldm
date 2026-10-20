"""Typed, serializable settings for one atomic LDM experiment."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from enum import Enum


class DatasetVariant(str, Enum):
    WITH_SUBD = "with_subd"
    NO_SUBD = "no_subd"


class AttentionLayout(str, Enum):
    GLOBAL = "global"
    ALTERNATING_GLOBAL_STRUCTURED = "alternating_global_structured"


class Activation(str, Enum):
    TANH = "tanh"
    GELU = "gelu"
    RELU = "relu"
    SILU = "silu"
    LEAKY_RELU = "leaky_relu"


class PositionEncoding(str, Enum):
    ROPE = "rope"
    SINUSOIDAL = "sinusoidal"


@dataclass(frozen=True)
class StructuredStreams:
    inside_beat: bool = True
    beat_memory: bool = True
    cycle_memory: bool = True


@dataclass(frozen=True)
class DataConfig:
    variant: DatasetVariant = DatasetVariant.WITH_SUBD
    context_size: int = 1024
    stride: int | None = None
    max_beats_per_cycle: int = 16
    include_decorated: bool = True
    decorated_weight: float = 1.0

    @property
    def resolved_stride(self) -> int:
        return self.stride if self.stride is not None else max(1, self.context_size // 2)


@dataclass(frozen=True)
class ModelConfig:
    layers: int = 4
    heads: int = 4
    embeddings: int = 256
    mlp_ratio: int = 4
    dropout: float = 0.1
    attention: AttentionLayout = AttentionLayout.ALTERNATING_GLOBAL_STRUCTURED
    activation: Activation = Activation.TANH
    position: PositionEncoding = PositionEncoding.ROPE
    rope_theta: float = 10000.0
    rope_fraction: float = 1.0
    tie_weights: bool = True
    use_type_embeddings: bool = True
    concat_type_embeddings: bool = False
    streams: StructuredStreams = field(default_factory=StructuredStreams)


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 10
    batch_size: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    amp: bool = True
    seed: int = 1337
    save_every_steps: int = 200
    log_every_steps: int = 50
    max_steps: int | None = None
    max_train_time_minutes: float | None = None
    stop_on_nonfinite_loss: bool = True
    divergence_threshold: float | None = None
    early_stopping: bool = False
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.001
    early_stopping_warmup_evals: int = 0


@dataclass(frozen=True)
class EvalConfig:
    batch_size: int = 4
    every_steps: int = 200
    save_plots: bool = True


@dataclass(frozen=True)
class RunConfig:
    name: str | None = None
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)

    def __post_init__(self) -> None:
        if self.name is not None and not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,79}", self.name):
            raise ValueError("Custom run name needs 1-80 letters, digits, underscores or hyphens")
        d, m, t, e = self.data, self.model, self.train, self.evaluation
        for value, expected in ((d.variant, DatasetVariant), (m.attention, AttentionLayout),
                                (m.activation, Activation), (m.position, PositionEncoding)):
            if not isinstance(value, expected):
                raise TypeError(f"Use {expected.__name__} choices from ldm.experiment_config")
        if not isinstance(m.streams, StructuredStreams):
            raise TypeError("Model streams must be StructuredStreams")
        if d.context_size < 2 or not 1 <= d.resolved_stride <= d.context_size or d.max_beats_per_cycle < 1:
            raise ValueError("Invalid dataset capacities or stride")
        if not math.isfinite(d.decorated_weight) or d.decorated_weight <= 0:
            raise ValueError("decorated_weight must be positive and finite")
        if m.layers < 1 or m.heads < 1 or m.embeddings < 2 or m.embeddings % m.heads:
            raise ValueError("Embeddings must be divisible by positive head count")
        if m.mlp_ratio < 1 or not math.isfinite(m.dropout) or not 0 <= m.dropout < 1:
            raise ValueError("Invalid MLP ratio or dropout")
        if m.position is PositionEncoding.ROPE and (m.embeddings // m.heads) < 2:
            raise ValueError("RoPE requires at least two dimensions per head")
        if not math.isfinite(m.rope_theta) or m.rope_theta <= 0 or not 0 < m.rope_fraction <= 1:
            raise ValueError("Invalid RoPE settings")
        if m.position is PositionEncoding.ROPE and int((m.embeddings // m.heads) * m.rope_fraction) < 2:
            raise ValueError("RoPE fraction leaves fewer than two dimensions per head")
        if m.attention is AttentionLayout.ALTERNATING_GLOBAL_STRUCTURED:
            if m.layers < 2 or m.position is not PositionEncoding.ROPE:
                raise ValueError("Structured attention needs at least two layers and RoPE")
        if min(t.epochs, t.batch_size, t.save_every_steps, t.log_every_steps, e.batch_size, e.every_steps) < 1:
            raise ValueError("Training and evaluation counts must be positive")
        for name, value in (("learning_rate", t.learning_rate), ("weight_decay", t.weight_decay),
                            ("grad_clip", t.grad_clip), ("early_stopping_min_delta", t.early_stopping_min_delta)):
            if not math.isfinite(value) or value < 0 or (name == "learning_rate" and value == 0):
                raise ValueError(f"Invalid {name}")
        if t.early_stopping_patience < 1 or t.early_stopping_warmup_evals < 0:
            raise ValueError("Invalid early stopping settings")
        if t.max_steps is not None and t.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if t.max_train_time_minutes is not None and (
            not math.isfinite(t.max_train_time_minutes) or t.max_train_time_minutes <= 0
        ):
            raise ValueError("max_train_time_minutes must be positive and finite")
        if t.divergence_threshold is not None and (
            not math.isfinite(t.divergence_threshold) or t.divergence_threshold < 0
        ):
            raise ValueError("Invalid divergence threshold")

    def to_dict(self) -> dict:
        return json.loads(json.dumps(asdict(self)))

    @classmethod
    def from_dict(cls, payload: dict) -> "RunConfig":
        data = dict(payload["data"])
        data["variant"] = DatasetVariant(data["variant"])
        model = dict(payload["model"])
        model["attention"] = AttentionLayout(model["attention"])
        model["activation"] = Activation(model["activation"])
        model["position"] = PositionEncoding(model["position"])
        model["streams"] = StructuredStreams(**model["streams"])
        return cls(payload.get("name"), DataConfig(**data), ModelConfig(**model),
                   TrainConfig(**payload["train"]), EvalConfig(**payload["evaluation"]))

    @property
    def fingerprint(self) -> str:
        settings = self.to_dict()
        settings.pop("name", None)
        raw = json.dumps(settings, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()
