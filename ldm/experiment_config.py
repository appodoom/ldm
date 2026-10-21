"""Typed, serializable settings for one atomic LDM experiment."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from types import UnionType
from typing import Union, get_args, get_origin, get_type_hints

from .constants import NORMAL_DATA, DECORATED_DATA
from .validation import ValidationError, format_errors, validate_rules


class DatasetVariant(str, Enum):
    WITH_SUBD = "with_subd"
    NO_SUBD = "no_subd"


class DecorationMode(str, Enum):
    WITHOUT_DECORATION = NORMAL_DATA
    WITH_DECORATION = DECORATED_DATA


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


class ConfigValidationError(ValidationError):
    """Invalid values, reported together with their field names."""

    heading = "Invalid configuration"


class ConfigTypeError(TypeError):
    """Invalid field types, reported before arithmetic or cross-field checks."""

    def __init__(self, errors: dict[str, str]):
        self.errors = errors
        super().__init__(format_errors(errors, "Invalid configuration"))


def _matches_type(value, expected) -> bool:
    if get_origin(expected) in (Union, UnionType):
        return any(_matches_type(value, option) for option in get_args(expected))
    if expected is float:
        return type(value) in (int, float)
    if expected in (int, bool, str):
        return type(value) is expected
    return isinstance(value, expected)


class _ValidatedConfig:
    """One validation path; each settings class declares its own value rules."""

    def __post_init__(self) -> None:
        hints = get_type_hints(type(self))
        errors = {item.name: f"expected {hints[item.name]}, got {type(getattr(self, item.name)).__name__}"
                  for item in fields(self) if not _matches_type(getattr(self, item.name), hints[item.name])}
        if errors:
            raise ConfigTypeError(errors)
        validate_rules(self._rules(), error_type=ConfigValidationError)

    def _rules(self) -> dict[str, tuple[bool, str]]:
        return {}


@dataclass(frozen=True)
class StructuredStreams(_ValidatedConfig):
    inside_beat: bool = True
    beat_memory: bool = True
    cycle_memory: bool = True


@dataclass(frozen=True)
class DataConfig(_ValidatedConfig):
    variant: DatasetVariant = DatasetVariant.WITH_SUBD
    context_size: int = 1024
    stride: int | None = None
    max_beats_per_cycle: int = 16
    decoration: DecorationMode = DecorationMode.WITHOUT_DECORATION

    def _rules(self):
        return {
            "context_size": (self.context_size >= 2, "must be at least 2"),
            "stride": (1 <= self.resolved_stride <= self.context_size, "must be between 1 and context_size"),
            "max_beats_per_cycle": (self.max_beats_per_cycle >= 1, "must be positive"),
        }

    @property
    def resolved_stride(self) -> int:
        return self.stride if self.stride is not None else max(1, self.context_size // 2)


@dataclass(frozen=True)
class ModelConfig(_ValidatedConfig):
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

    def _rules(self):
        head_dim = self.embeddings // self.heads if self.heads > 0 else 0
        uses_rope = self.position is PositionEncoding.ROPE
        structured = self.attention is AttentionLayout.ALTERNATING_GLOBAL_STRUCTURED
        return {
            "layers": (self.layers >= 1, "must be positive"),
            "heads": (self.heads >= 1, "must be positive"),
            "embeddings": (self.embeddings >= 2 and self.heads > 0 and self.embeddings % self.heads == 0,
                           "must be at least 2 and divisible by heads"),
            "mlp_ratio": (self.mlp_ratio >= 1, "must be positive"),
            "dropout": (math.isfinite(self.dropout) and 0 <= self.dropout < 1, "must be finite in [0, 1)"),
            "rope_theta": (math.isfinite(self.rope_theta) and self.rope_theta > 0, "must be positive and finite"),
            "rope_fraction": (math.isfinite(self.rope_fraction) and 0 < self.rope_fraction <= 1,
                              "must be finite in (0, 1]"),
            "rotary_dimensions": (not uses_rope or (math.isfinite(self.rope_fraction) and
                                   head_dim * self.rope_fraction >= 2), "RoPE needs at least two dimensions per head"),
            "attention": (not structured or (self.layers >= 2 and uses_rope),
                          "structured attention requires at least two layers and RoPE"),
        }


@dataclass(frozen=True)
class TrainConfig(_ValidatedConfig):
    epochs: int = 10
    batch_size: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    amp: bool = True
    allow_tf32: bool = True
    num_workers: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
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

    def _rules(self):
        positive_counts = ("epochs", "batch_size", "save_every_steps", "log_every_steps",
                           "prefetch_factor", "early_stopping_patience")
        nonnegative_counts = ("num_workers", "early_stopping_warmup_evals")
        nonnegative_values = ("weight_decay", "grad_clip", "early_stopping_min_delta")
        return {
            **{name: (getattr(self, name) >= 1, "must be positive") for name in positive_counts},
            **{name: (getattr(self, name) >= 0, "must be nonnegative") for name in nonnegative_counts},
            **{name: (math.isfinite(getattr(self, name)) and getattr(self, name) >= 0,
                      "must be nonnegative and finite") for name in nonnegative_values},
            "learning_rate": (math.isfinite(self.learning_rate) and self.learning_rate > 0,
                              "must be positive and finite"),
            "seed": (0 <= self.seed < 2**32, "must be between 0 and 2**32 - 1"),
            "max_steps": (self.max_steps is None or self.max_steps >= 1, "must be None or positive"),
            "max_train_time_minutes": (self.max_train_time_minutes is None or
                (math.isfinite(self.max_train_time_minutes) and self.max_train_time_minutes > 0),
                "must be None or positive and finite"),
            "divergence_threshold": (self.divergence_threshold is None or
                (math.isfinite(self.divergence_threshold) and self.divergence_threshold >= 0),
                "must be None or nonnegative and finite"),
        }


@dataclass(frozen=True)
class EvalConfig(_ValidatedConfig):
    batch_size: int = 4
    every_steps: int = 200
    save_plots: bool = True

    def _rules(self):
        return {"batch_size": (self.batch_size >= 1, "must be positive"),
                "every_steps": (self.every_steps >= 1, "must be positive")}


@dataclass(frozen=True)
class RunConfig(_ValidatedConfig):
    name: str | None = None
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)

    def _rules(self):
        return {"name": (self.name is None or re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,79}", self.name) is not None,
                         "must start with a letter and contain 1-80 letters, digits, underscores or hyphens")}

    def to_dict(self) -> dict:
        return json.loads(json.dumps(asdict(self)))

    @classmethod
    def from_dict(cls, payload: dict) -> "RunConfig":
        data = dict(payload["data"])
        data["decoration"] = DecorationMode(data["decoration"])
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
