"""Configuration errors are early, grouped, and safe for existing run metadata."""

import unittest

from ldm.experiment_config import (
    ConfigTypeError, ConfigValidationError, DataConfig, EvalConfig, DecorationMode,
    ModelConfig, RunConfig, StructuredStreams, TrainConfig,
)


class ConfigValidationTests(unittest.TestCase):
    def test_shared_validation_collects_errors_and_preserves_value_error_api(self):
        from ldm.validation import ValidationError, validate_rules
        validate_rules({"ok": (True, "unused")})
        with self.assertRaises(ValidationError) as caught:
            validate_rules({"context_size": (False, "too small"), "stride": (False, "too large")})
        self.assertIsInstance(caught.exception, ValueError)
        self.assertEqual(caught.exception.errors, {"context_size": "too small", "stride": "too large"})

    def test_reports_multiple_value_errors_in_one_config(self):
        with self.assertRaises(ConfigValidationError) as caught:
            TrainConfig(epochs=0, batch_size=-1, learning_rate=float("nan"), num_workers=-1)
        self.assertEqual(set(caught.exception.errors), {"epochs", "batch_size", "learning_rate", "num_workers"})
        self.assertIn("epochs: must be positive", str(caught.exception))

    def test_rejects_wrong_types_before_computing_rules(self):
        cases = (
            lambda: ModelConfig(heads="four", embeddings=16.5),
            lambda: ModelConfig(streams={}),
            lambda: DataConfig(variant="with_subd"),
            lambda: TrainConfig(batch_size=True),
            lambda: TrainConfig(amp="yes"),
            lambda: EvalConfig(every_steps=1.5),
            lambda: StructuredStreams(inside_beat=1),
            lambda: RunConfig(data={}),
        )
        for constructor in cases:
            with self.subTest(constructor=constructor), self.assertRaises(ConfigTypeError):
                constructor()

    def test_invalid_dimensions_and_optional_limits_have_clear_errors(self):
        cases = (
            lambda: ModelConfig(heads=0),
            lambda: ModelConfig(rope_fraction=float("nan")),
            lambda: DataConfig(context_size=1, stride=2),
            lambda: TrainConfig(max_steps=0),
            lambda: TrainConfig(max_train_time_minutes=float("inf")),
            lambda: TrainConfig(seed=-1),
            lambda: DataConfig(context_size=0),
            lambda: RunConfig(name="bad name"),
        )
        for constructor in cases:
            with self.subTest(constructor=constructor), self.assertRaises(ConfigValidationError):
                constructor()

    def test_round_trip_distinguishes_sources(self):
        config = RunConfig()
        self.assertEqual(RunConfig.from_dict(config.to_dict()), config)
        decorated = RunConfig(data=DataConfig(decoration=DecorationMode.WITH_DECORATION))
        self.assertNotEqual(config.fingerprint, decorated.fingerprint)
        TrainConfig(learning_rate=1, num_workers=0, max_steps=None)


if __name__ == "__main__":
    unittest.main()
