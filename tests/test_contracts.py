"""Contract tests that run without Torch or network access."""

from __future__ import annotations

import tempfile
import json
import sys
import types
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
import numpy as np

from ldm import cli
from ldm.cycle_collector import CycleCollector, IncompleteGenerationError
from ldm.derbake_ldm import Beat, RhythmSequence, parse_derbake, write_derbake
from ldm.schema_ldm import RhythmSchema
from ldm.splits import split_cycle_ranges
from ldm.vocabulary import AtomicVocabulary, atomic_vocabulary
from ldm.dataset_ldm import prepare_dataset
from ldm.experiment_config import DataConfig, DatasetVariant, DecorationMode, RunConfig
from ldm.experiment_workflow import check_run_name, make_run_id, prepare_or_load_dataset
from ldm.inference_ldm import InferenceConfig
from ldm.dotderbake import render_wav
from dataclasses import replace


class CliDispatchTests(unittest.TestCase):
    def test_run_path_does_not_shadow_command_handler(self):
        run = Path("runs/example")
        with patch.object(cli, "_train") as training:
            cli.main(["train", "--dataset", "datasets/example", "--run", str(run),
                      "--log-every-steps", "25"])
        self.assertEqual(training.call_args.args[0].run, run)
        self.assertEqual(training.call_args.args[0].log_every_steps, 25)

        with patch.object(cli, "_snapshot") as snapshot:
            cli.main(["snapshot", "--run", str(run), "--output", "snapshots/example"])
        self.assertEqual(snapshot.call_args.args[0].run, run)


class DerbakeContractTests(unittest.TestCase):
    def test_hierarchy_requires_cycle_start_before_each_beat(self):
        from ldm.rhythm_ldm import build_rhythm_lookup_arrays, compute_hierarchy_ids_for_sequence
        vocab = list(atomic_vocabulary("no_subd"))
        lookup = build_rhythm_lookup_arrays(vocab, max_beat_positions=8)
        def ids(tokens):
            return [vocab.index(token) for token in tokens]
        cycle = ["<SOC>", "<SOB>", "HIT_D", "<EOB>", "<EOC>"]
        for tokens in (cycle[1:], cycle + cycle[1:]):
            with self.subTest(tokens=tokens), self.assertRaisesRegex(ValueError, "opening <SOC>"):
                compute_hierarchy_ids_for_sequence(ids(tokens), lookup)
        hierarchy = compute_hierarchy_ids_for_sequence(ids(cycle + cycle), lookup)
        self.assertEqual(hierarchy["cycle_uid"].tolist(), [0] * 5 + [1] * 5)
        from ldm.rhythm_ldm import StructuredGenerationTracker, RhythmStateTracker
        with self.assertRaisesRegex(ValueError, "opening <SOC>"):
            RhythmStateTracker(lookup, 4).append(vocab.index("<SOB>"))
        tracker = StructuredGenerationTracker(lookup)
        step = tracker.append(vocab.index("<SOB>"))
        self.assertFalse(step["starts_cycle"])
        self.assertFalse(step["starts_beat"])
        self.assertFalse(step["in_beat"])
        # An unfinished prefix is valid while autoregressive generation is in progress.
        compute_hierarchy_ids_for_sequence(ids(cycle[:-2]), lookup)

    def test_optional_wav_renderer_and_sampling_validation(self):
        with self.assertRaises(ValueError):
            InferenceConfig(top_p=0)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root = Path(folder)
            rhythm = root / "rhythm.derbake"
            write_derbake(RhythmSequence(((Beat(2, ("HIT_D_decorated", "HIT_S")),),)), rhythm)
            lines = rhythm.read_text(encoding="utf-8").splitlines()
            lines[3] = "DELAY_0.5 HIT_OTA DEV_3"
            rhythm.write_text("\n".join(lines) + "\n", encoding="utf-8")
            self.assertEqual(parse_derbake(rhythm).rhythm.cycles[0][0].hits,
                             ("HIT_D_decorated", "HIT_OTA"))
            self.assertEqual(parse_derbake(rhythm, overlay_skeleton=False).rhythm.cycles[0][0].hits,
                             ("HIT_D_decorated", "HIT_S"))
            folders = {group: root / group for group in ("normal_hits", "decorated_hits")}
            for group, names in (("normal_hits", ("doums", "taks", "tiks", "pa2s")),
                                 ("decorated_hits", ("doums", "taks", "pa2s"))):
                for name in names:
                    directory = folders[group] / name
                    directory.mkdir(parents=True)
                    (directory / "sample.wav").touch()
                    (directory / "ignored.flac").touch()
                    (directory / "ignored.ogg").touch()
            (folders["decorated_hits"] / "doums" / "alternate.wav").touch()
            captured = {}
            def read_sample(path, **kwargs):
                self.assertEqual(Path(path).suffix.lower(), ".wav")
                captured.setdefault("chosen_samples", []).append(Path(path))
                return np.ones(4, dtype=np.float32), 8000
            def resample(audio, **kwargs):
                captured["resampled"] = True
                return np.repeat(audio, 6)
            def write_sample(path, audio, rate, **kwargs):
                captured["audio"] = audio.copy()
                captured["rate"] = rate
            fake_soundfile = types.SimpleNamespace(read=read_sample, write=write_sample)
            fake_librosa = types.SimpleNamespace(resample=resample)
            with patch.dict(sys.modules, {"soundfile": fake_soundfile, "librosa": fake_librosa}):
                output = render_wav(rhythm, folders, root / "preview.wav", seed=0)
                first_choice = captured["chosen_samples"][0]
                render_wav(rhythm, folders, root / "preview.wav", seed=1)
            self.assertEqual(output, root / "preview.wav")
            self.assertNotEqual(first_choice, captured["chosen_samples"][2])
            self.assertEqual(first_choice.parent.parent, folders["decorated_hits"])
            self.assertEqual(captured["chosen_samples"][1].parent.parent,
                             folders["normal_hits"])
            self.assertEqual(captured["rate"], 48000)
            self.assertTrue(captured["resampled"])
            self.assertGreater(captured["audio"][5], 0)
            self.assertGreater(captured["audio"][12008], 0)
            self.assertEqual(len(captured["audio"]), 24000)

            # Generated files have a blank skeleton line and still need full-length audio.
            generated = root / "generated.derbake"
            write_derbake(RhythmSequence(((Beat(1, ("HIT_D",)),),
                                          (Beat(1, ("HIT_D_decorated",)),))), generated)
            generated_lines = generated.read_text(encoding="utf-8").splitlines()
            generated_lines[1] = "120 60"
            generated.write_text("\n".join(generated_lines) + "\n", encoding="utf-8")
            with patch.dict(sys.modules, {"soundfile": fake_soundfile, "librosa": fake_librosa}):
                render_wav(generated, folders, root / "generated.wav", seed=0)
            self.assertEqual(len(captured["audio"]), 72000)
            self.assertGreater(captured["audio"][5], 0)
            self.assertGreater(captured["audio"][24005], 0)
            self.assertEqual(captured["chosen_samples"][-2].parent.parent,
                             folders["normal_hits"])
            self.assertEqual(captured["chosen_samples"][-1].parent.parent,
                             folders["decorated_hits"])

    def test_typed_run_config_round_trip_and_cache_recipe(self):
        config = RunConfig("demo", data=DataConfig(
            variant=DatasetVariant.NO_SUBD, context_size=16, stride=8,
            decoration=DecorationMode.WITHOUT_DECORATION))
        self.assertEqual(RunConfig.from_dict(config.to_dict()), config)
        self.assertEqual(make_run_id(config), "demo")
        automatic = replace(config, name=None)
        self.assertEqual(automatic.fingerprint, config.fingerprint)
        self.assertRegex(make_run_id(automatic, datetime(2026, 10, 20, tzinfo=timezone.utc)),
                         r"^run_20261020-000000_no_subd_normal_ctx16_s8_L4H4E256_[0-9a-f]{8}$")
        with self.assertRaises(TypeError):
            RunConfig("bad", data=replace(config.data, variant="no_subd"))
        rhythm = RhythmSequence(tuple((Beat(1, ("HIT_D",)),) for _ in range(10)))
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root = Path(folder)
            with self.assertRaises(ValueError):
                check_run_name(config, root, [config, config])
            (root / "demo").mkdir()
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                check_run_name(config, root)
            source = root / "normal"
            source.mkdir()
            write_derbake(rhythm, source / "one.derbake")
            first = prepare_or_load_dataset(config, source, root / "cache")
            self.assertEqual(first, prepare_or_load_dataset(config, source, root / "cache"))
            changed = replace(config, data=replace(config.data, context_size=24))
            self.assertNotEqual(first, prepare_or_load_dataset(changed, source, root / "cache"))

    def test_skeleton_target_mask_is_scored_once(self):
        rhythm = RhythmSequence(tuple((Beat(1, ("HIT_D",)),) for _ in range(10)))
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root = Path(folder)
            source = root / "normal"
            source.mkdir()
            path = source / "one.derbake"
            write_derbake(rhythm, path)
            lines = path.read_text(encoding="utf-8").splitlines()
            lines[3] = "DELAY_0 HIT_D DEV_0"
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            prepare_dataset(source, root / "prepared", variant="with_subd",
                            context_size=8, max_beats_per_cycle=16, stride=4)
            skeleton = np.load(root / "prepared" / "train_skeleton_mask.npy")
            self.assertEqual(int(skeleton.sum()), 1)

    def test_invalid_atom_ids_and_subdivisions_are_rejected(self):
        vocab = AtomicVocabulary(atomic_vocabulary("with_subd"))
        with self.assertRaises(ValueError):
            vocab.decode([-1])
        schema = RhythmSchema("with_subd", 16)
        for subd in ("SUBD_0", "SUBD_01", "SUBD_9"):
            with self.subTest(subd=subd), self.assertRaises(ValueError):
                schema.decode(["<SOC>", "<SOB>", subd, "HIT_D", "<EOB>", "<EOC>"])

    def test_writer_round_trip_both_variants(self):
        rhythm = RhythmSequence(((Beat(2, ("HIT_D", "HIT_S")),),
                                 (Beat(4, ("HIT_S", "HIT_OTA_decorated", "HIT_S", "HIT_PA2")),
                                  Beat(1, ("HIT_OTI",)))))
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            path = Path(folder) / "generated.derbake"
            write_derbake(rhythm, path)
            self.assertEqual(path.read_text(encoding="utf-8").splitlines()[3], "")
            self.assertEqual(parse_derbake(path).rhythm, rhythm)
        for variant in ("with_subd", "no_subd"):
            schema = RhythmSchema(variant, 16)
            vocab = AtomicVocabulary(atomic_vocabulary(variant))
            atoms = schema.encode(rhythm)
            self.assertEqual(schema.decode(vocab.decode(vocab.encode(atoms))), rhythm)
            if variant == "no_subd":
                self.assertFalse(any(atom.startswith("SUBD_") for atom in atoms))

    def test_skeleton_override_and_bad_counts(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            path = Path(folder) / "prompt.derbake"
            path.write_text("120\n120 120\n2\nDELAY_0 HIT_D DEV_0\n"
                            "SUBD_1 HIT_S AMP_1 SUBD_1 HIT_OTA AMP_1\n", encoding="utf-8")
            rhythm = parse_derbake(path).rhythm
            self.assertEqual(rhythm.cycles[0][0].hits, ("HIT_D",))
            path.write_text(path.read_text(encoding="utf-8").replace("120 120\n", "120\n"),
                            encoding="utf-8")
            with self.assertRaises(ValueError):
                parse_derbake(path)

    def test_cycle_collector_skips_invalid_candidate(self):
        schema = RhythmSchema("no_subd", 8)
        vocab = AtomicVocabulary(atomic_vocabulary("no_subd"))
        collector = CycleCollector(schema, 2)
        sampled = ["<SOC>", "<SOB>", "<EOB>", "<EOC>",
                   "<SOC>", "<SOB>", "HIT_D", "<EOB>", "<EOC>",
                   "<SOC>", "<SOB>", "HIT_OTA_decorated", "<EOB>", "<EOC>"]
        done = False
        for atom in sampled:
            done = collector.append(vocab.encode([atom])[0], atom)
        ids, report = collector.finish()
        self.assertTrue(done)
        self.assertEqual(report["accepted_cycles"], 2)
        self.assertEqual(report["discarded_cycles"], 1)
        self.assertEqual(len(schema.decode(vocab.decode(ids)).cycles), 2)

    def test_incomplete_generation_reports_discard(self):
        collector = CycleCollector(RhythmSchema("with_subd", 8), 1)
        collector.append(1, "<SOC>")
        with self.assertRaises(IncompleteGenerationError) as context:
            collector.finish()
        self.assertEqual(context.exception.report["discarded_cycles"], 1)

    def test_nested_cycle_resynchronizes_and_orphans_are_reported(self):
        schema = RhythmSchema("no_subd", 8)
        vocab = AtomicVocabulary(atomic_vocabulary("no_subd"))
        collector = CycleCollector(schema, 1)
        atoms = ["<EOC>", "<SOC>", "<SOB>", "<SOC>", "<SOB>",
                 "HIT_D", "<EOB>", "<EOC>"]
        for atom in atoms:
            done = collector.append(vocab.encode([atom])[0], atom)
        ids, report = collector.finish()
        self.assertTrue(done)
        self.assertEqual(report["orphan_tokens"], 1)
        self.assertEqual(report["discarded_cycles"], 1)
        self.assertEqual(schema.decode(vocab.decode(ids)).cycle_lengths, (1,))

    def test_per_file_split(self):
        for size in (5, 6, 10, 100):
            ranges = split_cycle_ranges(size)
            train, val, test = ranges.values()
            self.assertEqual(train[0], 0)
            self.assertEqual(train[1], val[0])
            self.assertEqual(val[1], test[0])
            self.assertEqual(test[1], size)
            self.assertTrue(all(end > start for start, end in ranges.values()))
        self.assertEqual(split_cycle_ranges(100),
                         {"train": (0, 70), "val": (70, 90), "test": (90, 100)})

    def test_notebook_downloads_independent_and_decorated_only_sources(self):
        notebook = json.loads((Path(__file__).resolve().parents[1] / "notebooks" / "ldm_training.ipynb").read_text())
        code = next("".join(cell["source"]) for cell in notebook["cells"]
                    if cell["cell_type"] == "code" and "SELECT_DATA_COLLECTIONS = {" in "".join(cell["source"]))
        code = code.replace("'five-line'", "'normal-source'", 1).replace("'five-line'", "'decorated-source'", 1)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root = Path(folder)
            normal = "normal-source/normal/a.derbake"
            decorated = ["decorated-source/decorated/b.derbake", "decorated-source/decorated/c.derbake"]
            for name in [normal, *decorated]:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            downloads = []
            def download(**kwargs):
                downloads.append(kwargs['allow_patterns'])
                return str(root)
            def environment():
                return {"Path": Path, "project": root, "DATA_REPO_ID": "test/data", "DATA_COMMIT": "test-sha",
                        "DATA_COLLECTIONS": {"normal-source": {"normal": [normal]},
                                             "decorated-source": {"decorated": decorated}}}
            with patch.dict(sys.modules, {"huggingface_hub": types.SimpleNamespace(snapshot_download=download)}):
                scope = environment()
                exec(code, scope)
                self.assertEqual(set(scope['DATA_SOURCES']), set(DecorationMode))
                self.assertEqual(set(downloads[-1]), {normal, *decorated})
                scope['SELECT_DATA_COLLECTIONS'][DecorationMode.WITHOUT_DECORATION] = None
                self.assertNotEqual(scope['current_data_selection'](), scope['SELECTED_DATA_SOURCE'])
                scope = environment()
                exec(code.replace("DecorationMode.WITHOUT_DECORATION: 'normal-source'",
                                  "DecorationMode.WITHOUT_DECORATION: None"), scope)
                self.assertEqual(set(scope['DATA_SOURCES']), {DecorationMode.WITH_DECORATION})
                self.assertEqual(set(downloads[-1]), set(decorated))

    def test_preparation_uses_independent_sources_and_splits_each_file(self):
        normal = RhythmSequence(tuple((Beat(1, ("HIT_D",)),) for _ in range(10)))
        decorated = RhythmSequence(tuple((Beat(1, ("HIT_D_decorated",)),) for _ in range(12)))
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root = Path(folder)
            (root / "normal").mkdir()
            (root / "decorated").mkdir()
            write_derbake(normal, root / "normal" / "normal_only.derbake")
            write_derbake(decorated, root / "decorated" / "unrelated.derbake")
            caches = []
            for variant in DatasetVariant:
                for mode in DecorationMode:
                    config = RunConfig(data=DataConfig(variant=variant, decoration=mode,
                                                        context_size=16, max_beats_per_cycle=16))
                    destination = prepare_or_load_dataset(config, root / mode.value, root / "cache")
                    self.assertEqual(destination, prepare_or_load_dataset(config, root / mode.value, root / "cache"))
                    caches.append(destination)
                    manifest = json.loads((destination / "dataset_manifest.json").read_text())
                    from ldm.constants import PREPARATION_VERSION, NORMALIZATION_RULE, SPLIT_RULE
                    self.assertEqual(manifest["preparation_version"], PREPARATION_VERSION)
                    self.assertEqual(manifest["normalization_rule"], NORMALIZATION_RULE)
                    self.assertEqual(manifest["split_rule"], SPLIT_RULE)
                    self.assertIn("artifacts", manifest)
                    self.assertTrue({"format_version", "normalization_version", "split_version",
                                     "integrity_version"}.isdisjoint(manifest))
                    self.assertEqual(manifest["decoration"], mode.value)
                    self.assertEqual(manifest["files"][0]["cycles"], 10 if mode is DecorationMode.WITHOUT_DECORATION else 12)
                    self.assertEqual(manifest["files"][0]["file"], "normal_only.derbake" if mode is DecorationMode.WITHOUT_DECORATION else "unrelated.derbake")
                    for split in ("train", "val", "test"):
                        self.assertEqual(manifest["stats"][split]["sequences_seen"], 1)
                        self.assertGreater(manifest["stats"][split]["unique_targets"], 0)
                        self.assertEqual(set(np.load(destination / f"{split}_source_file.npy")), {0})
                        self.assertFalse((destination / f"{split}_source_kind.npy").exists())
            self.assertEqual(len(set(caches)), 4)


if __name__ == "__main__":
    unittest.main()
