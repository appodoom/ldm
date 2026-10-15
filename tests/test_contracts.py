"""Contract tests that run without Torch or network access."""

from __future__ import annotations

import tempfile
import unittest
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
            path.write_text("120\n120 120\n2\nDELAY_0 HIT_D_SKEL DEV_0\n"
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

    def test_preparation_splits_each_paired_file_before_chunking(self):
        ordinary = RhythmSequence(tuple((Beat(1, ("HIT_D",)),) for _ in range(10)))
        decorated = RhythmSequence(tuple((Beat(1, ("HIT_D_decorated",)),) for _ in range(10)))
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root = Path(folder)
            (root / "ordinary").mkdir()
            (root / "decorated").mkdir()
            write_derbake(ordinary, root / "ordinary" / "one.derbake")
            write_derbake(decorated, root / "decorated" / "one.derbake")
            for variant in ("with_subd", "no_subd"):
                destination = root / variant
                manifest = prepare_dataset(root / "ordinary", destination,
                    variant=variant, context_size=16, max_beats_per_cycle=16,
                    decorated_dir=root / "decorated")
                self.assertEqual(manifest["files"][0]["ranges"],
                                 {"train": [0, 7], "val": [7, 9], "test": [9, 10]})
                for split in ("train", "val", "test"):
                    self.assertEqual(manifest["stats"][split]["sequences_seen"], 2)
                    self.assertGreater(manifest["stats"][split]["unique_targets"], 0)
                    self.assertEqual(np.load(destination / f"{split}.npy").shape[0],
                                     manifest["stats"][split]["chunks"])
                    self.assertEqual(set(np.load(destination / f"{split}_source_kind.npy")),
                                     {0, 1})


if __name__ == "__main__":
    unittest.main()
