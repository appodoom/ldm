"""The shared HIT inventory preserves token IDs and the five-line contract."""

import tempfile
import unittest
from pathlib import Path

from ldm.constants import HITS, HIT_ORDER, HIT_FOLDERS, DECORATED_SUFFIX, SILENT_HIT, DERBAKE_LINE_COUNT
from ldm.config import AudioSamples
from ldm.derbake_ldm import Beat, RhythmSequence, parse_derbake, write_derbake
from ldm.schema_ldm import RhythmSchema
from ldm.vocabulary import atomic_vocabulary, vocabulary_hash


class FormatConstantsTests(unittest.TestCase):
    def test_inventory_is_shared_and_token_ids_are_unchanged(self):
        self.assertEqual(HITS, frozenset(HIT_ORDER))
        expected = {
            "with_subd": "07ecc9eaa36d8a352e86f7faf18cb6524def356a2c6971df8cb67525753b1cdb",
            "no_subd": "bd6374a464f900b5eb621c713484aa8894f52dc2a600db1b585f5aeabc40afb7",
        }
        for variant, fingerprint in expected.items():
            self.assertEqual(vocabulary_hash(atomic_vocabulary(variant)), fingerprint)
            for hit in HIT_ORDER:
                rhythm = RhythmSequence(((Beat(1, (hit,)),),))
                schema = RhythmSchema(variant, 1)
                self.assertEqual(schema.decode(schema.encode(rhythm)), rhythm)
                if hit != SILENT_HIT:
                    self.assertIn(hit.removesuffix(DECORATED_SUFFIX), HIT_FOLDERS)
        with self.assertRaises(ValueError):
            AudioSamples({}, 48000, 0).get_audio_data("HIT_OTI_decorated")

    def test_document_contains_all_five_lines(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            source = Path(folder) / "source.derbake"
            source.write_text("90\n100 120\n1 1\nDELAY_0 HIT_OTA DEV_3\n"
                              "SUBD_1 HIT_D AMP_0.5 SUBD_1 HIT_D_decorated AMP_2\n", encoding="utf-8")
            document = parse_derbake(source, overlay_skeleton=False)
            self.assertEqual(document.initial_tempo, 90)
            self.assertEqual(document.tempos, (100, 120))
            self.assertEqual(document.rhythm.cycle_lengths, (1, 1))
            self.assertEqual(document.skeleton_events[0].hit, "HIT_OTA")
            self.assertEqual(document.skeleton_events[0].deviation, 3)
            self.assertEqual(document.rhythm.cycles[0][0].hits, ("HIT_D",))
            self.assertEqual(document.variation_amplitudes, ((0.5,), (2,)))
            generated = Path(folder) / "generated.derbake"
            write_derbake(document.rhythm, generated)
            self.assertEqual(len(generated.read_text().splitlines()), DERBAKE_LINE_COUNT)
            self.assertEqual(parse_derbake(generated).rhythm, document.rhythm)


if __name__ == "__main__":
    unittest.main()
