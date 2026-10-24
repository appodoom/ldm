"""Behavioral checks for shared preparation, device metrics, and reusable inference."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from ldm.data_loading import ChunkDataset, close_loader, evaluation_loader
from ldm.dataset_integrity import _sha256
from ldm.dataset_ldm import prepare_dataset
from ldm.derbake_ldm import Beat, RhythmSequence, parse_derbake, write_derbake
from ldm.evaluation_ldm import DeviceCounts, evaluate_model
from ldm.metric_counts import TokenCounts
from ldm.normalized_data import load_normalized_source
from ldm.schema_ldm import RhythmSchema
from ldm.splits import split_cycle_ranges
from ldm.vocabulary import atomic_vocabulary


def source_file(root):
    source = root / 'source'
    source.mkdir()
    path = source / 'example.derbake'
    rhythm = RhythmSequence(tuple((Beat(2, ('HIT_D', 'HIT_S')), Beat(1, ('HIT_OTA',)))
                                 for _ in range(10)))
    write_derbake(rhythm, path)
    lines = path.read_text().splitlines()
    # Duplicate slot, decorated skeleton HIT, silent skeleton HIT, split boundary,
    # and a fractional event in the last beat.
    lines[3] = ('DELAY_0 HIT_D DEV_0 DELAY_0 HIT_OTA_decorated DEV_1 '
                'DELAY_0.5 HIT_S DEV_0 DELAY_13.5 HIT_PA2 DEV_0 '
                'DELAY_5.75 HIT_D DEV_0')
    path.write_text('\n'.join(lines) + '\n')
    return source, path


class PreparationReuseTests(unittest.TestCase):
    def test_variants_and_contexts_share_normalization_and_preserve_provenance(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            source, path = source_file(root)
            document = parse_derbake(path)
            with patch('ldm.normalized_data.parse_derbake', wraps=parse_derbake) as parser:
                for variant, context in (('with_subd', 16), ('no_subd', 16), ('with_subd', 32)):
                    prepared = root / f'{variant}{context}'
                    prepare_dataset(source, prepared, variant=variant, context_size=context, max_beats_per_cycle=4)
                self.assertEqual(parser.call_count, 1)
            normalized = load_normalized_source(path, root / 'normalized', _sha256(path))
            for variant in ('with_subd', 'no_subd'):
                schema = RhythmSchema(variant, 4)
                flat = [beat for cycle in document.rhythm.cycles for beat in cycle]
                slots = set()
                for event in document.skeleton_events:
                    b = int(event.time_in_beats)
                    slot = min(flat[b].subdivision - 1, int((event.time_in_beats - b) * flat[b].subdivision))
                    slots.add((b, slot))
                for start, end in split_cycle_ranges(len(document.rhythm.cycles)).values():
                    self.assertEqual(schema.encode_cycles(normalized.iter_cycles(start, end)),
                                     schema.encode(RhythmSequence(document.rhythm.cycles[start:end])))
                    expected = []
                    beat_index = sum(document.rhythm.cycle_lengths[:start])
                    for cycle in document.rhythm.cycles[start:end]:
                        expected.append(False)
                        for beat in cycle:
                            expected.extend([False] * (2 if variant == 'with_subd' else 1))
                            expected.extend((beat_index, slot) in slots for slot in range(beat.subdivision))
                            expected.append(False)
                            beat_index += 1
                        expected.append(False)
                    self.assertEqual(normalized.atom_mask(start, end, variant), expected)
            del normalized
            with self.assertRaisesRegex(ValueError, 'capacity'):
                prepare_dataset(source, root / 'too-small', variant='no_subd', context_size=16,
                                max_beats_per_cycle=1)
            with patch('ldm.normalized_data.parse_derbake', wraps=parse_derbake) as parser:
                path.write_text(path.read_text().replace('HIT_D AMP_', 'HIT_PA2 AMP_'))
                data = load_normalized_source(path, root / 'normalized', _sha256(path))
                self.assertEqual(parser.call_count, 1)
                self.assertIn('HIT_PA2', next(data.iter_cycles(1, 2))[0].hits)
                del data
            folder = next((root / 'normalized').glob('*/hits.npy')).parent
            with (folder / 'hits.npy').open('ab') as stream:
                stream.write(b'corrupt')
            recipe = json.loads((folder / 'manifest.json').read_text())['recipe']
            with self.assertRaisesRegex(ValueError, 'content check failed'):
                load_normalized_source(path, root / 'normalized', recipe['source_hash'])
            # An interrupted directory is rejected rather than treated as normalized data.
            (folder / 'manifest.json').unlink()
            with self.assertRaises(FileNotFoundError):
                load_normalized_source(path, root / 'normalized', recipe['source_hash'])


class DeviceMetricTests(unittest.TestCase):
    def check_counts(self, device):
        rng = np.random.default_rng(7)
        for variant in ('with_subd', 'no_subd'):
            vocab = list(atomic_vocabulary(variant))
            reference = TokenCounts(vocab)
            counters = DeviceCounts(vocab, ['a', 'b'], device)
            by_file = {'a': [0, 0], 'b': [0, 0]}
            for width in (1, 4, 29):
                targets = rng.integers(0, len(vocab), (5, width))
                predictions = rng.integers(0, len(vocab), (5, width))
                predictions[0] = vocab.index('HIT_D')  # Overlong runs and row-end runs.
                inputs = rng.integers(0, len(vocab), (5, width))
                if variant == 'with_subd':
                    inputs[0] = vocab.index('SUBD_1')
                targets[rng.random(targets.shape) < .3] = -100
                targets[-1] = -100  # Entirely unscored row.
                losses = rng.random(targets.shape).astype(np.float32)
                losses[targets == -100] = 0
                skeleton = rng.random(targets.shape) < .4
                hit_count = rng.random(targets.shape) < .8
                file_ids = np.array([0, 1, 0, 1, 1])
                valid = targets != -100
                reference.add(targets[valid], predictions[valid], losses[valid], (skeleton & valid)[valid])
                for row, position in np.argwhere(hit_count):
                    token = vocab[int(inputs[row, position])]
                    if variant != 'with_subd' or not token.startswith('SUBD_'):
                        continue
                    actual = 0
                    for pred in predictions[row, position:]:
                        if not reference.hit_ids[pred]:
                            break
                        actual += 1
                    reference.hit_count_total += 1
                    reference.hit_count_correct += int(actual == int(token[5:]))
                for row, file_id in enumerate(file_ids):
                    selected = skeleton[row] & valid[row]
                    by_file[['a', 'b'][file_id]][0] += int(((targets[row] == predictions[row]) & selected).sum())
                    by_file[['a', 'b'][file_id]][1] += int(selected.sum())
                tensor = lambda array: torch.as_tensor(array, device=device)
                counters.add(tensor(inputs), tensor(targets), tensor(predictions), tensor(losses),
                             {'skeleton': tensor(skeleton), 'hit_count': tensor(hit_count),
                              'source_file': tensor(file_ids)}, variant == 'with_subd')
            actual, files = counters.result()
            expected = reference.result()
            self.assertAlmostEqual(actual.pop('loss'), expected.pop('loss'), places=6)
            self.assertEqual(actual, expected)
            for name, (correct, total) in by_file.items():
                self.assertEqual(files[name]['correct'], correct)
                self.assertEqual(files[name]['total'], total)

    def test_cpu_counts_match_unbounded_reference(self):
        self.check_counts(torch.device('cpu'))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA runtime required')
    def test_cuda_counts_match_unbounded_reference(self):
        self.check_counts(torch.device('cuda'))

    def test_empty_owned_batches_are_skipped_and_failed_one_pass_workers_close(self):
        from types import SimpleNamespace
        from unittest.mock import Mock

        vocab = list(atomic_vocabulary('no_subd'))
        model = Mock(vocab=vocab, config=SimpleNamespace(dataset_variant='no_subd'),
                     core=SimpleNamespace(blocks=[]))
        model.side_effect = AssertionError('An unowned batch must not reach the model')
        empty = {'input_ids': torch.zeros((1, 3), dtype=torch.long),
                 'labels': torch.full((1, 3), -100, dtype=torch.long), 'evaluation': {}}
        class UnownedLoader:
            persistent_workers = False
            def __iter__(self):
                return iter([empty])
        data = SimpleNamespace(evaluation=True, file_names=[])
        with self.assertRaisesRegex(ValueError, 'no scored targets'):
            evaluate_model(model, data, 1, torch.device('cpu'), loader=UnownedLoader())
        model.assert_not_called()

        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            source, _ = source_file(root)
            prepare_dataset(source, root / 'data', variant='no_subd', context_size=16,
                            max_beats_per_cycle=4)
            dataset = ChunkDataset(root / 'data', 'val', evaluation=True)
            loader = evaluation_loader(dataset, 1, torch.device('cuda'), num_workers=1,
                                       pin_memory=False, persistent_workers=False)
            iterator = iter(loader)
            class SinglePassLoader:
                persistent_workers = False
                def __iter__(self):
                    return iterator
            model.side_effect = RuntimeError('intentional evaluation failure')
            try:
                with self.assertRaisesRegex(RuntimeError, 'intentional evaluation failure'):
                    evaluate_model(model, dataset, 1, torch.device('cpu'), loader=SinglePassLoader())
                self.assertTrue(all(not worker.is_alive() for worker in iterator._workers))
            finally:
                close_loader(loader, iterator)
                del dataset, loader, iterator

    def test_global_loading_skips_routing_and_ordered_workers_preserve_metrics(self):
        from ldm.configuration_ldm import LDMConfig
        from ldm.modeling_ldm import LDMModel
        from ldm.rhythm_ldm import STRUCTURED_ROUTING_FIELDS
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
                root = Path(directory)
                source, _ = source_file(root)
                folder = root / 'data'
                prepare_dataset(source, folder, variant='with_subd', context_size=16, max_beats_per_cycle=4)
                for layout in ('global', 'alternating_global_structured'):
                    config = LDMConfig(dataset_variant='with_subd', context_size=16, max_beats_per_cycle=4,
                                       attention_layout=layout, number_of_layers=2, number_of_heads=2,
                                       number_of_embeddings=16, dropout=0, max_tokens_per_beat=16,
                                       max_beat_slots_per_chunk=16, max_cycles_per_chunk=16)
                    model = LDMModel(config).eval()
                    with patch('ldm.data_loading.np.load', wraps=np.load) as loaded:
                        dataset = ChunkDataset(folder, 'train', model_config=config, evaluation=True)
                        opened = [Path(call.args[0]).name for call in loaded.call_args_list]
                    if layout == 'global':
                        self.assertFalse(dataset.routing)
                        for field in STRUCTURED_ROUTING_FIELDS:
                            self.assertNotIn(f'train_{field}.npy', opened)
                    else:
                        self.assertEqual(set(dataset.routing), set(STRUCTURED_ROUTING_FIELDS))
                    full = ChunkDataset(folder, 'train')
                    with torch.no_grad():
                        lean_batch = dataset.batch([0], torch.device('cpu'))
                        lean_batch.pop('evaluation')
                        old_logits = model(**full.batch([0], torch.device('cpu'))).logits
                        torch.testing.assert_close(model(**lean_batch).logits, old_logits, rtol=0, atol=0)
                    expected = evaluate_model(model, dataset, 2, torch.device('cpu'))
                    loader = evaluation_loader(dataset, 2, torch.device('cuda'), pin_memory=False, num_workers=2)
                    before = torch.get_rng_state().clone()
                    try:
                        for _ in range(2):
                            self.assertEqual(evaluate_model(model, dataset, 2, torch.device('cpu'), loader=loader), expected)
                        torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
                    finally:
                        close_loader(loader)
                    del loader, dataset, full
        finally:
            torch.set_num_threads(old_threads)
