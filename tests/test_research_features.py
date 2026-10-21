"""Content corruption, metric outcomes, analysis math, and attention correctness."""

import importlib.util
import json
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

from ldm.dataset_ldm import prepare_dataset
from ldm.dataset_integrity import dataset_fingerprint, verify_dataset
from ldm.derbake_ldm import Beat, RhythmSequence, write_derbake
from ldm.embedding_analysis import embedding_statistics
from ldm.metric_counts import TokenCounts
from ldm.vocabulary import atomic_vocabulary


class IntegrityTests(unittest.TestCase):
    def test_model_dataset_compatibility_is_checked(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root = Path(folder)
            source = root / 'source'
            source.mkdir()
            write_derbake(RhythmSequence(tuple((Beat(1, ('HIT_D',)),) for _ in range(10))), source / 'a.derbake')
            manifest = prepare_dataset(source, root / 'data', variant='with_subd', context_size=16, max_beats_per_cycle=4)
            config = {key: manifest[key] for key in ('vocab_hash', 'context_size', 'max_beats_per_cycle')}
            config['dataset_variant'] = 'with_subd'
            config['attention_layout'] = 'alternating_global_structured'
            for key in ('max_tokens_per_beat', 'max_beat_slots_per_chunk', 'max_cycles_per_chunk'):
                config[key] = max(stats[key] for stats in manifest['stats'].values())
            self.assertEqual(verify_dataset(root / 'data', model_config=config), verify_dataset(root / 'data'))
            for key, value in (('dataset_variant', 'no_subd'), ('vocab_hash', 'wrong'),
                               ('context_size', 32), ('max_cycles_per_chunk', 0)):
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'differs from model'):
                    verify_dataset(root / 'data', model_config={**config, key: value})
            verify_dataset(root / 'data', model_config={**config, 'attention_layout': 'global', 'max_cycles_per_chunk': 0})

    def test_routing_corruption_and_missing_payload_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root = Path(folder)
            source = root / 'source'
            source.mkdir()
            write_derbake(RhythmSequence(tuple((Beat(1, ('HIT_D',)),) for _ in range(10))), source / 'a.derbake')
            target = root / 'prepared'
            manifest = prepare_dataset(source, target, variant='with_subd', context_size=16, max_beats_per_cycle=4)
            self.assertEqual(len(manifest['artifacts']), 3 * 15)
            original = verify_dataset(target)
            # Absolute source locations do not affect dataset identity.
            manifest['ordinary_dir'] = '/different/source/location'
            (target / 'dataset_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            self.assertEqual(dataset_fingerprint(target), original)
            array = np.load(target / 'val_beat_slot.npy', mmap_mode='r+')
            array[0, 0] += 1
            array.flush()
            del array
            with self.assertRaisesRegex(ValueError, 'content check failed'):
                verify_dataset(target)
            (target / 'test_cycle_slot.npy').unlink()
            with self.assertRaisesRegex(ValueError, 'missing='):
                verify_dataset(target)


class ResearchMetricTests(unittest.TestCase):
    def test_decoration_outcomes_are_mutually_exclusive(self):
        vocab = list(atomic_vocabulary('no_subd'))
        counts = TokenCounts(vocab)
        targets = np.array([vocab.index('HIT_D_decorated')] * 5)
        predictions = np.array([vocab.index(token) for token in
            ('HIT_D_decorated', 'HIT_OTA_decorated', 'HIT_D', 'HIT_OTA', 'HIT_S')])
        counts.add(targets, predictions, np.zeros(5), np.zeros(5, dtype=bool))
        result = counts.result()
        self.assertEqual(result['decorated_outcomes'], {'total': 5, 'exact_decorated': 1,
            'wrong_decorated_type': 1, 'correct_plain_hit': 1, 'other': 2})
        self.assertEqual(result['decorated_plain_hit_accuracy'], 0.2)
        self.assertEqual(result['decorated_outcomes_by_type']['HIT_D_decorated'], result['decorated_outcomes'])
        self.assertIsNone(TokenCounts(vocab).result()['decorated_plain_hit_accuracy'])

    def test_subdivision_neighbors_exclude_self_and_report_purity(self):
        result = embedding_statistics(['SUBD_1', 'SUBD_2', 'HIT_D', 'HIT_S'],
            np.array([[1, 0], [.9, .1], [-1, 0], [0, -1]]), top_k=1)
        self.assertEqual(result['metrics']['overall_purity'], 1.0)
        self.assertTrue(all(row['token'] != row['neighbor'] for row in result['neighbors']))
        plain = embedding_statistics(['HIT_D', 'HIT_S'], np.eye(2))
        self.assertFalse(plain['metrics']['subdivision_analysis_applicable'])
        self.assertIsNone(plain['metrics']['overall_purity'])


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch runtime required')
class AttentionChecks(unittest.TestCase):
    def test_precision_policy_and_config_round_trip(self):
        import torch
        from ldm.model_core import configure_cuda_precision
        from ldm.experiment_config import RunConfig, TrainConfig
        original = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                    torch.get_float32_matmul_precision())
        try:
            for enabled in (True, False):
                report = configure_cuda_precision('cuda', allow_tf32=enabled)
                self.assertEqual(report['allow_tf32'], enabled)
                self.assertEqual(report['cudnn_allow_tf32'], enabled)
                self.assertEqual(report['matmul_precision'], 'high' if enabled else 'highest')
        finally:
            torch.set_float32_matmul_precision(original[2])
            torch.backends.cuda.matmul.allow_tf32 = original[0]
            torch.backends.cudnn.allow_tf32 = original[1]
        config = RunConfig(train=TrainConfig(num_workers=0, persistent_workers=False, allow_tf32=False))
        self.assertEqual(config, RunConfig.from_dict(config.to_dict()))
        with self.assertRaises(ValueError):
            RunConfig(train=TrainConfig(prefetch_factor=0))

    def test_precision_scope_and_validation_restore_settings_on_failure(self):
        import torch
        from unittest.mock import patch
        from ldm.model_core import cuda_precision_policy
        from ldm.model_checks import validate_core
        before = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                  torch.get_float32_matmul_precision())
        with self.assertRaisesRegex(RuntimeError, 'intentional'):
            with cuda_precision_policy('cuda', allow_tf32=True):
                self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
                raise RuntimeError('intentional')
        self.assertEqual(before, (torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32, torch.get_float32_matmul_precision()))
        core = torch.nn.Module()
        with patch.object(core, 'parameters', return_value=iter([types.SimpleNamespace(device=torch.device('cuda'))])):
            def fail_fixture(*args):
                self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
                self.assertFalse(torch.is_autocast_enabled('cuda'))
                raise RuntimeError('intentional validation failure')
            with patch('ldm.model_checks._fixture', side_effect=fail_fixture):
                with self.assertRaisesRegex(RuntimeError, 'intentional validation'):
                    validate_core(core, [], 'no_subd')
        self.assertTrue(core.training)
        self.assertEqual(before, (torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32, torch.get_float32_matmul_precision()))

    def test_prefetched_workers_match_direct_batches_and_resume_order(self):
        import torch
        import pickle
        from ldm.data_loading import ChunkDataset, chunk_loader, batch_to_device
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root = Path(folder)
            source = root / 'source'
            source.mkdir()
            write_derbake(RhythmSequence(tuple((Beat(1, ('HIT_D',)),) for _ in range(10))), source / 'a.derbake')
            prepare_dataset(source, root / 'data', variant='no_subd', context_size=8, max_beats_per_cycle=4)
            dataset = ChunkDataset(root / 'data', 'train')
            reopened = pickle.loads(pickle.dumps(dataset))
            self.assertIsInstance(reopened.ids, np.memmap)
            batches = [[0], [1, 0], [1]]
            # Exercise real spawned workers on a CPU-only machine; no CUDA tensors
            # are created in workers and pinning is independently disabled here.
            loader = chunk_loader(dataset, batches, device=torch.device('cuda'), pin_memory=False,
                                  num_workers=2, persistent_workers=True, prefetch_factor=2)
            before = torch.get_rng_state().clone()
            try:
                for order in ([[0], [1, 0], [1]], [[1], [0]]):
                    batches[:] = order
                    for rows, actual in zip(order, loader):
                        actual = batch_to_device(actual, torch.device('cpu'))
                        expected = dataset.batch(rows, torch.device('cpu'))
                        for key in expected:
                            if isinstance(expected[key], dict):
                                for field in expected[key]:
                                    torch.testing.assert_close(actual[key][field], expected[key][field])
                            else:
                                torch.testing.assert_close(actual[key], expected[key])
                torch.testing.assert_close(before, torch.get_rng_state())
            finally:
                # Closing the loader shuts persistent workers before temp-file cleanup.
                del loader
            cpu_loader = chunk_loader(dataset, [[0]], device=torch.device('cpu'), num_workers=2)
            self.assertEqual(cpu_loader.num_workers, 0)
            self.assertFalse(cpu_loader.pin_memory)
            self.assertEqual(len(list(cpu_loader)), 1)
            del cpu_loader, dataset, reopened
            import gc
            gc.collect()

    def test_flash_cpu_fallback_and_verified_dispatch(self):
        import torch
        from unittest.mock import Mock, patch
        from ldm.model_core import Transformer_Attention_Block, configure_global_flash
        block = Transformer_Attention_Block(16, 2, 2, 0)
        core = torch.nn.Module()
        core.blocks = torch.nn.ModuleList([block])
        core.context_size = 32
        before = torch.get_rng_state().clone()
        report = configure_global_flash(core)
        self.assertFalse(report['verified'])
        self.assertIsNone(block.flash_dtype)
        torch.testing.assert_close(before, torch.get_rng_state())
        x = torch.randn(1, 4, 16)
        # A verified CUDA dtype forces Flash; FP32 still dispatches automatically.
        query = Mock(is_cuda=True, dtype=torch.float16)
        block.flash_dtype = torch.float16
        with patch.object(block, '_project_qkv', return_value=(query, query, query)), \
             patch('ldm.model_core.F.scaled_dot_product_attention', return_value=torch.zeros(1, 2, 4, 8)), \
             patch('ldm.model_core.sdpa_kernel') as kernel:
            block(x)
            kernel.assert_called_once()
            kernel.reset_mock()
            query.dtype = torch.float32
            block(x)
            kernel.assert_not_called()
        # An actual CPU backward remains valid even if a prior CUDA policy exists.
        block(x).sum().backward()
        self.assertIsNotNone(block.attention_layer.in_proj_weight.grad)

    def test_flash_failure_falls_back_except_on_a100(self):
        import torch
        from contextlib import nullcontext
        from unittest.mock import patch
        from ldm.model_core import Transformer_Attention_Block, configure_global_flash
        core = torch.nn.Module()
        core.blocks = torch.nn.ModuleList([Transformer_Attention_Block(128, 2, 2, 0.1)])
        core.context_size = 32
        for name in ('Tesla T4', 'NVIDIA A100-SXM4-40GB'):
            with self.subTest(device=name), \
                 patch.object(core, 'parameters', return_value=iter([types.SimpleNamespace(device=torch.device('cuda:0'))])), \
                 patch('ldm.model_core.torch.cuda.get_device_name', return_value=name), \
                 patch('ldm.model_core.torch.random.fork_rng', return_value=nullcontext()), \
                 patch('ldm.model_core.torch.randn', side_effect=RuntimeError('unsupported Flash configuration')):
                if 'A100' in name:
                    with self.assertRaisesRegex(RuntimeError, 'Flash verification failed on A100'):
                        configure_global_flash(core)
                else:
                    report = configure_global_flash(core)
                    self.assertFalse(report['verified'])
                    self.assertEqual(report['global_policy'], 'automatic')
                self.assertIsNone(core.blocks[0].flash_dtype)

    @unittest.skipUnless(importlib.util.find_spec('torch') and __import__('torch').cuda.is_available(), 'CUDA runtime required')
    def test_flash_real_cuda_forward_backward_preserves_rng(self):
        import torch
        from ldm.model_core import Transformer_Attention_Block, configure_global_flash
        core = torch.nn.Module()
        core.blocks = torch.nn.ModuleList([Transformer_Attention_Block(128, 2, 2, 0.1)])
        core.context_size = 32
        core.cuda()
        before = torch.cuda.get_rng_state().clone()
        report = configure_global_flash(core)
        torch.testing.assert_close(before, torch.cuda.get_rng_state())
        if not report['verified']:
            self.skipTest(report['reason'])
        with torch.autocast('cuda'):
            output = core.blocks[0](torch.randn(1, 32, 128, device='cuda'))
        output.sum().backward()
        self.assertTrue(torch.isfinite(core.blocks[0].attention_layer.in_proj_weight.grad).all())

    def test_cache_causality_padding_and_disabled_streams_for_both_variants(self):
        import torch
        from ldm.model_core import GPT
        from ldm.model_checks import validate_core
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for variant in ('with_subd', 'no_subd'):
                for layout in ('global', 'alternating_global_structured'):
                    with self.subTest(variant=variant, layout=layout):
                        vocab = list(atomic_vocabulary(variant))
                        core = GPT(vocab=vocab, context_size=64, number_of_layers=2,
                            number_of_heads=2, number_of_embeddings=16, mlp_ratio=2, dropout=0,
                            position_encoding='rope', max_beat_positions=8, max_beats_per_cycle=4,
                            attention_layout=layout, max_tokens_per_beat=8,
                            max_beat_slots_per_chunk=16, max_cycles_per_chunk=8)
                        report = validate_core(core, vocab, variant)
                        self.assertTrue(report['passed'])
                        self.assertEqual(report['disabled_stream_checks'], 3 if layout != 'global' else 0)
        finally:
            torch.set_num_threads(old_threads)

    def test_static_representations_follow_type_projection(self):
        import torch
        from ldm.embedding_analysis import EmbeddingRepresentation as Mode, representation_matrix
        core = types.SimpleNamespace(token_embeddings=torch.nn.Embedding(2, 2),
            type_embeddings=torch.nn.Embedding(2, 2), token_type_ids=torch.tensor([0, 1]),
            head=torch.nn.Linear(2, 2, bias=False), use_type_embeddings=True,
            concat_type_embeddings=False)
        model = types.SimpleNamespace(core=core)
        torch.testing.assert_close(representation_matrix(model, Mode.BASE_INPUT),
            core.token_embeddings.weight + core.type_embeddings(core.token_type_ids))
        core.concat_type_embeddings = True
        core.type_concat_projection = torch.nn.Linear(4, 2)
        expected = core.type_concat_projection(torch.cat((core.token_embeddings.weight,
            core.type_embeddings(core.token_type_ids)), dim=-1))
        torch.testing.assert_close(representation_matrix(model, Mode.BASE_INPUT), expected)
        torch.testing.assert_close(representation_matrix(model, Mode.OUTPUT_HEAD), core.head.weight)
        torch.testing.assert_close(representation_matrix(model, Mode.TOKEN_ONLY), core.token_embeddings.weight)

    def test_profile_comparison_checks_hardware_and_data(self):
        from ldm.model_checks import compare_profiles, profile_cuda
        from unittest.mock import patch
        import torch
        with patch.object(torch.cuda, 'is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'GPU runtime'):
                profile_cuda('unused', 'unused', 'unused')
        keys = ('number_of_layers', 'number_of_heads', 'number_of_embeddings', 'mlp_ratio',
                'dropout', 'position_encoding', 'rope_theta', 'rope_fraction', 'tie_weights',
                'use_type_embeddings', 'concat_type_embeddings', 'dataset_variant', 'vocab_hash',
                'activation_name', 'max_beat_positions', 'max_beats_per_cycle', 'max_tokens_per_beat',
                'max_beat_slots_per_chunk', 'max_cycles_per_chunk')
        config = dict.fromkeys(keys, 1)
        baseline = {'profile_version': 1, 'device_name': 'test GPU', 'torch_version': 'test',
            'cuda_version': 'test', 'dataset_fingerprint': 'data', 'batch_size': 1,
            'context_size': 32, 'amp': True, 'warmup_steps': 3, 'measured_steps': 10,
            'device_capability': [8, 0], 'allow_tf32': False, 'cudnn_allow_tf32': False, 'matmul_precision': 'highest',
            'sdpa_backends': {}, 'config': {**config, 'attention_layout': 'global'},
            'mean_step_ms': 10, 'peak_allocated_gib': 2}
        candidate = {**baseline, 'config': {**config, 'attention_layout': 'alternating_global_structured'},
                     'mean_step_ms': 8, 'peak_allocated_gib': 1}
        self.assertTrue(compare_profiles(baseline, candidate)['optimization_gate_passed'])
        with self.assertRaisesRegex(ValueError, 'dataset_fingerprint'):
            compare_profiles(baseline, {**candidate, 'dataset_fingerprint': 'different'})


@unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'Matplotlib runtime required')
class ComparisonPlotTests(unittest.TestCase):
    def test_plot_saves_dataset_groups_and_accuracy_values(self):
        import matplotlib
        matplotlib.use('Agg')
        from ldm.report_ldm import save_test_accuracy_comparison
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root = Path(folder)
            runs = []
            for i, accuracy in enumerate((.5, .8)):
                run = root / f'run{i}'
                run.mkdir()
                (run / 'experiment.json').write_text(json.dumps({'run_id': run.name, 'status': 'completed',
                    'config_fingerprint': 'config', 'dataset_fingerprint': f'data{i}',
                    'config': {'data': {'variant': 'with_subd'}}}), encoding='utf-8')
                (run / 'metrics.json').write_text(json.dumps({'test_report': {'metric_version': 2,
                    'combined': {'token_accuracy': accuracy}}}), encoding='utf-8')
                (run / 'dataset_ref.json').write_text(json.dumps({'split_version': 'v1'}), encoding='utf-8')
                runs.append(run)
            output = save_test_accuracy_comparison(runs, root / 'accuracy.png')
            self.assertGreater(output.stat().st_size, 0)
            import csv
            with output.with_suffix('.csv').open(encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row['comparison_group'] for row in rows], ['1', '2'])


if __name__ == '__main__':
    unittest.main()
