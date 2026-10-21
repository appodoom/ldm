"""Exercise verified data, training/resume, HF snapshots, and embedding reports."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ('torch', 'transformers')),
                     'Torch and Transformers required')
class TrainingIntegrationTests(unittest.TestCase):
    def test_interrupted_resume_matches_uninterrupted_with_dropout(self):
        import torch
        from unittest.mock import patch
        from ldm.derbake_ldm import Beat, RhythmSequence, write_derbake
        from ldm.experiment_config import RunConfig, DataConfig, ModelConfig, TrainConfig, EvalConfig, AttentionLayout, DecorationMode
        from ldm.experiment_workflow import prepare_or_load_dataset, model_config_for_run, _train_kwargs
        from ldm.train_ldm import train, _save
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
                root = Path(folder)
                source = root / 'source'
                source.mkdir()
                write_derbake(RhythmSequence(tuple((Beat(2, ('HIT_D', 'HIT_S')),)
                                                   for _ in range(20))), source / 'a.derbake')
                config = RunConfig(data=DataConfig(context_size=16, stride=8, max_beats_per_cycle=4, decoration=DecorationMode.WITHOUT_DECORATION),
                    model=ModelConfig(layers=1, heads=2, embeddings=16, mlp_ratio=2,
                                      dropout=0.2, attention=AttentionLayout.GLOBAL),
                    train=TrainConfig(epochs=2, batch_size=2, amp=False, save_every_steps=1, log_every_steps=50),
                    evaluation=EvalConfig(batch_size=2, every_steps=100))
                dataset = prepare_or_load_dataset(config, source, root / 'cache')
                model_config = model_config_for_run(config, dataset)
                kwargs = _train_kwargs(config)
                uninterrupted = train(dataset, root / 'full', model_config, **kwargs, device='cpu')
                def interrupt_after_checkpoint(path, payload):
                    _save(path, payload)
                    if path.name == 'last.pt' and payload['step'] == 1:
                        raise KeyboardInterrupt('simulated interruption')
                with patch('ldm.train_ldm._save', side_effect=interrupt_after_checkpoint):
                    with self.assertRaises(KeyboardInterrupt):
                        train(dataset, root / 'resume', model_config, **kwargs, device='cpu')
                resumed = train(dataset, root / 'resume', model_config, **kwargs, device='cpu', resume=True)
                self.assertEqual(uninterrupted['steps'], resumed['steps'])
                self.assertEqual(uninterrupted['history'], resumed['history'])
                full_state = torch.load(root / 'full' / 'last.pt', weights_only=False)
                resume_state = torch.load(root / 'resume' / 'last.pt', weights_only=False)
                for name, tensor in full_state['model'].items():
                    torch.testing.assert_close(tensor, resume_state['model'][name], atol=0, rtol=0)
                limited = {**kwargs, 'max_steps': 1}
                first = train(dataset, root / 'limited', model_config, **limited, device='cpu')
                second = train(dataset, root / 'limited', model_config, **limited, device='cpu', resume=True)
                self.assertEqual(first['steps'], second['steps'])
                with self.assertRaisesRegex(ValueError, 'Target epochs'):
                    train(dataset, root / 'full', model_config, **{**kwargs, 'epochs': 1}, device='cpu', resume=True)
        finally:
            torch.set_num_threads(old_threads)

    def test_all_data_modes_train_resume_export_and_generate(self):
        import torch
        from ldm.derbake_ldm import Beat, RhythmSequence, write_derbake
        from ldm.experiment_config import (RunConfig, DataConfig, ModelConfig, TrainConfig,
                                           EvalConfig, DatasetVariant, AttentionLayout, DecorationMode)
        from ldm.experiment_workflow import prepare_or_load_dataset, start_run, resume_run
        from ldm.hub_ldm import build_local_snapshot
        from ldm.embedding_analysis import EmbeddingRepresentation, analyze_static_embeddings
        from ldm.model_checks import run_model_checks
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
                root = Path(folder)
                for group, hit in (('normal', 'HIT_D'), ('decorated', 'HIT_D_decorated')):
                    (root / group).mkdir()
                    rhythm = RhythmSequence(tuple((Beat(2, (hit, 'HIT_S')),
                                                    Beat(1, ('HIT_OTA',))) for _ in range(10 if group == "normal" else 12)))
                    write_derbake(rhythm, root / group / f'{group}_only.derbake')
                from itertools import product
                for (variant, layout), decoration in product(
                        ((DatasetVariant.WITH_SUBD, AttentionLayout.ALTERNATING_GLOBAL_STRUCTURED),
                         (DatasetVariant.NO_SUBD, AttentionLayout.GLOBAL)), DecorationMode):
                    config = RunConfig(data=DataConfig(variant=variant, decoration=decoration, context_size=32, stride=16,
                                                       max_beats_per_cycle=4),
                        model=ModelConfig(layers=2, heads=2, embeddings=16, mlp_ratio=2,
                                          dropout=0, attention=layout),
                        train=TrainConfig(epochs=1, batch_size=2, amp=False, log_every_steps=2,
                                          save_every_steps=2),
                        evaluation=EvalConfig(batch_size=2, every_steps=2))
                    data = prepare_or_load_dataset(config, root / decoration.value, root / 'cache')
                    run = start_run(config, data, root / 'runs', device='cpu')
                    self.assertTrue(json.loads((run / 'model_checks.json').read_text())['passed'])
                    runtime = json.loads((run / 'runtime_settings.json').read_text())
                    self.assertEqual(runtime['num_workers'], 0)
                    self.assertFalse(runtime['amp'])
                    first = json.loads((run / 'metrics.json').read_text())
                    self.assertEqual(first['test_report']['metric_version'], 3)
                    self.assertEqual(first['test_report']['decoration'], decoration.value)
                    self.assertIn('correct_plain_hit', first['test_report']['combined']['decorated_outcomes'])
                    record = json.loads((run / 'experiment.json').read_text())
                    record['last_error'] = 'older failed attempt'
                    (run / 'experiment.json').write_text(json.dumps(record))
                    resumed = resume_run(run, target_epochs=2, device='cpu')
                    self.assertNotIn('last_error', json.loads((run / 'experiment.json').read_text()))
                    self.assertGreater(resumed['steps'], first['steps'])
                    snapshot = root / 'snapshots' / f'{variant.value}-{decoration.value}'
                    build_local_snapshot(run, snapshot)
                    self.assertTrue((snapshot / "constants.py").exists())
                    self.assertTrue((snapshot / "validation.py").exists())
                    reports = analyze_static_embeddings(snapshot, run / 'analysis',
                        representations=list(EmbeddingRepresentation), top_k=3)
                    self.assertEqual(len(reports), 3)
                    self.assertEqual(reports[0]['metrics']['subdivision_analysis_applicable'],
                                     variant is DatasetVariant.WITH_SUBD)
                    self.assertTrue(run_model_checks(snapshot, run / 'validation')['passed'])
                    # Exercise the exported code in a fresh process without the source package.
                    import subprocess, sys, inspect, importlib, warnings, os
                    from unittest.mock import patch
                    from transformers import AutoModel
                    from ldm.inference_ldm import run_inference, InferenceConfig, contextual_projection
                    from ldm.derbake_ldm import parse_derbake
                    isolated_env = {**os.environ, 'TMP': str(root), 'TEMP': str(root)}
                    isolated = subprocess.run([sys.executable, '-I', str(Path(__file__).with_name('hf_snapshot_smoke.py')),
                                               str(snapshot)], capture_output=True, text=True, env=isolated_env)
                    self.assertEqual(isolated.returncode, 0, isolated.stderr)
                    model = AutoModel.from_pretrained(snapshot, trust_remote_code=True).eval()
                    prompt = root / f'prompt_{variant.value}.derbake'
                    write_derbake(RhythmSequence(((Beat(1, ('HIT_D',)),),)), prompt)
                    expected = RhythmSequence(((Beat(1, ('HIT_D_decorated',)),),))
                    # Long orphan prefix forces a cache refresh, followed by an invalid
                    # cycle and one accepted cycle. Sampling alone is scripted.
                    from ldm.schema_ldm import RhythmSchema
                    oversized = RhythmSequence(((Beat(1, ('HIT_D',)),) * 5,))
                    atoms = (['<SOB>', 'HIT_D', '<EOB>', '<PAD>'] * 8 + ['<SOC>', '<EOC>']
                             + RhythmSchema(variant.value, 8).encode(oversized)
                             + RhythmSchema(variant.value, 4).encode(expected))
                    sampled = iter(model.vocab.index(atom) for atom in atoms)
                    generation_module = importlib.import_module(model.__class__.__module__)
                    generator = inspect.unwrap(generation_module.generate_continuation_ids)
                    with patch.object(AutoModel, 'from_pretrained', return_value=model), \
                         patch('torch.cuda.is_available', return_value=False), \
                         patch.dict(generator.__globals__, {'sample_next_id': lambda **kwargs: next(sampled)}), \
                         warnings.catch_warnings(record=True) as caught:
                        report = run_inference(snapshot, prompt, root / 'inference',
                            InferenceConfig(num_cycles=1, max_new_tokens=128, cache_refresh_tokens=8))
                    self.assertEqual(parse_derbake(report['output']).rhythm, expected)
                    self.assertEqual(report['generation']['accepted_cycles'], 1)
                    self.assertEqual(report['generation']['discarded_cycles'], 2)
                    self.assertEqual(report['generation']['orphan_tokens'], 32)
                    self.assertTrue(any('skipped' in str(w.message) for w in caught))
                    self.assertTrue((Path(report['output']).parent / 'report.json').exists())
                    contextual = contextual_projection(snapshot, data, run / 'contextual', max_rows=1, max_points=10)
                    self.assertTrue(contextual.exists())

        finally:
            torch.set_num_threads(old_threads)


if __name__ == '__main__':
    unittest.main()
