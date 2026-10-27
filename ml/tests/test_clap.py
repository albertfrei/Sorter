"""Offline contracts and pipeline integration; no pretrained accuracy claims."""
import contextlib
import importlib.util
import json
import shutil
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import joblib
import numpy as np
import soundfile as sf

import test_pipeline
import sorter_ml as ml
from clap_features import ClapFeatures, checkpoint_fingerprint, read_audio


class ClapTests(unittest.TestCase):
    setUp = test_pipeline.PipelineTests.setUp
    tearDown = test_pipeline.PipelineTests.tearDown
    audio = test_pipeline.PipelineTests.audio
    fixture_rows = test_pipeline.PipelineTests.fixture_rows
    def checkpoint(self):
        directory = self.root / 'checkpoint'
        directory.mkdir()
        (directory / 'model.safetensors').write_bytes(b'test weights, not a real checkpoint')
        (directory / 'config.json').write_text('{}')
        (directory / 'preprocessor_config.json').write_text('{}')
        return directory

    def test_clap_combined_training_prediction_and_checkpoint_binding(self):
        directory = self.checkpoint()
        manifest = self.root / 'labels.csv'
        rows = self.fixture_rows()
        for row in rows:
            row['instrument'] = 'snare' if row['bright'] == '1' else 'kick'
        ml.write_rows(manifest, rows)
        manifest_before = manifest.read_bytes()
        calls = []

        class FakeClap:
            def __init__(self, path, expected_fingerprint=None):
                digest = checkpoint_fingerprint(path)
                if expected_fingerprint and digest != expected_fingerprint:
                    raise ValueError('CLAP checkpoint mismatch')
                self.config = dict(backend='clap', version=1, model_path=str(path), checkpoint_sha256=digest)
                calls.append(str(path))

            def __call__(self, path):
                return ml.features(path)[:8]

        model, report, output = (self.root / name for name in ['model.joblib', 'report.json', 'predictions.json'])
        with patch('clap_features.ClapFeatures', FakeClap):
            result = ml.train(manifest, model, report, ['bright'], 'clap', directory)
            self.assertEqual(result['feature_config']['backend'], 'clap')
            self.assertEqual(result['instrument']['status'], 'trained')
            self.assertIn('bright', result['metrics'])
            self.assertEqual(joblib.load(model)['feature_config'], result['feature_config'])
            unseen = self.audio('unseen/sample.wav', seed=6000, bright=True)
            (unseen.parent / 'bad.wav').write_bytes(b'invalid')
            before = unseen.read_bytes()
            relocated = self.root / 'relocated'
            shutil.copytree(directory, relocated)
            ml.predict(model, unseen.parent, output, relocated)
            predictions = json.loads(output.read_text())
            valid = next(p for p in predictions if 'error' not in p)
            self.assertEqual(set(valid['instrument_scores']), {'kick', 'snare'})
            self.assertEqual(set(valid['scores']), {'bright'})
            self.assertTrue(valid['needs_review'])
            self.assertTrue(any('error' in p for p in predictions))
            self.assertEqual(unseen.read_bytes(), before)
            self.assertEqual(len(calls), 2)  # one encoder per command, not per sample
            (relocated / 'model.safetensors').write_bytes(b'different weights')
            with self.assertRaisesRegex(ValueError, 'checkpoint mismatch'):
                ml.predict(model, unseen.parent, self.root / 'should-not-exist.json', relocated)
            self.assertFalse((self.root / 'should-not-exist.json').exists())
        self.assertEqual(manifest.read_bytes(), manifest_before)

    def test_legacy_artifact_without_backend_metadata(self):
        manifest = self.root / 'labels.csv'
        ml.write_rows(manifest, self.fixture_rows())
        model = self.root / 'model.joblib'
        ml.train(manifest, model, self.root / 'report.json', ['bright'])
        artifact = joblib.load(model)
        del artifact['feature_config']
        joblib.dump(artifact, model)
        audio = self.audio('unseen/test.wav', seed=7000)
        with patch('clap_features.ClapFeatures', side_effect=AssertionError('baseline must not load CLAP')):
            self.assertEqual(ml.predict(model, audio.parent, self.root / 'output.json'), 1)

    def test_backend_validation(self):
        with self.assertRaisesRegex(ValueError, 'requires --clap-model'):
            ml.feature_extractor('clap')
        with self.assertRaisesRegex(ValueError, 'requires the clap'):
            ml.feature_extractor(clap_model='anything')
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            ml.feature_extractor('invalid')
        with self.assertRaisesRegex(ValueError, 'version mismatch'):
            ml.feature_extractor(saved=dict(backend='clap', version=99))
        with self.assertRaisesRegex(ValueError, 'existing local'):
            checkpoint_fingerprint(self.root / 'missing')
        with self.assertRaisesRegex(ValueError, 'no model weights'):
            checkpoint_fingerprint(self.root)

    def test_clap_audio_is_mono_resampled_bounded_and_validated(self):
        path = self.root / 'long.wav'
        sf.write(path, np.full((44100 * 11, 2), [.2, .4]), 44100, subtype='FLOAT')
        samples = read_audio(path)
        self.assertEqual(samples.shape, (480000,))
        self.assertAlmostEqual(float(samples[1000:-1000].mean()), .3, places=3)
        sf.write(path, np.zeros((64, 2)), 22050)
        self.assertTrue(np.isfinite(read_audio(path)).all())
        sf.write(path, np.zeros(0), 48000)
        with self.assertRaisesRegex(ValueError, 'Empty'):
            read_audio(path)

    def test_adapter_loads_offline_once_and_normalizes_embeddings(self):
        directory = self.checkpoint()
        audio = self.audio('test.wav')
        from unittest.mock import Mock
        tensor = Mock()
        tensor.cpu.return_value.numpy.return_value = np.array([3., 4.])
        model = Mock()
        model.to.return_value = model
        model.eval.return_value = model
        model.config.enable_fusion = False
        model.return_value = SimpleNamespace(audio_embeds=[tensor])
        processor = Mock(sampling_rate=48000, truncation='rand_trunc')
        processor.return_value = {'input_features': 'test'}
        model_loader = Mock()
        model_loader.from_pretrained.return_value = (model, {})
        processor_loader = Mock()
        processor_loader.from_pretrained.return_value = processor
        torch = SimpleNamespace(inference_mode=contextlib.nullcontext)
        transformers = SimpleNamespace(ClapAudioModelWithProjection=model_loader, ClapFeatureExtractor=processor_loader)
        with patch.dict(sys.modules, torch=torch, transformers=transformers):
            extractor = ClapFeatures(directory)
            np.testing.assert_allclose(extractor(audio), [.6, .8])
            extractor(audio)
            model_loader.from_pretrained.assert_called_once_with(str(directory.resolve()), local_files_only=True, output_loading_info=True)
            processor_loader.from_pretrained.assert_called_once_with(str(directory.resolve()), local_files_only=True)
            model.to.assert_called_once_with('cpu')
            model.eval.assert_called_once()
            self.assertEqual(processor.call_args.kwargs['sampling_rate'], 48000)
            tensor.cpu.return_value.numpy.return_value = np.array([np.nan, 0.])
            with self.assertRaisesRegex(ValueError, 'invalid embedding'):
                extractor(audio)
            processor.truncation = 'fusion'
            with self.assertRaisesRegex(ValueError, 'unfused'):
                ClapFeatures(directory)
            processor.truncation = 'rand_trunc'
            model_loader.from_pretrained.return_value = (model, {'missing_keys': ['audio_projection.weight']})
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                ClapFeatures(directory)

    @unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('transformers'),
                         'optional CLAP dependencies are not installed')
    def test_real_transformers_audio_forward_offline(self):
        # Small random model checks the actual library API, not pretrained accuracy.
        import torch
        from transformers import ClapAudioConfig, ClapAudioModelWithProjection, ClapFeatureExtractor
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            directory = self.root / 'tiny-clap'
            config = ClapAudioConfig(hidden_size=64, patch_embeds_hidden_size=8,
                                     depths=[1, 1, 1, 1], num_attention_heads=[1, 2, 4, 8], projection_dim=16)
            ClapAudioModelWithProjection(config).save_pretrained(directory)
            ClapFeatureExtractor(truncation='rand_trunc').save_pretrained(directory)
            extractor = ClapFeatures(directory)
            audio = self.audio('actual-forward.wav')
            first, second = extractor(audio), extractor(audio)
            self.assertEqual(first.shape, (16,))
            self.assertAlmostEqual(float(np.linalg.norm(first)), 1., places=5)
            np.testing.assert_array_equal(first, second)
        finally:
            torch.set_num_threads(previous_threads)
