"""Synthetic fixtures verify the pipeline; they do not measure real-world tagging accuracy."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sorter_ml as ml


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def audio(self, name, seed=1, bright=False):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(seed)
        t = np.arange(4096) / 22050
        audio = .2 * np.sin(2 * np.pi * (5000 if bright else 80) * t) + rng.normal(0, .005, len(t))
        sf.write(path, audio, 22050)
        return path

    def test_manifest_preserves_unreviewed_and_existing_labels(self):
        self.audio('library/Pack A/kick.wav')
        manifest = self.root / 'labels.csv'
        self.assertEqual(ml.init_manifest(self.root / 'library', manifest), 1)
        row = ml.read_rows(manifest)[0]
        self.assertEqual(row['pack'], 'Pack A')
        self.assertEqual(row['reviewed'], '0')
        self.assertTrue(all(row[t] == '' for t in ml.TAGS))
        with self.assertRaisesRegex(ValueError, 'exists'):
            ml.init_manifest(self.root / 'library', manifest)

    def test_features_read_audio_not_names_and_support_silence(self):
        path = self.audio('bright.wav')
        renamed = self.root / 'dark.wav'
        renamed.write_bytes(path.read_bytes())
        np.testing.assert_array_equal(ml.features(path), ml.features(renamed))
        sf.write(self.root / 'silence.wav', np.zeros((64, 2)), 44100)
        self.assertTrue(np.isfinite(ml.features(self.root / 'silence.wav')).all())
        empty = self.root / 'empty.wav'
        sf.write(empty, np.zeros(0), 22050)
        with self.assertRaisesRegex(ValueError, 'Empty'):
            ml.features(empty)

    def test_too_few_labels_rejected(self):
        manifest = self.root / 'labels.csv'
        ml.write_rows(manifest, [])
        with self.assertRaisesRegex(ValueError, 'at least 30'):
            ml.train(manifest, self.root / 'model', self.root / 'report')

    def fixture_rows(self):
        rows = []
        for pack in range(10):
            for sample in range(10):
                positive = sample % 2
                path = self.audio(f'library/pack-{pack}/{sample}.wav', seed=100 * pack + sample, bright=positive)
                rows.append(dict(path=str(path), pack=f'pack-{pack}', reviewed='1', **{t: str(positive) for t in ml.TAGS}))
        return rows

    def test_training_held_out_packs_and_prediction(self):
        rows = self.fixture_rows()
        manifest = self.root / 'labels.csv'
        ml.write_rows(manifest, rows)
        model, report = self.root / 'model.joblib', self.root / 'report.json'
        result = ml.train(manifest, model, report)
        self.assertEqual(result['samples'], 100)
        packs = [set(split['packs']) for split in result['splits'].values()]
        for i in range(3):
            for j in range(i):
                self.assertFalse(packs[i] & packs[j])
        self.assertEqual(set(result['metrics']), set(ml.TAGS))
        unseen = self.audio('unseen/sample.wav', seed=4000, bright=True)
        before = unseen.read_bytes()
        (unseen.parent / 'bad.wav').write_bytes(b'not audio')
        output = self.root / 'predictions.json'
        self.assertEqual(ml.predict(model, unseen.parent, output), 2)
        predictions = json.loads(output.read_text())
        valid = next(row for row in predictions if row['path'] == str(unseen))
        self.assertEqual(set(valid['scores']), set(ml.TAGS))
        self.assertTrue(valid['needs_review'])
        self.assertTrue(any('error' in row for row in predictions))
        self.assertEqual(unseen.read_bytes(), before)
        subset = ml.train(manifest, model, report, ['soft', 'bright'])
        self.assertEqual(set(subset['metrics']), {'soft', 'bright'})
        ml.predict(model, unseen.parent, output)
        valid = next(row for row in json.loads(output.read_text()) if row['path'] == str(unseen))
        self.assertEqual(set(valid['scores']), {'soft', 'bright'})

    def test_missing_class_is_not_reported_as_success(self):
        rows = self.fixture_rows()
        for row in rows:
            row['acoustic'] = '0'
        manifest = self.root / 'labels.csv'
        ml.write_rows(manifest, rows)
        with self.assertRaisesRegex(ValueError, 'acoustic: need positive AND negative'):
            ml.train(manifest, self.root / 'model', self.root / 'report', ['acoustic'])
        self.assertFalse((self.root / 'model').exists())

    def test_duplicate_audio_and_incomplete_labels_rejected(self):
        rows = self.fixture_rows()
        manifest = self.root / 'labels.csv'
        rows[0]['soft'] = ''
        ml.write_rows(manifest, rows)
        with self.assertRaisesRegex(ValueError, 'explicit 0/1'):
            ml.train(manifest, self.root / 'model', self.root / 'report')
        rows[0]['soft'] = '0'
        Path(rows[1]['path']).write_bytes(Path(rows[0]['path']).read_bytes())
        ml.write_rows(manifest, rows)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            ml.train(manifest, self.root / 'model', self.root / 'report')


if __name__ == '__main__':
    unittest.main()
