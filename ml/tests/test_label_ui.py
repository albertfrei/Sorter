import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sorter_ml as ml


class LabelUiTests(unittest.TestCase):
    def test_listen_label_and_save(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library = root / 'library' / 'Original pack'
            library.mkdir(parents=True)
            sf.write(library / 'sample.wav', np.sin(np.arange(4096) * .2) * .2, 22050)
            manifest = root / 'labels.csv'
            ml.init_manifest(root / 'library', manifest)
            with patch.object(sys, 'argv', ['label.py', str(manifest)]):
                app = AppTest.from_file(str(Path(ml.__file__).with_name('label.py'))).run()
                self.assertFalse(app.exception)
                self.assertEqual(app.subheader[0].value, 'sample.wav')
                app.multiselect[0].set_value(['soft', 'bright', 'acoustic'])
                app.button[0].click().run()
                self.assertFalse(app.exception)
            row = ml.read_rows(manifest)[0]
            self.assertEqual(row['reviewed'], '1')
            self.assertEqual(row['pack'], 'Original pack')
            self.assertEqual(row['soft'], '1')
            self.assertEqual(row['bright'], '1')
            self.assertEqual(row['acoustic'], '1')
            self.assertEqual(row['distorted'], '0')

    def test_unreadable_audio_does_not_offer_label_save(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'bad.wav').write_bytes(b'bad audio')
            manifest = root / 'labels.csv'
            ml.init_manifest(root, manifest)
            with patch.object(sys, 'argv', ['label.py', str(manifest)]):
                app = AppTest.from_file(str(Path(ml.__file__).with_name('label.py'))).run()
                self.assertFalse(app.exception)
                self.assertTrue(app.error)
                self.assertEqual(len(app.button), 0)
            self.assertEqual(ml.read_rows(manifest)[0]['reviewed'], '0')


if __name__ == '__main__':
    unittest.main()
