import os
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
from label_session import import_folder, save_labels


def click(app, label):
    next(button for button in app.button if button.label == label).click().run()
    if app.exception:
        raise AssertionError(app.exception[0].message)


class LabelUiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.library = self.root / 'Original pack'
        self.library.mkdir()
        for name in ['a.wav', 'b.wav']:
            sf.write(self.library / name, np.sin(np.arange(4096) * .2) * .2, 22050)
        self.env = patch.dict(os.environ, {'SORTER_DATA_DIR': str(self.root / 'labels')})
        self.env.start()
        self.argv = patch.object(sys, 'argv', ['label.py'])
        self.argv.start()

    def tearDown(self):
        self.argv.stop()
        self.env.stop()
        self.temp.cleanup()

    def app(self):
        app = AppTest.from_file(str(Path(ml.__file__).with_name('label.py'))).run()
        self.assertFalse(app.exception)
        return app

    def imported_app(self):
        app = self.app()
        app.text_input(key='folder').set_value(str(self.library))
        click(app, 'Import / resume folder')
        self.assertEqual(app.subheader[0].value, 'a.wav')
        return app

    def test_import_label_auto_advance_and_resume(self):
        app = self.imported_app()
        next(s for s in app.selectbox if s.label == 'Instrument').select('kick')
        app.multiselect[0].set_value(['soft', 'bright', 'acoustic'])
        click(app, 'Save & next →')
        self.assertEqual(app.subheader[0].value, 'b.wav')
        self.assertEqual(app.session_state['session_saved'], 1)
        rows = ml.read_rows(app.session_state['manifest'])
        self.assertEqual(rows[0]['reviewed'], '1')
        self.assertEqual(rows[0]['pack'], 'Original pack')
        self.assertEqual(rows[0]['instrument'], 'kick')
        self.assertEqual(rows[0]['soft'], '1')
        self.assertEqual(rows[0]['bright'], '1')
        self.assertEqual(rows[0]['acoustic'], '1')
        self.assertEqual(rows[0]['distorted'], '0')
        self.assertEqual(rows[1]['reviewed'], '0')
        self.assertEqual(app.multiselect[0].value, [])
        resumed = self.app()
        click(resumed, 'Resume saved dataset')
        self.assertEqual(resumed.subheader[0].value, 'b.wav')

    def test_skip_stays_unlabeled_and_revisit(self):
        app = self.imported_app()
        click(app, 'Skip for now →')
        self.assertEqual(app.subheader[0].value, 'b.wav')
        click(app, 'Skip for now →')
        self.assertFalse(app.subheader)
        self.assertTrue(all(r['reviewed'] == '0' for r in ml.read_rows(app.session_state['manifest'])))
        self.assertEqual(app.session_state['session_saved'], 0)
        click(app, 'Revisit skipped samples')
        self.assertEqual(app.subheader[0].value, 'a.wav')

    def test_reuse_previous_tags_is_editable_and_does_not_save(self):
        app = self.imported_app()
        self.assertTrue(app.button(key='use_previous').disabled)
        next(s for s in app.selectbox if s.label == 'Instrument').select('snare')
        app.multiselect[0].set_value(['soft', 'acoustic'])
        click(app, 'Save & next →')
        self.assertEqual(app.multiselect[0].value, [])
        click(app, 'Use previous tags')
        self.assertEqual(next(s for s in app.selectbox if s.label == 'Instrument').value, 'snare')
        self.assertEqual(app.multiselect[0].value, ['soft', 'acoustic'])
        self.assertEqual(ml.read_rows(app.session_state['manifest'])[1]['reviewed'], '0')
        app.multiselect[0].set_value(['soft', 'bright'])
        click(app, 'Save & next →')
        row = ml.read_rows(app.session_state['manifest'])[1]
        self.assertEqual(row['soft'], '1')
        self.assertEqual(row['bright'], '1')
        self.assertEqual(row['acoustic'], '0')

    def test_empty_previous_tags_can_clear_selection(self):
        app = self.imported_app()
        next(s for s in app.selectbox if s.label == 'Instrument').select('kick')
        click(app, 'Save & next →')
        app.multiselect[0].set_value(['bright']).run()
        click(app, 'Use previous tags')
        self.assertEqual(app.multiselect[0].value, [])

    def test_save_without_confirmation_and_relabel_does_not_inflate_progress(self):
        app = self.imported_app()
        next(s for s in app.selectbox if s.label == 'Instrument').select('kick')
        click(app, 'Save & next →')
        self.assertFalse(any(c.label == 'I listened and can judge these traits' for c in app.checkbox))
        app.selectbox(key='review_sample').select(str(self.library / 'a.wav'))
        click(app, 'Review this sample')
        click(app, 'Save & next →')
        self.assertEqual(app.session_state['session_saved'], 1)

    def test_instrument_required_but_timbre_can_stay_unlabeled(self):
        app = self.imported_app()
        click(app, 'Save & next →')
        self.assertTrue(app.error)
        self.assertEqual(app.session_state['session_saved'], 0)
        next(s for s in app.selectbox if s.label == 'Instrument').select('piano')
        click(app, 'Save & next →')
        row = ml.read_rows(app.session_state['manifest'])[0]
        self.assertEqual(row['instrument'], 'piano')
        self.assertEqual(row['traits_reviewed'], '0')
        self.assertTrue(all(row[tag] == '' for tag in ml.TAGS))

    def test_explicit_no_timbre_traits_and_conflicting_selection(self):
        app = self.imported_app()
        next(s for s in app.selectbox if s.label == 'Instrument').select('snare')
        app.multiselect[0].set_value(['__none__', 'bright'])
        click(app, 'Save & next →')
        self.assertTrue(app.error)
        app.multiselect[0].set_value(['__none__'])
        click(app, 'Save & next →')
        row = ml.read_rows(app.session_state['manifest'])[0]
        self.assertEqual(row['traits_reviewed'], '1')
        self.assertTrue(all(row[tag] == '0' for tag in ml.TAGS))

    def test_unreadable_audio_can_be_skipped(self):
        (self.library / 'a.wav').write_bytes(b'bad audio')
        app = self.imported_app()
        self.assertTrue(app.error)
        self.assertFalse(any(b.label == 'Save & next →' for b in app.button))
        click(app, 'Skip for now →')
        self.assertEqual(app.subheader[0].value, 'b.wav')

    def test_import_merge_and_label_preservation(self):
        manifest, count = import_folder(self.library)
        self.assertEqual(count, 2)
        save_labels(manifest, str(self.library / 'a.wav'), 'Pack A', ['soft'])
        manifest2, count = import_folder(self.library)
        self.assertEqual(manifest, manifest2)
        self.assertEqual(count, 0)
        self.assertEqual(ml.read_rows(manifest)[0]['soft'], '1')
        other = self.root / 'Another pack'
        other.mkdir()
        sf.write(other / 'c.wav', np.zeros(64), 22050)
        _, count = import_folder(other, manifest=manifest, pack_name='Pack B')
        self.assertEqual(count, 1)
        rows = ml.read_rows(manifest)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[-1]['pack'], 'Pack B')
        with self.assertRaisesRegex(ValueError, 'Choose a sample folder'):
            import_folder('')

    def test_folder_picker_and_completion(self):
        app = self.app()
        with patch('label_session.choose_folder', return_value=str(self.library)):
            click(app, 'Browse folders…')
        self.assertEqual(app.text_input(key='folder').value, str(self.library))
        click(app, 'Import / resume folder')
        for _ in range(2):
            next(s for s in app.selectbox if s.label == 'Instrument').select('kick')
            click(app, 'Save & next →')
        self.assertTrue(any('Library complete' in message.value for message in app.success))
        self.assertEqual(app.session_state['session_saved'], 2)
        self.assertTrue(all(r['reviewed'] == '1' for r in ml.read_rows(app.session_state['manifest'])))


if __name__ == '__main__':
    unittest.main()
