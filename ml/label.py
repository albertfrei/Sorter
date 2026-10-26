"""Run with: streamlit run label.py -- path/to/labels.csv"""
import sys
from pathlib import Path
import streamlit as st
import soundfile as sf
from sorter_ml import TAGS, read_rows, write_rows

st.set_page_config(page_title='Sorter · Label sounds', page_icon='🎧', layout='centered')
st.title('Listen. Describe. Train.')
st.caption('All audio and labels stay on this computer. Label what you hear, not the filename.')
with st.expander('How to describe a sound'):
    st.markdown('''
**Soft:** gentle, rounded onset. **Punchy:** strong, distinct attack.

**Bright:** prominent highs. **Dark:** subdued highs. **Warm:** full low-mid body, smooth highs.

**Acoustic:** physical instrument/recording character. **Electronic:** synthesized/electronic character.

**Clean:** little audible noise or saturation. **Distorted:** saturation, clipping or fuzz.
**Noisy:** hiss, hum or background noise. Intentional effects still count.

Traits can overlap. Keep playback volume consistent; soft does not simply mean quiet.
''')
if len(sys.argv) < 2:
    st.error('Supply the manifest: streamlit run label.py -- labels.csv')
    st.stop()
manifest = Path(sys.argv[1]).resolve()
rows = read_rows(manifest)
if not rows:
    st.info('No audio files in this manifest.'); st.stop()
reviewed = sum(r['reviewed'] == '1' for r in rows)
st.progress(reviewed / len(rows), text=f'{reviewed} / {len(rows)} reviewed')
index = st.number_input('Sample number', min_value=1, max_value=len(rows), step=1)
row = rows[index - 1]
st.subheader(Path(row['path']).name)
st.caption(row['path'])
try:
    with sf.SoundFile(row['path']) as audio:
        clip = audio.read(min(len(audio), audio.samplerate * 12), dtype='float32', always_2d=True)
        st.audio(clip.T, sample_rate=audio.samplerate)
    st.caption('Playback and training use the first 12 seconds. Keep this initial dataset to drum one-shots.')
except (OSError, RuntimeError, ValueError) as ex:
    st.error(str(ex))
    st.stop()
with st.form(f'sample-{index}'):
    pack = st.text_input('Original sample pack (used to keep evaluation packs separate)', value=row['pack'])
    selected = st.multiselect('Traits you actually hear', TAGS, default=[t for t in TAGS if row[t] == '1'])
    st.caption('Unselected traits become negative labels when saved. Skip this sample if you cannot judge all traits. Traits can overlap.')
    if st.form_submit_button('Save reviewed labels'):
        if not pack.strip():
            st.error('Enter the original pack name.')
        else:
            # Re-read before saving to preserve edits to other samples. One labeler per manifest.
            current = read_rows(manifest)
            if len(current) != len(rows) or current[index - 1]['path'] != row['path']:
                st.error('Manifest changed. Reload before saving.')
            else:
                current[index - 1].update(pack=pack.strip(), reviewed='1', **{t: str(int(t in selected)) for t in TAGS})
                write_rows(manifest, current)
                st.success('Saved. Choose the next sample number above.')
