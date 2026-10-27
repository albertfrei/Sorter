"""Local guided labeler: streamlit run ml/label.py (optional: -- labels.csv)."""
import sys
from pathlib import Path
import streamlit as st
import soundfile as sf
from sorter_ml import INSTRUMENTS, TAGS, read_rows
from label_session import choose_folder, data_directory, import_folder, save_labels

st.set_page_config(page_title='Sorter · Sound lab', page_icon='🎧', layout='centered')
st.markdown('''<style>
.stApp {background:#101217;color:#e9edf5} h1,h2,h3 {letter-spacing:-.025em}
div[data-testid="stMetric"] {background:#1c2130;border:1px solid #30384b;border-radius:12px;padding:12px}
.stButton>button {border-radius:10px} .stProgress>div>div>div>div {background:#b4a1ff}
</style>''', unsafe_allow_html=True)
st.title('🎧 Sound lab')
st.caption('Listen. Find the character. Build a smarter library. All audio stays on this computer.')
state = st.session_state
state.setdefault('manifest', sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else '')
state.setdefault('session_saved', 0)
state.setdefault('skipped', [])
state.setdefault('current_path', '')
state.setdefault('previous_tags', None)
state.setdefault('previous_instrument', None)

with st.expander('📂 Choose or resume a library', expanded=not bool(state.manifest)):
    st.caption('Choose a folder on the computer running this app. Import references the files; it never moves or copies audio.')
    if st.button('Browse folders…', key='browse'):
        try:
            chosen = choose_folder()
            if chosen:
                state['folder'] = chosen
        except Exception:
            st.warning('The desktop picker is unavailable. Paste the folder path below instead.')
    folder = st.text_input('Sample folder', key='folder', placeholder='D:\\Samples')
    pack_override = st.text_input('Original pack name (optional)', help='Use this when importing a single pack. Otherwise top-level folders become pack names. Categories such as Snares are not pack names.')
    append = st.checkbox('Add this folder to the current dataset', disabled=not bool(state.manifest))
    if st.button('Import / resume folder', type='primary', key='import'):
        try:
            manifest, added = import_folder(folder, pack_name=pack_override, manifest=state.manifest if append else None)
            state.manifest = str(manifest)
            state.current_path = ''
            state.skipped = []
            state.session_saved = 0
            state.previous_tags = None
            state.previous_instrument = None
            state.pop('review_sample', None)
            st.success(f'{added} new samples imported. Existing labels preserved.')
        except (OSError, ValueError) as ex:
            st.error(str(ex))
    storage = data_directory()
    saved = sorted(storage.glob('library-*.csv')) if storage.exists() else []
    if saved:
        # Names come from the folder path recorded in the manifest rather than opaque IDs.
        def dataset_name(path):
            try:
                r = read_rows(path)
                return f'{Path(r[0]["path"]).parent.name if r else "Empty library"} · {len(r)} samples · {path.stem[-6:]}'
            except (OSError, ValueError):
                return path.name
        resume = st.selectbox('Saved datasets', saved, format_func=dataset_name)
        if st.button('Resume saved dataset', key='resume'):
            state.manifest = str(resume)
            state.current_path = ''
            state.skipped = []
            state.session_saved = 0
            state.previous_tags = None
            state.previous_instrument = None
    st.caption(f'Labels are saved automatically here: {storage}')

if not state.manifest:
    st.info('Start a round: import a folder of drum one-shots above.')
    st.stop()
manifest = Path(state.manifest).resolve()
try:
    rows = read_rows(manifest)
except (OSError, ValueError) as ex:
    st.error(f'Cannot open dataset: {ex}')
    st.stop()
if not rows:
    st.info('No supported audio found. Choose another folder or add audio and import again.')
    st.stop()
reviewed = sum(r['reviewed'] == '1' and r['instrument'] in INSTRUMENTS for r in rows)
goal = 20
col1, col2, col3 = st.columns(3)
col1.metric('Labeled', f'{reviewed} / {len(rows)}')
col2.metric('This session', state.session_saved)
col3.metric('Round goal', f'{min(state.session_saved, goal)} / {goal}')
st.progress(reviewed / len(rows), text=f'{reviewed / len(rows):.0%} of this library reviewed')
if state.session_saved >= goal:
    st.success('🏅 Round goal reached! Take a break or keep listening. Careful labels matter more than speed.')
st.caption('Instrument first, timbre optional. Progress counts instrument labels; older trait-only samples need an instrument pass. Skips do not count.')

with st.expander('Trait guide & previous labels'):
    st.markdown('''**Soft**: gentle, rounded onset. **Punchy**: strong, distinct attack.

**Bright**: prominent highs. **Dark**: subdued highs. **Warm**: full low-mid body, smooth highs.

**Acoustic**: physical instrument/recording character. **Electronic**: synthesized character.

**Clean**: little noise or saturation. **Distorted**: clipping, fuzz or saturation. **Noisy**: hiss, hum or background noise.

Traits overlap. Keep playback volume consistent: soft does not simply mean quiet. Skip if unsure.''')
    previous = st.selectbox('Jump to a sample', [r['path'] for r in rows], format_func=lambda p: Path(p).name, key='review_sample')
    if st.button('Review this sample', key='review'):
        state.current_path = previous

pending = [r for r in rows if (r['reviewed'] != '1' or r['instrument'] not in INSTRUMENTS) and r['path'] not in state.skipped]
row = next((r for r in rows if r['path'] == state.current_path), None)
if row is None:
    row = pending[0] if pending else None
if row is None:
    if reviewed == len(rows):
        st.success('🎉 Library complete! Your labels are saved and ready for training.')
    else:
        st.info(f'Round complete. {len(rows) - reviewed} samples remain unlabeled, including skipped files.')
    if state.skipped and st.button('Revisit skipped samples', key='revisit'):
        state.skipped = []
        st.rerun()
    st.caption(f'Training manifest: {manifest}')
    st.stop()
state.current_path = row['path']
# Per-file keys prevent previous selections carrying into a new sound.
identity = row['path']
st.subheader(Path(identity).name)
st.caption(f'Pack: {row["pack"]} · Instrument: {INSTRUMENTS.get(row["instrument"], "Not labeled yet")}')
readable = False
try:
    with sf.SoundFile(identity) as audio:
        if not len(audio) or audio.channels > 32:
            raise ValueError('Empty file or unsupported channel count.')
        clip = audio.read(min(len(audio), audio.samplerate * 12), dtype='float32', always_2d=True)
        st.audio(clip.T, sample_rate=audio.samplerate)
    st.caption('Play the sound, then describe what you hear. Preview uses the first 12 seconds.')
    readable = True
except (OSError, RuntimeError, ValueError) as ex:
    st.error(f'Cannot play this sample: {ex}. Skip it to continue; no label will be saved.')

if readable:
    if st.button('Use previous tags', key='use_previous', disabled=state.previous_instrument is None,
                 help='Reuse the instrument and optional timbre tags from your last saved sample; edit before saving.'):
        state[f'instrument-{identity}'] = state.previous_instrument
        state[f'tags-{identity}'] = (list(state.previous_tags) or ['__none__']) if state.previous_tags is not None else []
    if state.previous_instrument is not None:
        st.caption('Previous: ' + INSTRUMENTS[state.previous_instrument] +
                   (' · ' + (', '.join(state.previous_tags) or 'No timbre traits') if state.previous_tags is not None else ' · Instrument only'))
    with st.form(f'label-{identity}'):
        instrument = st.selectbox('Instrument', ['', *INSTRUMENTS],
                                  index=list(['', *INSTRUMENTS]).index(row['instrument']) if row['instrument'] in INSTRUMENTS else 0,
                                  format_func=lambda value: INSTRUMENTS.get(value, 'Choose an instrument…'), key=f'instrument-{identity}')
        st.caption('Use Kick for an 808 kick and Bass for a sustained bass note. Use Mixed for layered instruments; skip unidentified sounds instead of guessing.')
        selected = st.multiselect('Timbre', ['__none__', *TAGS],
                                  default=([t for t in TAGS if row[t] == '1'] or ['__none__']) if row['traits_reviewed'] == '1' else [],
                                  format_func=lambda value: 'None of these traits' if value == '__none__' else value.title(),
                                  key=f'tags-{identity}')
        st.caption('Choose multiple timbre traits if needed. Leave empty to label only the instrument, or choose None of these traits for explicit negatives. Existing traits are preserved when left empty.')
        pack = st.text_input('Original sample pack', value=row['pack'], key=f'pack-{identity}')
        submitted = st.form_submit_button('Save & next →', type='primary')
        if submitted:
            try:
                if '__none__' in selected and len(selected) > 1:
                    raise ValueError('Choose timbre traits OR None of these traits, not both.')
                traits = [tag for tag in selected if tag != '__none__'] if selected else None
                newly_reviewed = save_labels(manifest, identity, pack, traits, instrument=instrument)
                state.previous_instrument = instrument
                state.previous_tags = traits
                state.session_saved += int(newly_reviewed)
                state.current_path = ''
                st.rerun()
            except (OSError, ValueError) as ex:
                st.error(f'Labels were not saved: {ex}')
if st.button('Skip for now →', key='skip'):
    state.skipped = list(set([*state.skipped, identity]))
    state.current_path = ''
    st.rerun()
with st.expander('Dataset details'):
    st.code(str(manifest), language=None)
    st.caption('Re-import the same folder to discover new samples and preserve labels. One labeling app per dataset at a time. Back up this CSV; deleted or moved audio paths are retained for review.')
