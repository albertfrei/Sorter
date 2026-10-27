"""Dataset and session operations shared by the local labeling UI."""
import hashlib
import os
from pathlib import Path
from sorter_ml import EXTENSIONS, INSTRUMENTS, TAGS, read_rows, write_rows


def data_directory():
    if os.environ.get('SORTER_DATA_DIR'):
        return Path(os.environ['SORTER_DATA_DIR']).expanduser()
    if os.environ.get('LOCALAPPDATA'):
        return Path(os.environ['LOCALAPPDATA']) / 'Sorter' / 'Training'
    return Path(__file__).resolve().parent / 'data'


def import_folder(folder, storage=None, pack_name='', manifest=None):
    if not str(folder).strip():
        raise ValueError('Choose a sample folder first.')
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise ValueError('Choose an existing sample folder.')
    identity = hashlib.sha256(os.path.normcase(str(root)).encode()).hexdigest()[:20]
    manifest = Path(manifest) if manifest else Path(storage or data_directory()) / f'library-{identity}.csv'
    rows = read_rows(manifest) if manifest.exists() else []
    known = {os.path.normcase(str(Path(r['path']).resolve())) for r in rows}
    added = 0
    for path in sorted(root.rglob('*')):
        if path.is_file() and not path.is_symlink() and path.suffix.lower() in EXTENSIONS:
            if os.path.normcase(str(path.resolve())) in known:
                continue
            relative = path.relative_to(root)
            pack = pack_name.strip() or (relative.parts[0] if len(relative.parts) > 1 else root.name)
            rows.append(dict(path=str(path), pack=pack, reviewed='0', **{tag: '' for tag in TAGS}))
            added += 1
    write_rows(manifest, rows)
    return manifest, added


def save_labels(manifest, path, pack, tags, instrument=None):
    if not pack.strip():
        raise ValueError('Enter the original sample pack name.')
    if tags is not None and not set(tags).issubset(TAGS):
        raise ValueError('Unknown trait.')
    if instrument is not None and instrument not in INSTRUMENTS:
        raise ValueError('Choose an instrument, or skip if you are unsure.')
    rows = read_rows(manifest)
    row = next((r for r in rows if r['path'] == path), None)
    if row is None:
        raise ValueError('Sample is no longer in this dataset. Reopen it before saving.')
    was_reviewed = row['reviewed'] == '1' and row['instrument'] in INSTRUMENTS
    row.update(pack=pack.strip(), reviewed='1')
    if instrument is not None:
        row['instrument'] = instrument
    if tags is not None:
        row.update(traits_reviewed='1', **{tag: str(int(tag in tags)) for tag in TAGS})
    write_rows(manifest, rows)
    return not was_reviewed


def choose_folder():
    # A local Windows desktop picker; browsing always happens on the machine running Python.
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    try:
        return filedialog.askdirectory(title='Choose your sample library', parent=root)
    finally:
        root.destroy()
