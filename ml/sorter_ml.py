"""Local instrument and multi-label timbre audio baseline. No filenames are used as training features."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.fft import dct
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TAGS = ['soft', 'punchy', 'bright', 'dark', 'warm', 'acoustic', 'electronic', 'clean', 'distorted', 'noisy']
INSTRUMENTS = {
    'kick': 'Kick', 'snare': 'Snare', 'clap': 'Clap', 'closed_hat': 'Closed hi-hat',
    'open_hat': 'Open hi-hat', 'cymbal': 'Cymbal / crash / ride', 'tom': 'Tom',
    'percussion': 'Other percussion', 'bass': 'Bass / sustained 808', 'piano': 'Piano / keys',
    'guitar': 'Guitar', 'strings': 'Strings', 'brass': 'Brass', 'woodwind': 'Woodwind',
    'synth': 'Synth', 'vocal': 'Vocal', 'fx': 'Sound effect',
    'mixed': 'Mixed / multiple instruments', 'other': 'Other known sound'
}
EXTENSIONS = {'.wav', '.flac', '.aif', '.aiff', '.ogg', '.mp3', '.m4a'}
LEGACY_FIELDS = ['path', 'pack', 'reviewed', *TAGS]
FIELDS = [*LEGACY_FIELDS, 'instrument', 'traits_reviewed']
VERSION = 1


def read_rows(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if not set(LEGACY_FIELDS).issubset(reader.fieldnames or []):
            raise ValueError('Manifest must contain: ' + ', '.join(LEGACY_FIELDS))
        rows = list(reader)
        for row in rows:
            # Old reviewed rows contain explicit trait labels, but no instrument annotation.
            row.setdefault('instrument', '')
            row.setdefault('traits_reviewed', row['reviewed'])
        return rows


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row, instrument=row.get('instrument', ''),
                                 traits_reviewed=row.get('traits_reviewed', row.get('reviewed', '0'))))
    temporary.replace(path)


def init_manifest(root, output):
    root, output = Path(root).resolve(), Path(output)
    if not root.is_dir():
        raise ValueError('Library folder does not exist')
    if output.exists():
        raise ValueError('Manifest exists; choose a new filename to preserve labels')
    rows = []
    for p in sorted(root.rglob('*')):
        if p.is_file() and not p.is_symlink() and p.suffix.lower() in EXTENSIONS:
            relative = p.relative_to(root)
            rows.append(dict(path=str(p), pack=relative.parts[0] if len(relative.parts) > 1 else 'library-root',
                             reviewed='0', **{t: '' for t in TAGS}))
    write_rows(output, rows)
    return len(rows)


def features(path):
    # Bounded read: first 12 seconds, preserving original dynamics (no peak normalization).
    with sf.SoundFile(path) as audio:
        if audio.samplerate <= 0 or audio.channels > 32:
            raise ValueError('Unsupported audio format')
        x = audio.read(min(len(audio), audio.samplerate * 12), dtype='float32', always_2d=True).mean(axis=1)
        rate = audio.samplerate
    if not len(x) or not np.isfinite(x).all():
        raise ValueError('Empty or non-finite audio')
    import math
    divisor = math.gcd(rate, 22050)
    x = signal.resample_poly(x, 22050 // divisor, rate // divisor)
    duration = len(x) / 22050
    if len(x) < 1024:
        x = np.pad(x, (0, 1024 - len(x)))
    _, _, spectrum = signal.stft(x, fs=22050, nperseg=1024, noverlap=512, boundary='zeros')
    power = np.abs(spectrum) ** 2
    energy = power.sum(axis=0) + 1e-12
    weights = power / energy
    frequencies = np.linspace(0, 11025, power.shape[0])[:, None]
    centroid = (weights * frequencies).sum(axis=0)
    spread = np.sqrt((weights * (frequencies - centroid) ** 2).sum(axis=0))
    flatness = np.exp(np.log(power + 1e-12).mean(axis=0)) / (power.mean(axis=0) + 1e-12)
    rolloff = (np.cumsum(weights, axis=0) >= .85).argmax(axis=0) * 22050 / 1024
    bands = [power[(frequencies[:, 0] >= low) & (frequencies[:, 0] < high)].sum(axis=0) / energy
             for low, high in [(0, 150), (150, 500), (500, 2000), (2000, 6000), (6000, 11026)]]
    # Log spectral cepstra: compact timbral descriptors (not a pretrained embedding).
    cepstra = dct(np.log(power + 1e-10), type=2, axis=0, norm='ortho')[1:21]
    rms = float(np.sqrt(np.mean(x ** 2)))
    peak = float(np.max(np.abs(x)))
    summary = [duration, rms, peak, peak / (rms + 1e-8), float(np.mean(np.abs(x) >= .999)),
               float(np.mean(np.diff(np.signbit(x)) != 0)), float(np.argmax(np.abs(x)) / len(x))]
    descriptors = np.vstack([centroid, spread, flatness, rolloff, *bands, cepstra])
    result = np.concatenate([summary, descriptors.mean(axis=1), descriptors.std(axis=1)])
    if not np.isfinite(result).all():
        raise ValueError('Feature extraction produced non-finite values')
    return result


def fingerprint(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def train(manifest, model_path, report_path, tags=None):
    tags = TAGS if tags is None else list(dict.fromkeys(tags))
    if not tags or not set(tags).issubset(TAGS):
        raise ValueError('Choose at least one supported trait')
    rows = [r for r in read_rows(manifest) if r['reviewed'] == '1']
    if len(rows) < 30:
        raise ValueError('Label at least 30 samples before training; hundreds per class are preferable')
    if any(not r['pack'].strip() or (r['instrument'] and r['instrument'] not in INSTRUMENTS) or
           (r['traits_reviewed'] == '1' and any(r[t] not in ('0', '1') for t in TAGS)) for r in rows):
        raise ValueError('Every reviewed row needs a pack, valid instrument if supplied, and explicit 0/1 values for reviewed traits')
    if len({r['pack'] for r in rows}) < 5:
        raise ValueError('Use at least five distinct packs for train/validation/test separation')
    hashes = [fingerprint(r['path']) for r in rows]
    if len(set(hashes)) != len(hashes):
        raise ValueError('Duplicate audio files detected. Remove duplicate manifest rows before training')
    X = np.array([features(r['path']) for r in rows])
    groups = np.array([r['pack'] for r in rows])
    trainval, test = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42).split(X, groups=groups))
    train_local, val_local = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=43).split(X[trainval], groups=groups[trainval]))
    training, validation = trainval[train_local], trainval[val_local]
    splits = [('train', training), ('validation', validation), ('test', test)]
    instrument_labels = np.array([r['instrument'] for r in rows])
    labeled_instruments = np.flatnonzero(instrument_labels != '')
    instrument_model = None
    instrument_report = dict(status='not_trained', reason='No instrument labels yet; existing trait-only data was preserved')
    if len(labeled_instruments):
        if len(labeled_instruments) < 30:
            raise ValueError('Label at least 30 instruments before training the combined model')
        classes = sorted(set(instrument_labels[labeled_instruments]))
        if len(classes) < 2:
            raise ValueError('Label at least two different instrument classes')
        instrument_splits = {name: indices[instrument_labels[indices] != ''] for name, indices in splits}
        for name, indices in instrument_splits.items():
            missing = set(classes) - set(instrument_labels[indices])
            if missing:
                raise ValueError(f'Instrument classes missing from {name} packs: {", ".join(sorted(missing))}. Label each class across more original packs')
        instrument_model = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42))
        indices = instrument_splits['train']
        instrument_model.fit(X[indices], instrument_labels[indices])
        def instrument_metrics(indices):
            predicted = instrument_model.predict(X[indices])
            return dict(samples=len(indices), metrics=classification_report(instrument_labels[indices], predicted, labels=classes, output_dict=True, zero_division=0),
                        confusion_matrix=confusion_matrix(instrument_labels[indices], predicted, labels=classes).tolist())
        instrument_report = dict(status='trained', classes=classes, validation=instrument_metrics(instrument_splits['validation']), test=instrument_metrics(instrument_splits['test']))
    models, thresholds, metrics, skipped_traits = {}, {}, {}, {}
    trait_reviewed = np.array([r['traits_reviewed'] == '1' for r in rows])
    for tag in tags:
        trait_splits = [indices[trait_reviewed[indices]] for _, indices in splits]
        labels = np.array([int(r[tag]) if r['traits_reviewed'] == '1' else -1 for r in rows])
        if sum(trait_reviewed) < 30 or any(len(np.unique(labels[indices])) < 2 for indices in trait_splits):
            reason = 'Need at least 30 explicitly trait-labeled samples and positive AND negative labels in each pack-held-out split'
            if instrument_model is None:
                raise ValueError(f'{tag}: need positive AND negative labels in each pack-held-out split and at least 30 trait labels')
            skipped_traits[tag] = reason
            continue
        train_indices, validation_indices, test_indices = trait_splits
        model = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42))
        model.fit(X[train_indices], labels[train_indices])
        val_scores = model.predict_proba(X[validation_indices])[:, 1]
        candidates = np.arange(.2, .85, .05)
        f1s = [precision_recall_fscore_support(labels[validation_indices], val_scores >= c, average='binary', zero_division=0)[2] for c in candidates]
        threshold = float(candidates[int(np.argmax(f1s))])
        scores = model.predict_proba(X[test_indices])[:, 1]
        precision, recall, f1, _ = precision_recall_fscore_support(labels[test_indices], scores >= threshold, average='binary', zero_division=0)
        metrics[tag] = dict(precision=float(precision), recall=float(recall), f1=float(f1), average_precision=float(average_precision_score(labels[test_indices], scores)), threshold=threshold, test_positives=int(labels[test_indices].sum()), test_samples=len(test_indices))
        models[tag], thresholds[tag] = model, threshold
    report = dict(feature_version=VERSION, tags=list(models), samples=len(rows),
                  splits={name: dict(samples=len(indices), packs=sorted(set(groups[indices]))) for name, indices in splits},
                  instrument=instrument_report, metrics=metrics, skipped_traits=skipped_traits,
                  warning='Baseline scores are not calibrated confidence. Near duplicates and pack similarity can still inflate evaluation. Validate on your own unseen libraries.')
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(feature_version=VERSION, models=models, thresholds=thresholds, instrument_model=instrument_model), model_path)
    Path(report_path).write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def predict(model_path, library, output):
    if not Path(library).is_dir():
        raise ValueError('Prediction library folder does not exist')
    # joblib uses pickle. Only load model files you trained or explicitly trust.
    model = joblib.load(model_path)
    if model['feature_version'] != VERSION:
        raise ValueError('Feature version mismatch; retrain the model')
    results = []
    for path in sorted(Path(library).resolve().rglob('*')):
        if not path.is_file() or path.is_symlink() or path.suffix.lower() not in EXTENSIONS:
            continue
        try:
            x = features(path).reshape(1, -1)
            scores = {tag: float(est.predict_proba(x)[0, 1]) for tag, est in model['models'].items()}
            tags = [tag for tag, score in scores.items() if score >= model['thresholds'][tag]]
            result = dict(path=str(path), tags=tags, scores=scores, needs_review=True)
            if model.get('instrument_model') is not None:
                estimator = model['instrument_model']
                instrument_scores = dict(zip(estimator.classes_, map(float, estimator.predict_proba(x)[0])))
                result.update(instrument=max(instrument_scores, key=instrument_scores.get), instrument_scores=instrument_scores)
            results.append(result)
        except (RuntimeError, ValueError, OSError) as ex:
            results.append(dict(path=str(path), error=str(ex), needs_review=True))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(results, indent=2), encoding='utf-8')
    return len(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    init = commands.add_parser('init'); init.add_argument('library'); init.add_argument('manifest')
    fit = commands.add_parser('train'); fit.add_argument('manifest'); fit.add_argument('--model', default='artifacts/sorter.joblib'); fit.add_argument('--report', default='artifacts/report.json')
    fit.add_argument('--tags', nargs='+', choices=TAGS, default=TAGS, help='Additional timbre traits to consider alongside instrument training; default: all')
    infer = commands.add_parser('predict'); infer.add_argument('model'); infer.add_argument('library'); infer.add_argument('--output', default='predictions.json')
    args = parser.parse_args()
    if args.command == 'init': print(f'Created {init_manifest(args.library, args.manifest)} unlabeled rows')
    elif args.command == 'train': print(json.dumps(train(args.manifest, args.model, args.report, args.tags), indent=2))
    else: print(f'Analyzed {predict(args.model, args.library, args.output)} files')


if __name__ == '__main__':
    main()
