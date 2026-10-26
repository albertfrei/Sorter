"""Local, multi-label audio baseline. No filenames are used as training features."""
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
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TAGS = ['soft', 'punchy', 'bright', 'dark', 'warm', 'acoustic', 'electronic', 'clean', 'distorted', 'noisy']
EXTENSIONS = {'.wav', '.flac', '.aif', '.aiff', '.ogg', '.mp3', '.m4a'}
FIELDS = ['path', 'pack', 'reviewed', *TAGS]
VERSION = 1


def read_rows(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if not set(FIELDS).issubset(reader.fieldnames or []):
            raise ValueError('Manifest must contain: ' + ', '.join(FIELDS))
        return list(reader)


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
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
        raise ValueError('Label at least 30 samples before training; hundreds per trait are preferable')
    if any(not r['pack'].strip() or any(r[t] not in ('0', '1') for t in TAGS) for r in rows):
        raise ValueError('Every reviewed row needs a pack and explicit 0/1 values for all tags')
    if len({r['pack'] for r in rows}) < 5:
        raise ValueError('Use at least five distinct packs for train/validation/test separation')
    # Identical files cannot appear twice, including renamed copies in other packs.
    hashes = [fingerprint(r['path']) for r in rows]
    if len(set(hashes)) != len(hashes):
        raise ValueError('Duplicate audio files detected. Remove duplicate manifest rows before training')
    X = np.array([features(r['path']) for r in rows])
    Y = np.array([[int(r[t]) for t in tags] for r in rows])
    groups = np.array([r['pack'] for r in rows])
    trainval, test = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42).split(X, Y, groups))
    train_local, val_local = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=43).split(X[trainval], Y[trainval], groups[trainval]))
    training, validation = trainval[train_local], trainval[val_local]
    models, thresholds, metrics = {}, {}, {}
    for index, tag in enumerate(tags):
        if any(len(np.unique(Y[split, index])) < 2 for split in (training, validation, test)):
            raise ValueError(f'{tag}: need positive AND negative labels in each pack-held-out split. Add diverse labeled packs; do not trust an untestable tag')
        model = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42))
        model.fit(X[training], Y[training, index])
        val_scores = model.predict_proba(X[validation])[:, 1]
        candidates = np.arange(.2, .85, .05)
        f1s = [precision_recall_fscore_support(Y[validation, index], val_scores >= c, average='binary', zero_division=0)[2] for c in candidates]
        threshold = float(candidates[int(np.argmax(f1s))])
        scores = model.predict_proba(X[test])[:, 1]
        precision, recall, f1, _ = precision_recall_fscore_support(Y[test, index], scores >= threshold, average='binary', zero_division=0)
        metrics[tag] = dict(precision=float(precision), recall=float(recall), f1=float(f1), average_precision=float(average_precision_score(Y[test, index], scores)), threshold=threshold, test_positives=int(Y[test, index].sum()))
        models[tag], thresholds[tag] = model, threshold
    report = dict(feature_version=VERSION, tags=tags, samples=len(rows), splits={name: dict(samples=len(indices), packs=sorted(set(groups[indices]))) for name, indices in [('train', training), ('validation', validation), ('test', test)]}, metrics=metrics,
                  warning='Baseline scores are not calibrated confidence. Near duplicates and pack similarity can still inflate evaluation. Validate on your own unseen libraries.')
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(feature_version=VERSION, models=models, thresholds=thresholds), model_path)
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
            results.append(dict(path=str(path), tags=tags, scores=scores, needs_review=True))
        except (RuntimeError, ValueError, OSError) as ex:
            results.append(dict(path=str(path), error=str(ex), needs_review=True))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(results, indent=2), encoding='utf-8')
    return len(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    init = commands.add_parser('init'); init.add_argument('library'); init.add_argument('manifest')
    fit = commands.add_parser('train'); fit.add_argument('manifest'); fit.add_argument('--model', default='artifacts/traits.joblib'); fit.add_argument('--report', default='artifacts/report.json')
    fit.add_argument('--tags', nargs='+', choices=TAGS, default=TAGS, help='Train only these traits; default: all')
    infer = commands.add_parser('predict'); infer.add_argument('model'); infer.add_argument('library'); infer.add_argument('--output', default='predictions.json')
    args = parser.parse_args()
    if args.command == 'init': print(f'Created {init_manifest(args.library, args.manifest)} unlabeled rows')
    elif args.command == 'train': print(json.dumps(train(args.manifest, args.model, args.report, args.tags), indent=2))
    else: print(f'Analyzed {predict(args.model, args.library, args.output)} files')


if __name__ == '__main__':
    main()
