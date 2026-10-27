"""Optional, offline CLAP audio embeddings. Imported only by CLAP training/prediction."""
import hashlib
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal


def checkpoint_fingerprint(directory):
    """Bind a classifier to the exact local CLAP weights and preprocessing config."""
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise ValueError('CLAP model must be an existing local model directory')
    files = sorted(p for p in directory.iterdir() if p.is_file() and
                   (p.suffix in {'.safetensors', '.bin'} or p.name.endswith('.json')))
    if not any(p.suffix in {'.safetensors', '.bin'} for p in files):
        raise ValueError('CLAP directory has no model weights (.safetensors or .bin)')
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode('utf-8') + b'\0')
        with path.open('rb') as source:
            for block in iter(lambda: source.read(1024 * 1024), b''):
                digest.update(block)
    return digest.hexdigest()


def read_audio(path):
    # Unfused CLAP uses 48 kHz mono, at most ten seconds. Bound the decode first.
    with sf.SoundFile(path) as audio:
        rate = audio.samplerate
        if rate <= 0 or audio.channels > 32:
            raise ValueError('Unsupported audio format')
        samples = audio.read(min(len(audio), rate * 10), dtype='float32', always_2d=True).mean(axis=1)
    if not len(samples) or not np.isfinite(samples).all():
        raise ValueError('Empty or non-finite audio')
    divisor = math.gcd(rate, 48000)
    return signal.resample_poly(samples, 48000 // divisor, rate // divisor)[:480000]


class ClapFeatures:
    def __init__(self, directory, expected_fingerprint=None):
        directory = str(Path(directory).resolve())
        fingerprint = checkpoint_fingerprint(directory)
        if expected_fingerprint is not None and fingerprint != expected_fingerprint:
            raise ValueError('CLAP checkpoint mismatch; use the original checkpoint or retrain')
        try:
            import torch
            from transformers import ClapAudioModelWithProjection, ClapFeatureExtractor
        except ImportError as error:
            raise ValueError('CLAP dependencies are missing; install ml/requirements-clap.txt') from error
        self.torch = torch
        self.processor = ClapFeatureExtractor.from_pretrained(directory, local_files_only=True)
        if self.processor.sampling_rate != 48000 or self.processor.truncation != 'rand_trunc':
            raise ValueError('Use an unfused 48 kHz CLAP checkpoint (laion/clap-htsat-unfused)')
        model, loading = ClapAudioModelWithProjection.from_pretrained(
            directory, local_files_only=True, output_loading_info=True)
        if loading.get('missing_keys') or loading.get('mismatched_keys') or loading.get('error_msgs'):
            raise ValueError('Incomplete or incompatible CLAP audio weights; use a complete unfused checkpoint')
        if model.config.enable_fusion:
            raise ValueError('Use an unfused CLAP checkpoint')
        self.model = model.to('cpu').eval()
        self.config = dict(backend='clap', version=1, model_path=directory, checkpoint_sha256=fingerprint)

    def __call__(self, path):
        inputs = self.processor(read_audio(path), sampling_rate=48000, return_tensors='pt')
        with self.torch.inference_mode():
            embedding = self.model(**inputs).audio_embeds[0].cpu().numpy()
        if embedding.ndim != 1 or not np.isfinite(embedding).all() or np.linalg.norm(embedding) == 0:
            raise ValueError('CLAP produced an invalid embedding')
        return embedding / np.linalg.norm(embedding)
