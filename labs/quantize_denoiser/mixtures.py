"""Speech + noise mixtures for calibration and distillation.

With real recordings (`speech_dir` and `noise_dir`: folders of .wav/.flac/.ogg files) the mixtures look like what the
denoiser sees in use, and that is what quality should be judged on. Without them the module falls back to synthetic
speech-like signals and noise: enough to check that a pipeline runs, not enough to judge quality.
"""

from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SR = 16000
AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3"}
MAX_CACHED_FILES = 256


def list_audio(directory) -> list[Path]:
    files = sorted(p for p in Path(directory).rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS)
    if not files:
        message = f"no .wav/.flac/.ogg/.mp3 files under {directory}"
        raise FileNotFoundError(message)
    return files


def read_mono(path) -> np.ndarray:
    """Float32 mono samples at 16 kHz."""
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        g = gcd(SR, sr)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x


def crop(x: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """A random window of n samples; a shorter signal is repeated first."""
    if len(x) < n:
        x = np.tile(x, -(-n // max(len(x), 1)))
    start = int(rng.integers(0, len(x) - n + 1))
    return x[start : start + n]


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """clean + k * noise, with k chosen so that 10 log10(P_clean / P_scaled_noise) is exactly snr_db."""
    p_clean = float(np.mean(clean.astype(np.float64) ** 2))
    p_noise = float(np.mean(noise.astype(np.float64) ** 2))
    k = np.sqrt(p_clean / (p_noise * 10.0 ** (snr_db / 10.0)))
    return (clean + k * noise).astype(np.float32)


def synthetic_speech(n: int, rng: np.random.Generator) -> np.ndarray:
    """A voiced, speech-like signal: moving pitch, formant-shaped harmonics, syllable-rate on/off envelope."""
    t = np.arange(n) / SR
    f0 = rng.uniform(90, 260) * (1 + 0.15 * np.sin(2 * np.pi * rng.uniform(0.3, 1.2) * t + rng.uniform(0, 6.28)))
    phase = 2 * np.pi * np.cumsum(f0) / SR
    formants = rng.uniform([300, 900, 2200], [800, 2200, 3200])
    x = np.zeros(n)
    for k in range(1, 30):
        fk = f0 * k
        gain = sum(np.exp(-0.5 * ((fk - f) / (0.15 * f + 80)) ** 2) for f in formants) * (fk < 7500)
        x += gain * np.sin(k * phase) / np.sqrt(k)
    env = np.clip(np.sin(2 * np.pi * rng.uniform(2, 5) * t + rng.uniform(0, 6.28)) + 0.3, 0, 1) ** 0.5
    return (x * env).astype(np.float32)


def synthetic_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    kind = rng.choice(["white", "pink", "brown", "babble"])
    if kind == "babble":
        return sum(synthetic_speech(n, rng) * rng.uniform(0.3, 1.0) for _ in range(4)).astype(np.float32)
    w = rng.standard_normal(n)
    if kind == "white":
        return w.astype(np.float32)
    spec = np.fft.rfft(w)
    f = np.fft.rfftfreq(n, 1 / SR)
    f[0] = f[1]
    spec /= f ** (0.5 if kind == "pink" else 1.0)
    return np.fft.irfft(spec, n).astype(np.float32)


class MixtureSource:
    """Random (mixture, clean) pairs: a speech crop plus a noise crop at a random SNR and a random level."""

    def __init__(self, speech_dir=None, noise_dir=None, seconds=4.0, snr_range=(-5.0, 20.0), seed=0):
        self.rng = np.random.default_rng(seed)
        self.n = int(seconds * SR)
        self.snr_range = snr_range
        self.speech = list_audio(speech_dir) if speech_dir else None
        self.noise = list_audio(noise_dir) if noise_dir else None
        self._cache: dict[Path, np.ndarray] = {}

    def _read(self, path: Path) -> np.ndarray:
        if path not in self._cache:
            if len(self._cache) >= MAX_CACHED_FILES:
                self._cache.pop(next(iter(self._cache)))
            self._cache[path] = read_mono(path)
        return self._cache[path]

    def _signal(self, files, synthetic) -> np.ndarray:
        for _ in range(100):
            if files:
                x = crop(self._read(files[int(self.rng.integers(len(files)))]), self.n, self.rng)
            else:
                x = synthetic(self.n, self.rng)
            if np.mean(x.astype(np.float64) ** 2) > 1e-9:
                return x
        message = "could not draw a non-silent signal; check the audio files"
        raise RuntimeError(message)

    def sample(self) -> tuple[np.ndarray, np.ndarray]:
        clean = self._signal(self.speech, synthetic_speech)
        noise = self._signal(self.noise, synthetic_noise)
        mix = mix_at_snr(clean, noise, float(self.rng.uniform(*self.snr_range)))
        level_db = float(self.rng.uniform(-30.0, -3.0))
        gain = 10.0 ** (level_db / 20.0) / max(float(np.abs(mix).max()), 1e-9)
        return mix * gain, clean * gain

    def batch(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        pairs = [self.sample() for _ in range(n)]
        return np.stack([p[0] for p in pairs]), np.stack([p[1] for p in pairs])


class NoisyRecordings:
    """Crops of recordings that are noisy already (real-world captures).

    There is no clean reference for them, so `batch` returns None in its place: quality can only be judged against the
    fp32 model's own output. They are the most realistic calibration data there is, and need no speech/noise mixing.
    """

    def __init__(self, directory, seconds=4.0, seed=0):
        self.files = list_audio(directory)
        self.n = int(seconds * SR)
        self.rng = np.random.default_rng(seed)
        self._cache: dict[Path, np.ndarray] = {}

    def _read(self, path: Path) -> np.ndarray:
        if path not in self._cache:
            if len(self._cache) >= MAX_CACHED_FILES:
                self._cache.pop(next(iter(self._cache)))
            self._cache[path] = read_mono(path)
        return self._cache[path]

    def sample(self) -> np.ndarray:
        for _ in range(100):
            x = crop(self._read(self.files[int(self.rng.integers(len(self.files)))]), self.n, self.rng)
            if np.mean(x.astype(np.float64) ** 2) > 1e-9:
                return x
        message = "could not draw a non-silent crop; check the audio files"
        raise RuntimeError(message)

    def batch(self, n: int) -> tuple[np.ndarray, None]:
        return np.stack([self.sample() for _ in range(n)]), None
