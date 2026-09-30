"""
corpus.py: deterministic synthetic audio corpus.

Speech-like signal (harmonic source with a gliding f0, a ~4 Hz syllabic
amplitude envelope and a little noise), written as 16-bit PCM mono WAV so
every library reads exactly the same bytes. Synthetic on purpose: it is
reproducible offline and its content does not change anything these DSP
kernels do (none of them are data-dependent).

Sample rates:
  16 kHz   -> stft / mfcc entries (what speech models actually consume)
  44.1 kHz -> resample entries (44.1 kHz -> 16 kHz)

Usage: corpus.py OUT_DIR   (writes OUT_DIR/*.wav and OUT_DIR/manifest.json)
"""
import argparse
import json
import os
import wave

import numpy as np

CORPUS = [
    ("speech16k_1s", 16000, 1),
    ("speech16k_10s", 16000, 10),
    ("speech16k_60s", 16000, 60),
    ("speech44k_10s", 44100, 10),
    ("speech44k_60s", 44100, 60),
]


def speech_like(duration_s, sample_rate, seed=0):
    """Returns float32 samples in [-0.5, 0.5]."""
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * sample_rate))
    t = np.arange(n, dtype=np.float64) / sample_rate
    f0 = 120.0 + 40.0 * np.sin(2 * np.pi * 0.3 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sample_rate
    sig = np.zeros(n)
    for k in range(1, 21):
        sig += np.sin(k * phase) / k
    sig *= 0.5 * (1.0 + np.sin(2 * np.pi * 4.0 * t))
    sig += 0.02 * rng.standard_normal(n)
    sig *= 0.5 / max(np.max(np.abs(sig)), 1e-9)
    return sig.astype(np.float32)


def write_wav(path, samples, sample_rate):
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out_dir")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    entries = []
    for name, sr, dur in CORPUS:
        path = os.path.join(args.out_dir, f"{name}.wav")
        x = speech_like(dur, sr)
        write_wav(path, x, sr)
        entries.append({
            "name": name,
            "path": os.path.abspath(path),
            "sample_rate": sr,
            "duration_s": dur,
            "samples": int(len(x)),
        })
        print(f"corpus: {name}: {len(x)} samples @ {sr} Hz", flush=True)

    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"entries": entries}, f, indent=2)


if __name__ == "__main__":
    main()
