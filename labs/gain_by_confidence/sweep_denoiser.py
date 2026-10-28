"""Control for gain_by_confidence.py: run a denoiser at fixed input gains (gain undone afterwards) to see whether
the entropy-selected gain does anything a plain 'quieter input' would not.

    python sweep_denoiser.py alex-noisy.mp3 out_dir gtcrn -4.6 -12 -20 -28.4 -36
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from gain_by_confidence import SR, scale

import libspeech


def rms(y: np.ndarray) -> float:
    return 20 * np.log10(np.sqrt(np.mean(y**2)) + 1e-12)


def main() -> None:
    import librosa
    import soundfile as sf

    src, out, backend = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    gains = [float(g) for g in sys.argv[4:]]
    out.mkdir(parents=True, exist_ok=True)
    x = librosa.load(src, sr=SR, mono=True)[0].astype(np.float32)
    with libspeech.Denoiser(backend) as denoiser:
        plain = np.asarray(denoiser.process(x), dtype=np.float32)
        print(f"{'gain dB':>8} {'out RMS dBFS':>13} {'vs plain, dB':>13}")
        print(f"{'plain':>8} {rms(plain):13.1f} {0.0:13.1f}")
        for g in gains:
            y = np.asarray(denoiser.process(scale(x, g)), dtype=np.float32) * 10 ** (
                -g / 20
            )
            sf.write(out / f"{backend}_gain{g:+.1f}dB_undone.wav", y, SR)
            print(f"{g:8.1f} {rms(y):13.1f} {rms(y) - rms(plain):13.1f}")


if __name__ == "__main__":
    main()
