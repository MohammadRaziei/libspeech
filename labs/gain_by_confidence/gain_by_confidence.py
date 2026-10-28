"""Pick the input gain of SileroVad from its own output confidence (experiment, see README.md).

    python gain_by_confidence.py alex-noisy.mp3 out_dir [denoiser_backend]

Writes the input, the gain-selected signal and (optionally) denoised versions without and with the gain, so they can be
listened to side by side. Nothing here is part of the libspeech API.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import libspeech

SR = 16000


def binary_entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def probabilities(x: np.ndarray, vad: libspeech.SileroVad) -> np.ndarray:
    vad.reset()
    vad.process(np.ascontiguousarray(x, dtype=np.float32).tolist())
    return np.asarray(vad.speech_probabilities(), dtype=np.float64)


def scale(x: np.ndarray, gain_db: float) -> np.ndarray:
    """Gain as a recording chain would apply it: the result is clipped to full scale."""
    return np.clip(x * 10 ** (gain_db / 20), -1.0, 1.0).astype(np.float32)


def objective(
    x: np.ndarray, gain_db: float, vad: libspeech.SileroVad
) -> dict[str, float]:
    p = probabilities(scale(x, gain_db), vad)
    return {
        "gain_db": gain_db,
        "entropy": float(binary_entropy(p).mean()),
        "undecided": float(((p > 0.2) & (p < 0.8)).mean()),
        "speech": float((p >= vad.threshold()).mean()),
    }


def select_gain_db(
    x: np.ndarray,
    vad: libspeech.SileroVad,
    lo: float = -30.0,
    hi: float = 30.0,
    step: float = 6.0,
) -> tuple[float, list[dict[str, float]]]:
    """Coarse grid over [lo, hi], then a golden-section refinement between the neighbours of the best grid point."""
    curve = [objective(x, g, vad) for g in np.arange(lo, hi + 1e-9, step)]
    best = min(range(len(curve)), key=lambda i: curve[i]["entropy"])
    a, b = (
        curve[max(best - 1, 0)]["gain_db"],
        curve[min(best + 1, len(curve) - 1)]["gain_db"],
    )
    phi = (5**0.5 - 1) / 2
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = objective(x, c, vad), objective(x, d, vad)
    for _ in range(5):
        if fc["entropy"] < fd["entropy"]:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = objective(x, c, vad)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = objective(x, d, vad)
    curve += [fc, fd]
    winner = min(curve, key=lambda r: r["entropy"])
    return winner["gain_db"], sorted(curve, key=lambda r: r["gain_db"])


def main() -> None:
    import librosa
    import soundfile as sf

    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    backend = sys.argv[3] if len(sys.argv) > 3 else None
    out.mkdir(parents=True, exist_ok=True)
    x, _ = librosa.load(src, sr=SR, mono=True)
    x = x.astype(np.float32)
    rms_db = 20 * np.log10(np.sqrt(np.mean(x**2)) + 1e-12)
    print(
        f"input: {len(x) / SR:.1f} s, RMS {rms_db:.1f} dBFS, peak {np.abs(x).max():.3f}"
    )

    vad = libspeech.SileroVad(sample_rate=SR)
    gain, curve = select_gain_db(x, vad)
    print(f"{'gain dB':>8} {'entropy':>8} {'undecided':>10} {'speech':>7}")
    for r in curve:
        mark = "  <- selected" if r["gain_db"] == gain else ""
        print(
            f"{r['gain_db']:8.1f} {r['entropy']:8.4f} {r['undecided']:10.3f} {r['speech']:7.3f}{mark}"
        )
    print(
        f"selected gain {gain:+.1f} dB (RMS-to-(-20 dBFS) would be {-20 - rms_db:+.1f} dB)"
    )

    sf.write(out / "1_alex_input.wav", x, SR)
    sf.write(out / "2_alex_gain_selected.wav", scale(x, gain), SR)
    if backend:
        with libspeech.Denoiser(backend) as denoiser:
            plain = np.asarray(denoiser.process(x), dtype=np.float32)
            boosted = np.asarray(denoiser.process(scale(x, gain)), dtype=np.float32)
        undo = boosted * 10 ** (
            -gain / 20
        )  # same output level as plain, so only the model's behaviour differs
        sf.write(out / f"3_alex_{backend}_plain.wav", plain, SR)
        sf.write(out / f"4_alex_{backend}_with_gain_then_undone.wav", undo, SR)
        n = min(len(plain), len(undo))
        diff = 20 * np.log10(
            np.linalg.norm(plain[:n]) / (np.linalg.norm(plain[:n] - undo[:n]) + 1e-12)
        )
        print(
            f"{backend}: plain vs with-gain outputs agree to {diff:.1f} dB SNR (higher = more alike)"
        )


if __name__ == "__main__":
    main()
