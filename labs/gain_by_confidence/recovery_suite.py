"""Recovery test with a (pseudo-)clean reference, so that noise suppression is measured too (see README.md).

1. reference = a first denoiser (DNS64 "facebook") applied to the noisy clip; extracted noise = clip - reference.
2. The test signals are built from the reference: speech damped by m(t) = eps + sin^2(pi t / period), plus noise.
3. Gain methods (from local_gain_recovery.py) are run on each test signal, alone and followed by a second denoiser (gtcrn).
4. Score: SI-SDR against the reference (higher is better), all files numbered, INDEX.md says which is which.

The reference is itself a denoiser output, not a studio recording: absolute numbers are optimistic and only comparisons
between rows of the same scenario mean something. Different denoisers are used for the reference and for the pipeline.

    python recovery_suite.py alex-noisy.mp3 out_dir [--eps 0.03] [--period 6] [--win 1.0] [--max-s 60]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from gain_by_confidence import SR
from local_gain_recovery import (
    apply_db,
    curve_resample,
    fp_times_grms_db,
    rms_gain_db,
    stream_fg_db,
)

import libspeech


def si_sdr(est: np.ndarray, ref: np.ndarray) -> float:
    n = min(len(est), len(ref))
    est, ref = est[:n].astype(np.float64), ref[:n].astype(np.float64)
    target = (np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)) * ref
    return float(
        10 * np.log10(np.sum(target**2) / (np.sum((est - target) ** 2) + 1e-12))
    )


def rms(v: np.ndarray) -> float:
    return float(np.sqrt(np.mean(v.astype(np.float64) ** 2)))


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    spec = np.fft.rfft(rng.standard_normal(n))
    spec[1:] /= np.sqrt(np.arange(1, len(spec)))
    spec[0] = 0
    return np.fft.irfft(spec, n)


def hum_plus_white(n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / SR
    hum = sum(
        a * np.sin(2 * np.pi * f * t)
        for f, a in ((50, 1.0), (100, 0.5), (150, 0.3), (200, 0.2))
    )
    return hum / rms(hum) + rng.standard_normal(n)


def at_snr(noise: np.ndarray, ref: np.ndarray, snr_db: float) -> np.ndarray:
    return noise * (rms(ref) / (rms(noise) + 1e-12)) * 10 ** (-snr_db / 20)


def main() -> None:
    import librosa
    import soundfile as sf

    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--eps", type=float, default=0.03)
    ap.add_argument("--period", type=float, default=6.0)
    ap.add_argument("--win", type=float, default=1.0)
    ap.add_argument("--max-s", type=float, default=0.0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    x = librosa.load(args.src, sr=SR, mono=True)[0].astype(np.float32)
    if args.max_s:
        x = x[: int(args.max_s * SR)]
    with libspeech.Denoiser("facebook") as first:
        ref = np.asarray(first.process(x), dtype=np.float32)
    n = min(len(x), len(ref))
    x, ref = x[:n], ref[:n]
    noise_alex = x - ref
    t = np.arange(n) / SR
    m = (args.eps + np.sin(np.pi * t / args.period) ** 2).astype(np.float32)
    rng = np.random.default_rng(0)
    print(
        f"reference = facebook(x); SI-SDR of the noisy clip against it: {si_sdr(x, ref):.1f} dB; "
        f"alex noise is {20 * np.log10(rms(ref) / rms(noise_alex)):.1f} dB below the reference"
    )

    scenarios = {
        "A": (
            "damped(reference + alex noise): everything damped",
            m * (ref + noise_alex),
        ),
        "B": (
            "damped(reference) + alex noise (noise not damped)",
            m * ref + noise_alex,
        ),
        "C": (
            "damped(reference) + pink noise at 10 dB SNR",
            m * ref + at_snr(pink_noise(n, rng), ref, 10),
        ),
        "D": (
            "damped(reference) + 50 Hz hum and white noise at 10 dB SNR",
            m * ref + at_snr(hum_plus_white(n, rng), ref, 10),
        ),
    }
    vad = libspeech.SileroVad(sample_rate=SR)
    centres = np.arange(0, n / SR, 0.01)
    gtcrn = libspeech.Denoiser("gtcrn")

    sf.write(args.out / "00_reference_clean.wav", ref, SR)
    sf.write(args.out / "01_noise_alex_extracted.wav", noise_alex, SR)
    sf.write(args.out / "02_alex_noisy_original.wav", x, SR)
    index = [
        "| file | what |",
        "|---|---|",
        "| 00_reference_clean.wav | the noisy clip passed through the first denoiser (DNS64): the reference |",
        "| 01_noise_alex_extracted.wav | noisy clip minus the reference: the extracted noise |",
        "| 02_alex_noisy_original.wav | the original noisy clip |",
    ]
    rows = []
    for key, (desc, raw_mix) in scenarios.items():
        mix = np.clip(raw_mix, -1, 1).astype(np.float32)
        curves = {
            "gain_rms_overlap": curve_resample(n, rms_gain_db(mix, centres, args.win)),
            "gain_fp_levelled": curve_resample(
                n, fp_times_grms_db(mix, vad, centres, args.win, "levelled")
            ),
            "gain_stream_fmin1": stream_fg_db(mix, vad, 1.0),
            "gain_stream_fmin0.5": stream_fg_db(mix, vad, 0.5),
        }
        files = [
            ("mix_input", mix),
            ("gtcrn_only", np.asarray(gtcrn.process(mix), dtype=np.float32)),
        ]
        gained = {name: apply_db(mix, c) for name, c in curves.items()}
        files[1:1] = list(gained.items())
        files += [
            (f"{name}_then_gtcrn", np.asarray(gtcrn.process(y), dtype=np.float32))
            for name, y in gained.items()
        ]
        index.append(f"| **{key}** | **{desc}** |")
        scores = {}
        for i, (name, sig) in enumerate(files):
            fname = f"{key}_{i:02d}_{name}.wav"
            sf.write(args.out / fname, sig, SR)
            scores[name] = si_sdr(sig, ref)
            index.append(f"| {fname} | SI-SDR vs reference {scores[name]:.1f} dB |")
        rows.append((key, desc, scores))

    print(
        "\nSI-SDR against the reference, dB (higher is better); 'then_gtcrn' = the gain method followed by gtcrn"
    )
    names = list(rows[0][2])
    print(f"{'':<28}" + "".join(f"{k:>5}" for k, _, _ in rows))
    for name in names:
        print(f"{name:<28}" + "".join(f"{s[name]:5.1f}" for _, _, s in rows))
    (args.out / "INDEX.md").write_text("\n".join(index) + "\n")
    gtcrn.close()


if __name__ == "__main__":
    main()
