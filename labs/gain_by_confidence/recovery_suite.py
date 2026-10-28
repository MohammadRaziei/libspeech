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
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from gain_by_confidence import SR, probabilities
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


REPEATS = 3


def timed(fn, repeats: int = REPEATS):
    """Runs fn `repeats` times; returns the last result and the median wall-clock seconds."""
    times, out = [], None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return out, float(np.median(times))


def level_stats(sig: np.ndarray) -> dict[str, float]:
    return {
        "rms_dbfs": round(
            20 * np.log10(np.sqrt(np.mean(sig.astype(np.float64) ** 2)) + 1e-12), 2
        ),
        "peak": round(float(np.max(np.abs(sig))), 4),
        "clipped_fraction": round(float(np.mean(np.abs(sig) >= 0.999)), 5),
    }


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
        t0 = time.perf_counter()
        ref = np.asarray(first.process(x), dtype=np.float32)
        ref_seconds = time.perf_counter() - t0
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
    duration = n / SR
    # warm-up so that the first timed call does not pay for session start-up
    gtcrn.process(x[:SR])
    probabilities(x[:SR], vad)

    methods = {  # name -> (function mix -> gain curve in dB, causal, lookahead s, frame ms)
        "gain_rms_overlap": (
            lambda mix: curve_resample(n, rms_gain_db(mix, centres, args.win)),
            False,
            args.win / 2,
            None,
        ),
        "gain_fp_levelled": (
            lambda mix: curve_resample(
                n, fp_times_grms_db(mix, vad, centres, args.win, "levelled")
            ),
            False,
            args.win / 2,
            None,
        ),
        "gain_stream_fmin1": (lambda mix: stream_fg_db(mix, vad, 1.0), True, 0.0, 32),
        "gain_stream_fmin0.5": (lambda mix: stream_fg_db(mix, vad, 0.5), True, 0.0, 32),
    }

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
    results = {"scenarios": {}}
    rows = []
    for key, (desc, raw_mix) in scenarios.items():
        mix = np.clip(raw_mix, -1, 1).astype(np.float32)
        # (name, signal, gain seconds, denoise seconds, causal, lookahead, frame ms)
        entries = [("mix_input", mix, 0.0, 0.0, None, None, None)]
        gained = {}
        for name, (fn, causal, look, frame) in methods.items():
            y, sec = timed(lambda fn=fn, mix=mix: apply_db(mix, fn(mix)))
            gained[name] = (y, sec, causal, look, frame)
            entries.append((name, y, sec, 0.0, causal, look, frame))
        den, sec = timed(
            lambda mix=mix: np.asarray(gtcrn.process(mix), dtype=np.float32)
        )
        entries.append(("gtcrn_only", den, 0.0, sec, None, None, None))
        for name, (y, gsec, causal, look, frame) in gained.items():
            den, sec = timed(lambda y=y: np.asarray(gtcrn.process(y), dtype=np.float32))
            entries.append((f"{name}_then_gtcrn", den, gsec, sec, causal, look, frame))

        index.append(f"| **{key}** | **{desc}** |")
        scores = {}
        records = []
        for i, (name, sig, gsec, dsec, causal, look, frame) in enumerate(entries):
            fname = f"{key}_{i:02d}_{name}.wav"
            sf.write(args.out / fname, sig, SR)
            scores[name] = si_sdr(sig, ref)
            index.append(f"| {fname} | SI-SDR vs reference {scores[name]:.1f} dB |")
            records.append(
                {
                    "index": i,
                    "file": fname,
                    "name": name,
                    "si_sdr_db": round(scores[name], 2),
                    **level_stats(sig),
                    "gain_seconds": round(gsec, 4),
                    "denoise_seconds": round(dsec, 4),
                    "total_seconds": round(gsec + dsec, 4),
                    "rtf_gain": round(gsec / duration, 5),
                    "rtf_denoise": round(dsec / duration, 5),
                    "rtf_total": round((gsec + dsec) / duration, 5),
                    "causal": causal,
                    "lookahead_seconds": look,
                    "frame_ms": frame,
                }
            )
        for r in records:
            r["si_sdr_vs_mix_db"] = round(r["si_sdr_db"] - scores["mix_input"], 2)
            r["si_sdr_vs_gtcrn_only_db"] = round(
                r["si_sdr_db"] - scores["gtcrn_only"], 2
            )
        results["scenarios"][key] = {"description": desc, "outputs": records}
        rows.append((key, desc, scores))

    summary = {}
    for name in [
        "mix_input",
        *methods,
        "gtcrn_only",
        *[f"{k}_then_gtcrn" for k in methods],
    ]:
        recs = [
            r
            for sc in results["scenarios"].values()
            for r in sc["outputs"]
            if r["name"] == name
        ]
        summary[name] = {
            "median_si_sdr_db": round(
                float(np.median([r["si_sdr_db"] for r in recs])), 2
            ),
            "median_rtf_gain": round(
                float(np.median([r["rtf_gain"] for r in recs])), 5
            ),
            "median_rtf_total": round(
                float(np.median([r["rtf_total"] for r in recs])), 5
            ),
        }
    results = {
        "meta": {
            "clip": args.src.name,
            "duration_seconds": round(duration, 3),
            "sample_rate": SR,
            "damping": f"m(t) = {args.eps} + sin^2(pi t / {args.period} s)",
            "window_seconds": args.win,
            "reference_denoiser": "facebook (DNS64)",
            "reference_seconds": round(ref_seconds, 3),
            "pipeline_denoiser": "gtcrn",
            "timing": f"median of {REPEATS} runs after a warm-up; rtf = seconds / audio seconds (below 1 is faster than real time)",
            "cpu_count": os.cpu_count(),
            "machine": platform.platform(),
            "python": platform.python_version(),
            "libspeech_version": getattr(libspeech, "__version__", None),
            "onnx_threads": 1,
            "note": "gain_stream_* are a per-frame Python simulation of a streaming implementation, so their time includes Python overhead; "
            "the reference is a denoiser output, not a studio recording",
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "reference": {
            "noisy_clip_si_sdr_db": round(si_sdr(x, ref), 2),
            "alex_noise_below_reference_db": round(
                20 * np.log10(rms(ref) / rms(noise_alex)), 2
            ),
        },
        "summary_over_scenarios": summary,
        **results,
    }
    (args.out / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(
        "\nSI-SDR against the reference, dB (higher is better); 'then_gtcrn' = the gain method followed by gtcrn"
    )
    names = list(rows[0][2])
    print(
        f"{'':<28}"
        + "".join(f"{k:>5}" for k, _, _ in rows)
        + "   median RTF (gain / total)"
    )
    for name in names:
        rtf = summary[name]
        print(
            f"{name:<28}"
            + "".join(f"{s[name]:5.1f}" for _, _, s in rows)
            + f"   {rtf['median_rtf_gain']:.4f} / {rtf['median_rtf_total']:.4f}"
        )
    (args.out / "INDEX.md").write_text("\n".join(index) + "\n")
    gtcrn.close()


if __name__ == "__main__":
    main()
