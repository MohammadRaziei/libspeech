"""Can a locally chosen gain recover a signal that was damped in places? (experiment, see README.md)

The clip is multiplied by m(t) = eps + sin^2(pi t / period), so the level dips towards eps at t = 0, period, 2 period...
The needed gain is therefore known (-20 log10 m(t)), unlike on a real recording. Gains are chosen per window and applied:

* "rms_3s_np_interp": the first version, 3 s window, 1 s hop, gain interpolated linearly in dB with numpy.
* "rms_overlap_resample": window `--win` s, hop 10 ms (heavy overlap), gain curve (dB) at 100 Hz brought to the audio
  rate with libspeech.Resample.
* "rms_vad_gain": level measured only over frames SileroVad calls speech (VAD run on the signal after the previous pass),
  target = median speech level of the clip, gain capped to [-20, +30] dB, windows without speech filled by interpolation,
  smoothed over 0.5 s, two passes.
* "fp_times_grms": gain = f(p) * g(1/rms) with p the VAD probability averaged over the window, f(p) = f_min + (1 - f_min) p
  (f_min = 0.5), g = target / rms capped to [-20, +30] dB. Two variants: p from the VAD run on the RMS-levelled signal
  ("levelled") or on the damped signal itself ("raw").
* "stream_fmin*": causal, frame by frame (32 ms), the streaming form of h(f(p) * g(1/rms)): g = target / EMA-rms
  (0.25 s), p = VAD probability of the frame after the previous gain (feedback), EMA 0.2 s, f = f_min + (1 - f_min) *
  smoothstep(p, 0.2, 0.5), h = dB-domain smoothing with attack 0.05 s (gain going down) and release 0.3 s (gain going up),
  gain ramped linearly in dB inside each frame. f_min in {1, 0.5, 0.25}.
* "entropy" (only with --entropy): per 3 s window, gain with the lowest SileroVad output entropy.

    python local_gain_recovery.py alex-noisy.mp3 out_dir [--eps 0.03] [--period 6] [--win 1.0] [--max-s 60] [--entropy]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from gain_by_confidence import SR, binary_entropy, probabilities, scale

import libspeech

CONTROL_HZ = 100
TARGET_DB = -20.0
GRID_DB = np.arange(-12.0, 36.0 + 1e-9, 4.0)


def rms_gain_db(x: np.ndarray, centres: np.ndarray, win_s: float) -> np.ndarray:
    """TARGET_DB minus the RMS level (dBFS) of the window of `win_s` s around each centre; the window is cut at the file edges."""
    cum = np.concatenate([[0.0], np.cumsum(x.astype(np.float64) ** 2)])
    a = np.clip(((centres - win_s / 2) * SR).astype(int), 0, len(x) - 1)
    b = np.clip(((centres + win_s / 2) * SR).astype(int), a + 1, len(x))
    rms = np.sqrt((cum[b] - cum[a]) / (b - a))
    return TARGET_DB - 20 * np.log10(rms + 1e-9)


def entropy_gain_db(
    x: np.ndarray, vad: libspeech.SileroVad, centres: np.ndarray, win_s: float
) -> np.ndarray:
    out = []
    for c in centres:
        seg = x[
            int(max(c - win_s / 2, 0) * SR) : int(min(c + win_s / 2, len(x) / SR) * SR)
        ]
        ent = [
            binary_entropy(probabilities(scale(seg, g), vad)).mean() for g in GRID_DB
        ]
        out.append(GRID_DB[int(np.argmin(ent))])
    return np.asarray(out)


def vad_mask(y: np.ndarray, vad: libspeech.SileroVad) -> np.ndarray:
    """Per-sample speech mask from the 32 ms (512 sample) VAD decisions."""
    flags = probabilities(y, vad) >= vad.threshold()
    return np.pad(np.repeat(flags, 512), (0, len(y)), mode="constant")[: len(y)]


def speech_level_db(
    x: np.ndarray,
    mask: np.ndarray,
    centres: np.ndarray,
    win_s: float,
    min_speech_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean-square level (dB) over the speech samples inside each window, and whether the window has enough speech."""
    m = mask.astype(np.float64)
    cum_e = np.concatenate([[0.0], np.cumsum(x.astype(np.float64) ** 2 * m)])
    cum_n = np.concatenate([[0.0], np.cumsum(m)])
    a = np.clip(((centres - win_s / 2) * SR).astype(int), 0, len(x) - 1)
    b = np.clip(((centres + win_s / 2) * SR).astype(int), a + 1, len(x))
    n, e = cum_n[b] - cum_n[a], cum_e[b] - cum_e[a]
    return 10 * np.log10(e / np.maximum(n, 1) + 1e-18), n >= min_speech_s * SR


def rms_vad_gain_db(
    x: np.ndarray,
    vad: libspeech.SileroVad,
    centres: np.ndarray,
    win_s: float,
    passes: int = 2,
    min_speech_s: float = 0.15,
    lo_db: float = -20.0,
    hi_db: float = 30.0,
    smooth_s: float = 0.5,
) -> np.ndarray:
    g = rms_gain_db(
        x, centres, win_s
    )  # first pass: plain overlapped RMS, only to give the VAD a roughly levelled signal
    for _ in range(passes):
        mask = vad_mask(apply_db(x, curve_resample(len(x), g)), vad)
        level, valid = speech_level_db(x, mask, centres, win_s, min_speech_s)
        if valid.sum() < 2:
            break
        wanted = np.clip(np.median(level[valid]) - level, lo_db, hi_db)
        g = np.interp(centres, centres[valid], wanted[valid])
        k = max(int(smooth_s * CONTROL_HZ), 1)
        g = np.convolve(
            np.pad(g, (k // 2, k - k // 2 - 1), mode="edge"),
            np.ones(k) / k,
            mode="valid",
        )
    return g


def window_mean(v: np.ndarray, centres: np.ndarray, win_s: float) -> np.ndarray:
    cum = np.concatenate([[0.0], np.cumsum(v.astype(np.float64))])
    a = np.clip(((centres - win_s / 2) * SR).astype(int), 0, len(v) - 1)
    b = np.clip(((centres + win_s / 2) * SR).astype(int), a + 1, len(v))
    return (cum[b] - cum[a]) / (b - a)


def fp_times_grms_db(
    x: np.ndarray,
    vad: libspeech.SileroVad,
    centres: np.ndarray,
    win_s: float,
    p_source: str,
    f_min: float = 0.5,
    lo_db: float = -20.0,
    hi_db: float = 30.0,
) -> np.ndarray:
    """gain = f(p) * g(1/rms), multiplied in the linear domain (added in dB)."""
    g_rms_db = rms_gain_db(x, centres, win_s)
    y = apply_db(x, curve_resample(len(x), g_rms_db)) if p_source == "levelled" else x
    p = np.pad(np.repeat(probabilities(y, vad), 512), (0, len(x)), mode="edge")[
        : len(x)
    ]
    f = f_min + (1 - f_min) * window_mean(p, centres, win_s)
    return np.clip(g_rms_db, lo_db, hi_db) + 20 * np.log10(f)


def smoothstep(v: float, e0: float, e1: float) -> float:
    t = min(max((v - e0) / (e1 - e0), 0.0), 1.0)
    return t * t * (3 - 2 * t)


def stream_fg_db(
    x: np.ndarray,
    vad: libspeech.SileroVad,
    f_min: float,
    target_db: float = TARGET_DB,
    lo_db: float = -20.0,
    hi_db: float = 30.0,
    tau_rms: float = 0.25,
    tau_p: float = 0.2,
    attack: float = 0.05,
    release: float = 0.3,
) -> np.ndarray:
    """Causal per-frame version; returns the applied gain in dB for every sample."""
    frame = 512
    dt = frame / SR
    a_rms, a_p, a_att, a_rel = (
        1 - np.exp(-dt / tau) for tau in (tau_rms, tau_p, attack, release)
    )
    vad.reset()
    gain_curve = np.zeros(len(x))
    power = None
    p_smooth, p_last, gain = 0.0, 0.0, 0.0
    for start in range(0, len(x), frame):
        fr = x[start : start + frame]
        ms = float(np.mean(fr.astype(np.float64) ** 2))
        power = ms if power is None else power + a_rms * (ms - power)
        g_db = float(np.clip(target_db - 10 * np.log10(power + 1e-12), lo_db, hi_db))
        if start == 0:
            gain = g_db
        vad.feed(apply_db(fr, np.full(len(fr), gain)).tolist())
        new = vad.pop_speech_probabilities()
        if len(new):
            p_last = float(new[-1])
        p_smooth += a_p * (p_last - p_smooth)
        f = f_min + (1 - f_min) * smoothstep(p_smooth, 0.2, 0.5)
        wanted = g_db + 20 * np.log10(f)
        new_gain = gain + (a_att if wanted < gain else a_rel) * (wanted - gain)
        gain_curve[start : start + len(fr)] = np.linspace(gain, new_gain, len(fr) + 1)[
            1:
        ]
        gain = new_gain
    return gain_curve


def curve_numpy(n: int, centres: np.ndarray, gains_db: np.ndarray) -> np.ndarray:
    return np.interp(np.arange(n) / SR, centres, gains_db)


def curve_resample(
    n: int, gains_db: np.ndarray, control_hz: int = CONTROL_HZ
) -> np.ndarray:
    """Brings a gain curve sampled at `control_hz` (first value at t = 0) to the audio rate with libspeech.Resample.
    The ends are padded with their edge values so the resampler's filter does not bend them."""
    pad = 20
    padded = np.concatenate(
        [np.full(pad, gains_db[0]), gains_db, np.full(pad, gains_db[-1])]
    ).astype(np.float32)
    up = np.asarray(
        libspeech.Resample(control_hz, SR).resample(padded), dtype=np.float64
    )
    start = pad * (SR // control_hz)
    up = up[start : start + n]
    return np.pad(up, (0, n - len(up)), mode="edge")


def apply_db(x: np.ndarray, gain_db: np.ndarray) -> np.ndarray:
    return np.clip(x * 10 ** (gain_db / 20), -1.0, 1.0).astype(np.float32)


def f1(pred: np.ndarray, ref: np.ndarray) -> float:
    tp = float(np.sum(pred & ref))
    return 2 * tp / max(2 * tp + np.sum(pred & ~ref) + np.sum(~pred & ref), 1)


def report(name, y, gain_curve_db, x_ref, m, vad, ref_speech, damped, ref_mask) -> None:
    """Gain error: std over all samples of (applied gain - needed gain); a constant offset does not count."""
    err = gain_curve_db + 20 * np.log10(m)
    a = float(np.dot(y, x_ref) / (np.dot(y, y) + 1e-12))
    snr = 10 * np.log10(np.sum(x_ref**2) / (np.sum((x_ref - a * y) ** 2) + 1e-12))
    sp = probabilities(y, vad) >= vad.threshold()
    n = min(len(sp), len(ref_speech))
    print(
        f"{name:<22} gain error std {np.std(err):5.1f} dB (speech samples {np.std(err[ref_mask]):5.1f}) | SNR vs original {snr:6.1f} dB | "
        f"VAD F1 all {f1(sp[:n], ref_speech[:n]):.3f}, damped parts "
        f"{f1(sp[:n] & damped[:n], ref_speech[:n] & damped[:n]):.3f}"
    )


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
    ap.add_argument("--entropy", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    x = librosa.load(args.src, sr=SR, mono=True)[0].astype(np.float32)
    if args.max_s:
        x = x[: int(args.max_s * SR)]
    t = np.arange(len(x)) / SR
    m = args.eps + np.sin(np.pi * t / args.period) ** 2
    damped_x = (x * m).astype(np.float32)
    vad = libspeech.SileroVad(sample_rate=SR)
    ref_p = probabilities(x, vad)
    ref_speech = ref_p >= vad.threshold()
    ref_mask = vad_mask(x, vad)
    damped = np.interp(np.arange(len(ref_p)) * 512 / SR, t, m) < 0.15
    print(
        f"{args.src.name}: {len(x) / SR:.1f} s, eps={args.eps} ({20 * np.log10(args.eps):.0f} dB), period {args.period} s, "
        f"window {args.win} s, reference speech {ref_speech.mean():.2f}, damped-window share {damped.mean():.2f}"
    )

    runs = {"damped, no correction": np.zeros(len(x))}
    c1 = np.arange(1.5, len(x) / SR, 1.0)
    runs["rms_3s_np_interp"] = curve_numpy(len(x), c1, rms_gain_db(damped_x, c1, 3.0))
    c2 = np.arange(0, len(x) / SR, 1 / CONTROL_HZ)
    g2 = rms_gain_db(damped_x, c2, args.win)
    runs["rms_overlap_resample"] = curve_resample(len(x), g2)
    runs["rms_vad_gain"] = curve_resample(
        len(x), rms_vad_gain_db(damped_x, vad, c2, args.win)
    )
    for src_name in ("levelled", "raw"):
        runs[f"fp_times_grms_{src_name}"] = curve_resample(
            len(x), fp_times_grms_db(damped_x, vad, c2, args.win, src_name)
        )
    for f_min in (1.0, 0.5, 0.25):
        runs[f"stream_fmin{f_min}"] = stream_fg_db(damped_x, vad, f_min)
    interp_check = curve_numpy(len(x), c2, g2)
    inner = slice(SR, -SR)
    print(
        f"resample vs np.interp of the same curve, inner part: max difference "
        f"{np.max(np.abs(runs['rms_overlap_resample'] - interp_check)[inner]):.2f} dB"
    )
    if args.entropy:
        c3 = np.arange(1.5, len(x) / SR, 1.0)
        runs["entropy"] = curve_numpy(
            len(x), c3, entropy_gain_db(damped_x, vad, c3, 3.0)
        )
    for name, g in runs.items():
        y = apply_db(damped_x, g)
        report(name, y, g, x, m, vad, ref_speech, damped, ref_mask)
        sf.write(args.out / f"{name.replace(', ', '_').replace(' ', '_')}.wav", y, SR)
    sf.write(args.out / "original.wav", x, SR)
    sf.write(args.out / "damped.wav", damped_x, SR)


if __name__ == "__main__":
    main()
