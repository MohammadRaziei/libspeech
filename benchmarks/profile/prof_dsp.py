"""Python-level profile of the DSP pieces outside the main benchmark suite: FFT forward/inverse/dct/idct,
the free `libspeech.dct`, and `STFT.istft`, against numpy / scipy / librosa.

Everything here is Python-vs-Python (lists in, lists out, which is what these APIs take); the
"pure list<->float conversion" rows show how much of each time is conversion rather than DSP.
Run in an environment with libspeech, numpy, scipy and librosa installed:  python prof_dsp.py
"""

from __future__ import annotations

import numpy as np  # first: keep numpy's import cost out of every timed call

# isort: split
import gc
import time
from functools import partial

import librosa
import scipy.fft

import libspeech as ls


def best(fn, reps=7, warm=2):
    for _ in range(warm):
        fn()
    times = []
    for _ in range(reps):
        gc.collect()
        gc.disable()
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
        gc.enable()
    return min(times) * 1000


def row(name, mine, ref, ref_name):
    print(f"{name:44s}{mine:>11.3f}{ref:>11.3f}{mine / ref:>7.1f}x   {ref_name}")


def main():
    rng = np.random.default_rng(0)
    print(
        f"{'DSP outside the main suite (min ms)':44s}{'libspeech':>11s}{'reference':>11s}{'ratio':>8s}   reference"
    )
    for expo in (9, 12, 16):
        n = 1 << expo
        x = rng.standard_normal(n).astype(np.float32)
        xl = x.tolist()
        f = ls.FFT(expo)
        row(
            f"FFT.forward  N={n} (list in/out)",
            best(partial(f.forward, xl)),
            best(partial(np.fft.fft, x)),
            "numpy.fft.fft",
        )
        re, im = f.forward(xl)
        spectrum = np.asarray(re) + 1j * np.asarray(im)
        row(
            f"FFT.inverse  N={n} (list in/out)",
            best(partial(f.inverse, re, im)),
            best(partial(np.fft.ifft, spectrum)),
            "numpy.fft.ifft",
        )
        ref_dct = partial(scipy.fft.dct, x, type=2, norm="ortho")
        ref_idct = partial(scipy.fft.idct, x, type=2, norm="ortho")
        row(
            f"FFT.dct      N={n} (list in/out)",
            best(partial(f.dct, xl)),
            best(ref_dct),
            "scipy.fft.dct",
        )
        row(
            f"FFT.idct     N={n} (list in/out)",
            best(partial(f.idct, xl)),
            best(ref_idct),
            "scipy.fft.idct",
        )
        conv = best(lambda: np.asarray(xl, dtype=np.float32).tolist())  # noqa: B023
        print(f"{'   (pure list<->float conversion of N samples)':44s}{conv:>11.3f}")

    for n, k in ((26, 13), (512, 512), (4096, 4096), (1000, 1000)):
        x = rng.standard_normal(n).astype(np.float32)
        xl = x.tolist()
        ref = partial(scipy.fft.dct, x, type=2, norm="ortho")
        row(
            f"libspeech.dct  n={n} -> {k} outputs",
            best(partial(ls.dct, xl, k), reps=5),
            best(ref, reps=5),
            "scipy.fft.dct",
        )

    t = np.arange(160000) / 16000
    sig = (
        0.4 * np.sin(2 * np.pi * 220 * t) + 0.01 * rng.standard_normal(160000)
    ).astype(np.float32)
    st = ls.STFT(9, ls.WindowType.hann, 128)
    re, im = st.stft(sig)
    rel, iml = re.tolist(), im.tolist()
    ref_spec = (re[:, :257] + 1j * im[:, :257]).T
    ref_istft = partial(
        librosa.istft, ref_spec, hop_length=128, n_fft=512, window="hann", center=False
    )
    row(
        "STFT.istft 10 s (nested lists in, list out)",
        best(partial(st.istft, rel, iml), reps=5),
        best(ref_istft, reps=5),
        "librosa.istft",
    )
    print(
        f"{'   (nested-list conversion of 1247x512 x2 alone)':44s}{best(lambda: (re.tolist(), im.tolist()), reps=5):>11.3f}"
    )
    out = np.asarray(st.istft(rel, iml))
    print(
        "istft round trip max|err| (interior):",
        float(np.abs(out[1000:-1000] - sig[: len(out)][1000:-1000]).max()),
    )


if __name__ == "__main__":
    main()
