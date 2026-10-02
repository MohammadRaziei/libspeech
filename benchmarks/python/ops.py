"""
ops.py: the single definition of what each benchmarked operation means for
each library. bench_throughput.py, bench_scaling.py and the memory driver all
import this, so speed and memory always measure exactly the same work.

Fixed parameters (speech front-end defaults, 16 kHz input):
  stft      n_fft=512, hop=128, Hann window, no centering/padding
            (bins returned: libspeech numpy and librosa 257, audioflux 512)
  mfcc      13 coefficients, 26 mel filters, n_fft=512, hop=128, no centering
  resample  44.1 kHz -> 16 kHz
  load      16-bit PCM WAV file -> float32 samples in Python

Libraries:
  libspeech       numpy in / numpy out: float32 arrays are read in place and
                  the result is a NumPy array owning the C++ buffer (no list
                  conversion, no copy). This is the fast path a numpy user gets.
                  STFT is requested one-sided (257 bins, like librosa); audioflux
                  and the list API return all 512 bins.
  libspeech_list  the original list[float] in / nested-list out API, kept so
                  the cost of the list path stays visible. The input list is
                  prepared OUTSIDE the timer (it is that API's native input).
  librosa, audioflux, scipy, soundfile: numpy in / numpy out.

Every runner constructs its transform object (STFT/MFCC/Resample) INSIDE the
timed call, for every library, so setup cost is counted consistently.
Numerical output is NOT bit-identical across libraries (window/filterbank
conventions differ); the recorded output shape shows what each produced.
"""
import gc
import math
import time

import numpy as np

SAMPLE_RATE = 16000
N_FFT_EXP = 9
N_FFT = 1 << N_FFT_EXP
HOP = 128
N_MELS = 26
N_MFCC = 13
RESAMPLE_TARGET = 16000

OPS = ("load", "resample", "stft", "mfcc")

LIBS_FOR_OP = {
    "load": ("libspeech", "libspeech_list", "librosa", "soundfile"),
    "resample": ("libspeech", "libspeech_list", "librosa", "audioflux", "scipy"),
    "stft": ("libspeech", "libspeech_list", "librosa", "audioflux"),
    "mfcc": ("libspeech", "libspeech_list", "librosa", "audioflux"),
}

_IMPORT_NAME = {
    "libspeech": "libspeech",
    "libspeech_list": "libspeech",
    "librosa": "librosa",
    "audioflux": "audioflux",
    "scipy": "scipy",
    "soundfile": "soundfile",
}


def op_applies(op, sample_rate):
    """Which corpus entries an operation runs on."""
    if op == "load":
        return True
    if op == "resample":
        return sample_rate != RESAMPLE_TARGET
    return sample_rate == SAMPLE_RATE  # stft / mfcc


def library_available(lib):
    import importlib.util

    return importlib.util.find_spec(_IMPORT_NAME[lib]) is not None


def library_version(lib):
    from importlib import metadata

    dist = {"libspeech_list": "libspeech"}.get(lib, lib)
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def read_samples(path):
    import soundfile as sf

    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1).astype(np.float32)
    return np.ascontiguousarray(x), sr


def describe(res):
    """Shape summary of whatever a runner returned (ndarray / list / tuple)."""
    if isinstance(res, tuple):
        return [describe(r) for r in res]
    if hasattr(res, "shape"):
        return list(res.shape)
    if isinstance(res, list):
        if res and isinstance(res[0], list):
            return [len(res), len(res[0])]
        return [len(res)]
    return None


def make_runner(op, lib, x, sr, path):
    """Import the library and prepare inputs (untimed), return run()."""
    if lib == "libspeech":
        import libspeech as ls

        # numpy in / numpy out. Transform objects are constructed inside the
        # timed call, like every other library.
        xf = np.ascontiguousarray(x, dtype=np.float32)
        if op == "load":
            def run():
                a = ls.Audio()
                a.load(path)
                return a.to_numpy(0)
        elif op == "resample":
            def run():
                return ls.Resample(sr, RESAMPLE_TARGET).resample(xf)
        elif op == "stft":
            def run():
                return ls.STFT(N_FFT_EXP, ls.WindowType.hann, HOP).stft(xf, onesided=True)
        elif op == "mfcc":
            def run():
                p = ls.MFCCParams()
                p.sample_rate = sr
                p.num_mel_filters = N_MELS
                p.num_coefficients = N_MFCC
                p.radix2_exp = N_FFT_EXP
                p.slide_length = HOP
                return ls.MFCC(p).compute(xf)
        else:
            raise ValueError(op)
        return run

    if lib == "libspeech_list":
        import libspeech as ls

        xl = x.tolist()
        if op == "load":
            def run():
                a = ls.Audio()
                a.load(path)
                return a.data(0)
        elif op == "resample":
            def run():
                return ls.Resample(sr, RESAMPLE_TARGET).resample(xl)
        elif op == "stft":
            def run():
                return ls.STFT(N_FFT_EXP, ls.WindowType.hann, HOP).stft(xl)
        elif op == "mfcc":
            def run():
                p = ls.MFCCParams()
                p.sample_rate = sr
                p.num_mel_filters = N_MELS
                p.num_coefficients = N_MFCC
                p.radix2_exp = N_FFT_EXP
                p.slide_length = HOP
                return ls.MFCC(p).compute(xl)
        else:
            raise ValueError(op)
        return run

    if lib == "librosa":
        import librosa

        if op == "load":
            return lambda: librosa.load(path, sr=None, mono=True)[0]
        if op == "resample":
            return lambda: librosa.resample(x, orig_sr=sr, target_sr=RESAMPLE_TARGET)
        if op == "stft":
            return lambda: librosa.stft(x, n_fft=N_FFT, hop_length=HOP, window="hann", center=False)
        if op == "mfcc":
            return lambda: librosa.feature.mfcc(
                y=x, sr=sr, n_mfcc=N_MFCC, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP, center=False
            )

    if lib == "audioflux":
        import audioflux as af
        from audioflux.type import SpectralDataType, SpectralFilterBankScaleType, WindowType

        if op == "resample":
            def run():
                r = af.Resample()
                r.set_samplate(sr, RESAMPLE_TARGET)
                return r.resample(x)
            return run
        if op == "stft":
            return lambda: af.STFT(radix2_exp=N_FFT_EXP, window_type=WindowType.HANN, slide_length=HOP).stft(x)
        if op == "mfcc":
            def run():
                bft = af.BFT(
                    num=N_MELS, radix2_exp=N_FFT_EXP, samplate=sr,
                    scale_type=SpectralFilterBankScaleType.MEL,
                    data_type=SpectralDataType.POWER, slide_length=HOP,
                )
                spec = bft.bft(x)
                xx = af.XXCC(bft.num)
                xx.set_time_length(spec.shape[-1])
                return xx.xxcc(spec, cc_num=N_MFCC)
            return run

    if lib == "scipy" and op == "resample":
        from scipy import signal

        g = math.gcd(sr, RESAMPLE_TARGET)
        up, down = RESAMPLE_TARGET // g, sr // g
        return lambda: signal.resample_poly(x, up, down)

    if lib == "soundfile" and op == "load":
        import soundfile as sf

        return lambda: sf.read(path, dtype="float32")[0]

    raise ValueError(f"{lib} does not implement {op}")


def time_runner(run, repeats, warmup=1):
    """Best/median wall-clock seconds over `repeats` timed calls (GC paused)."""
    out = None
    for _ in range(warmup):
        out = run()
    times = []
    for _ in range(repeats):
        gc.collect()
        gc.disable()
        try:
            t0 = time.perf_counter()
            out = run()
            t1 = time.perf_counter()
        finally:
            gc.enable()
        times.append(t1 - t0)
    times.sort()
    return times[0], times[len(times) // 2], out
