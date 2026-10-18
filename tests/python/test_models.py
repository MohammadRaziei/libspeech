"""Tests for the libspeech.Denoiser / libspeech.SileroVad / libspeech.SpeechTimestamp
bindings. These don't exercise real ONNX inference (that needs downloaded
model weights, out of scope for a fast unit test) -- they validate the
binding surface itself: construction, error propagation, and correctness
of the plain-data SpeechTimestamp helper.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

import libspeech


def test_speech_timestamp_seconds_conversion():
    # Regression test: start_s()/end_s() used to do integer division
    # (start/sample_rate with both as C++ int), always truncating to 0.0.
    ts = libspeech.SpeechTimestamp(start=100, end=500, sample_rate=16000)
    assert ts.start_s == pytest.approx(100 / 16000, abs=1e-6)
    assert ts.end_s == pytest.approx(500 / 16000, abs=1e-6)


def test_speech_timestamp_repr_contains_values():
    ts = libspeech.SpeechTimestamp(start=100, end=500, sample_rate=16000)
    text = repr(ts)
    assert "00000100" in text
    assert "00000500" in text


def test_denoiser_create_rejects_unknown_backend():
    with pytest.raises(ValueError):
        libspeech.Denoiser.create("not_a_real_backend", "foo.onnx")


def test_silero_vad_rejects_missing_model():
    # Not a URL and not a name libspeech's default model release actually
    # has, so this should fail cleanly (a 404 -> RuntimeError), not crash.
    with pytest.raises(RuntimeError):
        libspeech.SileroVad(model_path="this_model_definitely_does_not_exist.onnx")


def test_denoiser_create_rejects_negative_num_threads():
    with pytest.raises(ValueError):
        libspeech.Denoiser.create("facebook", "foo.onnx", num_threads=-1)


# Heavy test: needs the real model, so it is marked `models` and skipped unless run with
# `pytest -m models` (cmake.yml). It is kept as small as possible while still covering what broke:
# two model loads and ONE real inference (the model needs seconds per call whatever the clip
# length), run in a subprocess so a crash is reported as a failure instead of killing pytest.
_DENOISER_SCRIPT = """
import gc
import numpy as np
import libspeech

MODEL = "facebook-denoiser-dns64.onnx"   # downloaded to ~/.libspeech on first use
x = (0.05 * np.random.default_rng(0).standard_normal(4000)).astype(np.float32)

d = libspeech.Denoiser("facebook")  # the backend's standard model, downloaded on first use
assert d.sample_rate == 16000 and not d.closed
y = d(x)  # the only real inference: NumPy in -> float32 NumPy out, same length (d(x) is d.process(x))
assert isinstance(y, np.ndarray) and y.dtype == np.float32 and y.shape == x.shape
try:
    d.process(np.zeros(0, np.float32))
    raise SystemExit("empty input did not raise")
except ValueError:
    pass

# Destroying a denoiser that is still open used to abort the process ("free(): invalid pointer").
del d
gc.collect()

# `with` closes it. Afterwards process() must raise for both input kinds (this also proves the
# list overload exists, without running the model), close() is idempotent, and the closed object
# is left alive on purpose so the interpreter has to shut down cleanly with it.
with libspeech.Denoiser.create("facebook", MODEL) as d2:
    pass
assert d2.closed
for bad in (x, x.tolist()):
    try:
        d2.process(bad)
        raise SystemExit("process() after close() did not raise")
    except RuntimeError:
        pass
d2.close()
"""


@pytest.mark.models
def test_denoiser_end_to_end_and_clean_shutdown():
    r = subprocess.run(
        [sys.executable, "-c", _DENOISER_SCRIPT],
        capture_output=True, text=True, timeout=300, check=False,
    )
    assert r.returncode == 0, f"exit={r.returncode}\n{r.stderr[-1200:]}"


_LONG_INPUT_SCRIPT = """
import numpy as np, libspeech
# The exported Facebook graph only handles ~32 s per call (it pads to that window internally and
# truncates longer input). 40 s must still come back as 40 s, with no hole at the window seam.
n = 40 * 16000
rng = np.random.default_rng(0)
t = np.arange(n) / 16000
# amplitude-modulated tone: the model keeps it (a steady tone is treated as noise and removed)
x = (0.3 * np.sin(2 * np.pi * 220 * t) * np.sin(2 * np.pi * 3 * t) ** 2
     + 0.02 * rng.standard_normal(n)).astype(np.float32)
with libspeech.Denoiser("facebook") as d:
    y = d(x)
assert y.shape == x.shape, (y.shape, x.shape)
assert np.isfinite(y).all()
rms = lambda a: float(np.sqrt(np.mean(a ** 2)))
# windows are 512000 samples with a 16000-sample cross-fade; the overlap is [496000, 512000)
before, overlap, after = (rms(y[a:a + 16000]) for a in (480000, 496000, 512000))
assert overlap > 0.8 * min(before, after), (before, overlap, after)  # no hole at the seam
assert rms(y[-16000:]) > 0.1 * rms(y[16000:48000]), "tail was dropped"
"""


@pytest.mark.models
def test_denoiser_input_longer_than_the_graph_window():
    r = subprocess.run(
        [sys.executable, "-c", _LONG_INPUT_SCRIPT],
        capture_output=True, text=True, timeout=300, check=False,
    )
    assert r.returncode == 0, f"exit={r.returncode}\n{r.stderr[-1200:]}"


def test_gtcrn_only_accepts_16khz():
    # Rejected before any model file is fetched, so this needs no network.
    for rate in (8000, 44100):
        with pytest.raises(ValueError, match="16 kHz"):
            libspeech.Denoiser("gtcrn", sample_rate=rate)


# GTCRN is tiny (~0.5 MB, well under a second per call), so unlike the big denoisers its end-to-end
# test is cheap. Still `models` (it downloads the file) and run in a subprocess like the others.
_GTCRN_SCRIPT = """
import numpy as np, libspeech
rng = np.random.default_rng(0)
rms = lambda a: float(np.sqrt(np.mean(a ** 2)))
with libspeech.Denoiser("gtcrn") as d:
    assert d.sample_rate == 16000
    # any length comes back unchanged in length: shorter than a hop, around the hop/frame sizes, longer
    for n in (1, 100, 255, 256, 257, 511, 513, 16000):
        y = d((0.1 * rng.standard_normal(n)).astype(np.float32))
        assert y.shape == (n,) and y.dtype == np.float32 and np.isfinite(y).all(), n
    # pure noise is what the model exists to remove: at least 20 dB less energy
    noise = (0.05 * rng.standard_normal(32000)).astype(np.float32)
    assert rms(d(noise)) < 0.1 * rms(noise), (rms(d(noise)), rms(noise))
    # the model is a real function of the input (not an identity) and is deterministic
    x = (0.1 * rng.standard_normal(5000)).astype(np.float32)
    assert not np.allclose(d(x), x)
    assert np.array_equal(d(x), d(x))
    # Golden values from an independent implementation (NumPy FFT + onnxruntime stepping the same
    # streaming model frame by frame, torch.stft/istft conventions). This is what pins the numerics --
    # the window shape, the framing and the state carried between frames -- which the checks above
    # cannot see. The input is closed-form so no RNG version can change it.
    i = np.arange(16000)
    t = i / 16000
    g = (0.3 * np.sin(2 * np.pi * 220 * t) * np.sin(2 * np.pi * 3 * t) ** 2
         + 0.05 * np.sin(1e-3 * i * i) + 0.02 * np.sin(0.7 * i)).astype(np.float32)
    gold = [-0.03803, -0.13327, 0.00407, 0.00827, -0.00499, -0.00043, 0.01226, 0.00266,
            -0.00334, -0.02823, 0.00018, 0.00984, -0.02509, -0.00045, 0.01568, 0.00538]
    got = d(g)[500:16000:1000]
    assert np.allclose(got, gold, atol=2e-4), np.round(got, 5).tolist()
    try:
        d(np.zeros(0, np.float32))
        raise SystemExit("empty input did not raise")
    except ValueError:
        pass
try:
    d(x)
    raise SystemExit("process() after close() did not raise")
except RuntimeError:
    pass
"""


@pytest.mark.models
def test_gtcrn_end_to_end():
    r = subprocess.run(
        [sys.executable, "-c", _GTCRN_SCRIPT],
        capture_output=True, text=True, timeout=300, check=False,
    )
    assert r.returncode == 0, f"exit={r.returncode}\n{r.stderr[-1200:]}"


def test_dpdfnet_rejects_a_negative_thread_count():
    # Checked before any model file is fetched, so this needs no network.
    with pytest.raises(ValueError):
        libspeech.Denoiser.create("dpdfnet", "dpdfnet2", num_threads=-1)


# Real speech is needed here: DPDFNet is selective enough to remove synthetic tones and vowels
# entirely, which would leave nothing to compare. The sample is the one sherpa-onnx publishes next
# to the models (a 9.8 s noisy 16 kHz clip), fetched like the model itself, hence `models`.
_DPDFNET_SCRIPT = """
import urllib.request, tempfile, wave
import numpy as np, libspeech

URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/inp_16k.wav"
with tempfile.NamedTemporaryFile(suffix=".wav") as f:
    f.write(urllib.request.urlopen(URL, timeout=60).read())
    f.flush()
    with wave.open(f.name) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768

with libspeech.Denoiser("dpdfnet") as d:       # dpdfnet2, the default variant
    assert d.sample_rate == 16000
    y = d(x)
    assert y.shape == x.shape and y.dtype == np.float32 and np.isfinite(y).all()
    # Golden values from an independent implementation (NumPy FFT + onnxruntime stepping the same
    # streaming model: reflect-padded centre frames, Vorbis window, state initialised from the
    # model's metadata, output advanced by the model's 4-hop latency). They pin what the checks
    # above cannot see: the window, the state initialisation, the latency shift, the framing.
    idx = [9752, 15908, 21916, 36519, 47820, 48983, 65997, 77397, 81572, 89891, 98623, 116322, 120181,
           134680, 143988, 146755]
    gold = [-0.12947, -0.44897, -0.41319, 0.23036, 0.25831, -0.17385, -0.46184, 0.24363, 0.35783,
            -0.32971, -0.32768, 0.14166, 0.31609, 0.20303, 0.23455, 0.19838]
    assert np.allclose(y[idx], gold, atol=2e-4), np.round(y[idx], 5).tolist()
    # it actually denoises: the quietest 20 ms frames (the background noise) get >30 dB quieter
    def floor(a):
        frames = a[: len(a) // 320 * 320].reshape(-1, 320)
        return np.percentile(np.sqrt((frames ** 2).mean(1)), 10)
    assert floor(y) < 0.03 * floor(x), (floor(x), floor(y))
    # any length, including shorter than one hop and the flushed tail, comes back unchanged in length
    for n in (1, 100, 159, 161, 5000):
        z = d(x[:n])
        assert z.shape == (n,) and np.isfinite(z).all(), n
    # the model works at 16 kHz only; asking for another rate is a clear error
try:
    libspeech.Denoiser("dpdfnet", sample_rate=48000)
    raise SystemExit("a 48 kHz request for a 16 kHz model did not raise")
except ValueError as e:
    assert "16000" in str(e), e
"""


@pytest.mark.models
def test_dpdfnet_end_to_end():
    r = subprocess.run(
        [sys.executable, "-c", _DPDFNET_SCRIPT],
        capture_output=True, text=True, timeout=300, check=False,
    )
    assert r.returncode == 0, f"exit={r.returncode}\n{r.stderr[-1200:]}"


SILERO_RELEASE_MODELS = ["silero-vad.onnx", "silero-vad-half.onnx", "silero-vad-16k-op15.onnx"]


@pytest.mark.models
@pytest.mark.parametrize("model", SILERO_RELEASE_MODELS)
def test_silero_vad_runs_on_every_released_model(model):
    # Regression test: the output tensor was looked up as "state_n" and an "sr" input was always fed, but
    # the released models name it "stateN" and silero-vad-half has no "sr" input, so process() raised
    # "Invalid output name: state_n" (or "Invalid input name") on every one of them.
    noise = (0.1 * np.random.default_rng(0).standard_normal(16000 * 3)).astype(np.float32).tolist()
    vad = libspeech.SileroVad(model_path=model)
    vad.process(noise)
    first = [(s.start_s, s.end_s) for s in vad.get_speech_timestamps()]
    vad.process(noise)  # process() resets the state, so the same input gives the same answer
    assert [(s.start_s, s.end_s) for s in vad.get_speech_timestamps()] == first
    for start, end in first:
        assert 0.0 <= start <= end <= 3.0
