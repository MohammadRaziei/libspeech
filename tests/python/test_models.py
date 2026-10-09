"""Tests for the libspeech.Denoiser / libspeech.SileroVad / libspeech.SpeechTimestamp
bindings. These don't exercise real ONNX inference (that needs downloaded
model weights, out of scope for a fast unit test) -- they validate the
binding surface itself: construction, error propagation, and correctness
of the plain-data SpeechTimestamp helper.
"""

from __future__ import annotations

import subprocess
import sys

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

d = libspeech.Denoiser.create("facebook", MODEL)
assert d.sample_rate == 16000 and not d.closed
y = d.process(x)  # the only real inference: NumPy in -> float32 NumPy out, same length
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
