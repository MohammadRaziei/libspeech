"""Tests for the libspeech.Denoiser / libspeech.SileroVad / libspeech.SpeechTimestamp
bindings. These don't exercise real ONNX inference (that needs downloaded
model weights, out of scope for a fast unit test) -- they validate the
binding surface itself: construction, error propagation, and correctness
of the plain-data SpeechTimestamp helper.
"""

from __future__ import annotations

from pathlib import Path

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


_FB_MODEL = Path.home() / ".libspeech" / "facebook-denoiser-dns64.onnx"


@pytest.mark.skipif(not _FB_MODEL.exists(), reason="needs facebook-denoiser-dns64.onnx in ~/.libspeech")
def test_denoiser_close_and_destroy():
    # Regression: Denoiser.create() used to fail to convert its return value, and destroying
    # the object aborted the process (free(): invalid pointer).
    d = libspeech.Denoiser.create("facebook", str(_FB_MODEL))
    audio = [0.0] * 1600
    assert len(d.process(audio)) == len(audio)
    d.close()
    d.close()  # idempotent
    with pytest.raises(RuntimeError):
        d.process(audio)
    del d
    with libspeech.Denoiser.create("facebook", str(_FB_MODEL)) as d2:
        assert len(d2.process(audio)) == len(audio)
