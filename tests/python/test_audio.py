"""Tests for the libspeech Python package (nanobind bindings).

Run via `cmake --build build --target speech_test_python` (or plain
`pytest`, once the package is on PYTHONPATH -- see tests/python/CMakeLists.txt
for how CMake wires that up automatically). These exercise the PUBLIC
`import libspeech` surface, not the private `_audio`/`_about` extension
modules directly -- see libspeech/__init__.py for why.
"""

from __future__ import annotations

import math

import pytest

import libspeech


def make_sine(n: int, freq_hz: float, sample_rate: int) -> list[float]:
    return [math.sin(2 * math.pi * freq_hz * i / sample_rate) for i in range(n)]


def test_version_is_a_nonempty_string():
    assert isinstance(libspeech.__version__, str)
    assert len(libspeech.__version__) > 0


def test_load_from_raw_pcm_sets_sample_rate_and_duration():
    sample_rate = 16000
    samples = make_sine(sample_rate, 440.0, sample_rate)  # 1 second

    audio = libspeech.Audio()
    assert audio.load([samples], sample_rate) is True
    assert audio.sample_rate == sample_rate
    assert audio.duration == pytest.approx(1.0, abs=1e-6)


def test_len_is_sample_count_not_channel_count():
    # Regression test: __len__ returns samples-per-channel, not the number
    # of channels -- an earlier version of this binding had this backwards
    # in its docstring/repr.
    sample_rate = 8000
    samples = make_sine(sample_rate, 220.0, sample_rate)

    audio = libspeech.Audio()
    audio.load([samples], sample_rate)

    assert len(audio) == sample_rate
    assert len(audio.data()) == 1  # one channel


def test_to_mono_preserves_sample_rate_and_duration():
    sample_rate = 16000
    left = make_sine(sample_rate, 440.0, sample_rate)
    right = make_sine(sample_rate, 440.0, sample_rate)

    audio = libspeech.Audio()
    audio.load([left, right], sample_rate)
    assert len(audio.data()) == 2

    mono = audio.to_mono()
    assert len(mono.data()) == 1
    assert mono.sample_rate == sample_rate
    assert mono.duration == pytest.approx(audio.duration, abs=1e-6)


def test_resample_changes_sample_rate_and_length():
    source_rate, target_rate = 44100, 16000
    samples = make_sine(source_rate, 440.0, source_rate)

    audio = libspeech.Audio()
    audio.load([samples], source_rate)

    resampled = audio.resample(target_rate)
    assert resampled.sample_rate == target_rate
    # Duration should be (roughly) preserved across a resample.
    assert resampled.duration == pytest.approx(audio.duration, abs=0.05)


def test_resample_to_same_rate_is_a_no_op_copy():
    sample_rate = 16000
    samples = make_sine(sample_rate, 440.0, sample_rate)

    audio = libspeech.Audio()
    audio.load([samples], sample_rate)

    same = audio.resample(sample_rate)
    assert same.sample_rate == sample_rate
    assert same.data(0) == pytest.approx(audio.data(0), abs=1e-6)


def test_save_and_load_wav_round_trip(tmp_path):
    sample_rate = 16000
    samples = make_sine(sample_rate, 440.0, sample_rate)

    audio = libspeech.Audio()
    audio.load([samples], sample_rate)

    wav_path = tmp_path / "test_audio.wav"
    assert audio.save(str(wav_path)) is True
    assert wav_path.exists()

    reloaded = libspeech.Audio()
    assert reloaded.load(str(wav_path)) is True
    assert reloaded.sample_rate == sample_rate
    assert reloaded.duration == pytest.approx(audio.duration, abs=1e-3)


def test_repr_reports_correct_channel_count():
    # Regression test: __repr__ used to report len(audio) (sample count) as
    # the channel count. It should report len(audio.data()) instead.
    sample_rate = 8000
    samples = make_sine(sample_rate, 220.0, sample_rate)

    audio = libspeech.Audio()
    audio.load([samples, samples], sample_rate)  # 2 channels

    assert "channels=2" in repr(audio)
    assert f"channels={sample_rate}" not in repr(audio)
