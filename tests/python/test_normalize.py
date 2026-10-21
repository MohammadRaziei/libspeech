"""RMS normalization: libspeech.rms_normalize / libspeech.rms_dbfs and Audio.normalize_rms.

No models needed. Every expected value is computed independently in float64 NumPy, not with libspeech.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import libspeech

SR = 16000


def _sine(amplitude: float, seconds: float = 1.0, freq: float = 200.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _dbfs(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2))))


def _audio(channels: list[np.ndarray]) -> libspeech.Audio:
    audio = libspeech.Audio()
    assert audio.load([c.tolist() for c in channels], SR)
    return audio


def test_rms_dbfs_of_known_signals():
    assert libspeech.rms_dbfs(_sine(1.0)) == pytest.approx(-3.0103, abs=1e-3)
    assert libspeech.rms_dbfs(np.ones(100, dtype=np.float32)) == pytest.approx(0.0, abs=1e-9)
    assert libspeech.rms_dbfs(np.zeros(100, dtype=np.float32)) == -math.inf


@pytest.mark.parametrize("amplitude", [0.001, 0.05, 0.9, 7.0])
@pytest.mark.parametrize("target", [-30.0, -20.0, -12.0])
def test_hits_the_target_level_from_any_input_level(amplitude, target):
    y = libspeech.rms_normalize(_sine(amplitude), target_dbfs=target)
    assert y.dtype == np.float32
    assert _dbfs(y) == pytest.approx(target, abs=1e-3)
    assert float(np.max(np.abs(y))) <= 0.99


def test_default_target_is_minus_20_dbfs():
    assert _dbfs(libspeech.rms_normalize(_sine(0.01))) == pytest.approx(-20.0, abs=1e-3)


def test_input_is_not_modified_and_shape_is_kept():
    x = _sine(0.02)
    before = x.copy()
    y = libspeech.rms_normalize(x)
    np.testing.assert_array_equal(x, before)
    assert y.shape == x.shape
    ratio = y[x != 0] / x[x != 0]  # one gain for every sample
    assert np.ptp(ratio) < 1e-4 * float(ratio[0])


def test_peak_ceiling_wins_over_the_target():
    x = np.full(16000, 0.001, dtype=np.float32)
    x[::1000] = 0.5
    y = libspeech.rms_normalize(x, target_dbfs=-10.0)
    assert float(np.max(np.abs(y))) == pytest.approx(0.99, abs=1e-6)
    assert _dbfs(y) < -10.0
    z = libspeech.rms_normalize(x, target_dbfs=-10.0, peak_ceiling=0.5)
    assert float(np.max(np.abs(z))) == pytest.approx(0.5, abs=1e-6)


def test_silence_and_non_finite_input_come_back_unchanged():
    silence = np.zeros(1000, dtype=np.float32)
    np.testing.assert_array_equal(libspeech.rms_normalize(silence), silence)
    bad = _sine(0.1)[:100].copy()
    bad[10] = np.nan
    out = libspeech.rms_normalize(bad)
    np.testing.assert_array_equal(np.isnan(out), np.isnan(bad))
    np.testing.assert_array_equal(out[~np.isnan(out)], bad[~np.isnan(bad)])


def test_two_dimensional_input_gets_one_gain_for_all_channels():
    quiet, loud = _sine(0.02, freq=300.0), _sine(0.2, freq=200.0)
    x = np.stack([loud, quiet])  # (channels, samples)
    y = libspeech.rms_normalize(x, target_dbfs=-20.0)
    assert y.shape == x.shape
    assert _dbfs(y) == pytest.approx(-20.0, abs=1e-3)
    assert _dbfs(y[0]) - _dbfs(y[1]) == pytest.approx(_dbfs(loud) - _dbfs(quiet), abs=1e-3)  # balance kept


def test_non_contiguous_and_float64_input_are_accepted():
    x = _sine(0.05)
    strided = np.stack([x, x], axis=1)[:, 0]  # a non-contiguous view
    assert _dbfs(libspeech.rms_normalize(strided)) == pytest.approx(-20.0, abs=1e-3)
    assert _dbfs(libspeech.rms_normalize(x.astype(np.float64))) == pytest.approx(-20.0, abs=1e-3)


@pytest.mark.parametrize(
    "kwargs",
    [{"target_dbfs": float("nan")}, {"target_dbfs": float("-inf")}, {"peak_ceiling": 0.0}, {"peak_ceiling": -1.0}],
)
def test_bad_arguments_raise_value_error(kwargs):
    with pytest.raises(ValueError):
        libspeech.rms_normalize(_sine(0.1), **kwargs)


def test_three_dimensional_input_raises():
    with pytest.raises(ValueError):
        libspeech.rms_normalize(np.zeros((2, 2, 2), dtype=np.float32))


# --- Audio.normalize_rms ---------------------------------------------------------------------


def test_audio_normalize_rms_returns_a_new_normalized_audio():
    audio = _audio([_sine(0.01)])
    out = audio.normalize_rms(-20.0)
    assert isinstance(out, libspeech.Audio)
    assert out.sample_rate == SR
    assert out.shape == audio.shape
    assert _dbfs(out.to_numpy()) == pytest.approx(-20.0, abs=1e-3)
    assert _dbfs(audio.to_numpy()) == pytest.approx(_dbfs(_sine(0.01)), abs=1e-3)  # the original is untouched


def test_audio_normalize_rms_keeps_the_balance_between_channels():
    left, right = _sine(0.2, freq=200.0), _sine(0.02, freq=300.0)
    out = _audio([left, right]).normalize_rms(-20.0)
    both = np.stack([out.to_numpy(0), out.to_numpy(1)])
    assert _dbfs(both) == pytest.approx(-20.0, abs=1e-3)
    assert _dbfs(both[0]) - _dbfs(both[1]) == pytest.approx(_dbfs(left) - _dbfs(right), abs=1e-3)


def test_audio_normalize_rms_default_ceiling_and_errors():
    out = _audio([_sine(0.5)]).normalize_rms(-1.0)  # -1 dBFS RMS needs a peak above 1: the ceiling stops it
    assert float(np.max(np.abs(out.to_numpy()))) == pytest.approx(0.99, abs=1e-6)
    with pytest.raises(ValueError):
        _audio([_sine(0.5)]).normalize_rms(float("nan"))
    with pytest.raises(ValueError):
        libspeech.Audio().normalize_rms(-20.0, 0.0)  # arguments are checked even for empty audio


def test_empty_audio_normalizes_to_empty_audio():
    assert len(libspeech.Audio().normalize_rms()) == 0
