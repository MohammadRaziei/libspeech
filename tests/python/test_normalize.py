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


# --- speech-weighted level (no model: the weights are given) -------------------------------------


def _weighted_rms(x: np.ndarray, w: np.ndarray, window: int) -> float:
    energy = count = 0.0
    for k, weight in enumerate(w):
        seg = np.asarray(x[k * window : (k + 1) * window], dtype=np.float64)
        energy += float(weight) * float(np.sum(seg**2))
        count += float(weight) * len(seg)
    return float(np.sqrt(energy / count)) if count else 0.0


def test_speech_weights_hard_and_soft():
    p = np.array([0.0, 0.4, 0.5, 0.6, 0.75, 1.0], dtype=np.float32)
    np.testing.assert_array_equal(libspeech.speech_weights(p, "hard", 0.5), [0, 0, 1, 1, 1, 1])
    np.testing.assert_allclose(libspeech.speech_weights(p, "soft", 0.5), [0, 0, 0, 0.2, 0.5, 1.0], atol=1e-6)
    assert libspeech.speech_weights(p).dtype == np.float32  # default mode is hard, threshold 0.5


@pytest.mark.parametrize(
    "kwargs",
    [{"mode": "x"}, {"threshold": 0.0}, {"threshold": 1.5}, {"mode": "soft", "threshold": 1.0}, {"threshold": float("nan")}],
)
def test_speech_weights_bad_arguments_raise(kwargs):
    with pytest.raises(ValueError):
        libspeech.speech_weights(np.array([0.2, 0.9], dtype=np.float32), **kwargs)


def test_speech_weights_probability_outside_unit_interval_raises():
    with pytest.raises(ValueError):
        libspeech.speech_weights(np.array([0.2, 1.3], dtype=np.float32))


def test_weighted_normalize_brings_the_weighted_level_to_the_target():
    rng = np.random.default_rng(3)
    window = 512
    x = (rng.standard_normal(window * 20) * np.repeat([0.002, 0.2, 0.01, 0.15], window * 5)).astype(np.float32)
    w = np.repeat(np.array([0, 1, 0.3, 0.7], dtype=np.float32), 5)
    y = libspeech.rms_normalize_weighted(x, w, window, target_dbfs=-25.0)
    assert 20 * np.log10(_weighted_rms(y, w, window)) == pytest.approx(-25.0, abs=1e-3)
    ratio = y[x != 0] / x[x != 0]  # one gain for every sample, including the weight-0 windows
    assert np.ptp(ratio) < 1e-4 * float(ratio[0])


def test_weighted_normalize_differs_from_plain_when_there_are_pauses():
    window = 512
    speech = _sine(0.1, 0.5)
    silence = np.full(SR * 3, 0.001, dtype=np.float32)
    x = np.concatenate([speech, silence])
    w = np.zeros(len(x) // window, dtype=np.float32)
    w[: len(speech) // window] = 1.0
    gated = libspeech.rms_normalize_weighted(x, w, window)
    plain = libspeech.rms_normalize(x)
    assert _dbfs(gated[: len(speech)]) == pytest.approx(-20.0, abs=0.05)
    assert _dbfs(plain[: len(speech)]) > -20.0 + 6.0


def test_weighted_normalize_without_weight_or_with_silence_is_unchanged_and_input_untouched():
    x = _sine(0.05)
    before = x.copy()
    np.testing.assert_array_equal(libspeech.rms_normalize_weighted(x, np.zeros(10, dtype=np.float32), 512), x)
    np.testing.assert_array_equal(libspeech.rms_normalize_weighted(x, np.zeros(0, dtype=np.float32), 512), x)
    z = np.zeros(2048, dtype=np.float32)
    np.testing.assert_array_equal(libspeech.rms_normalize_weighted(z, np.ones(4, dtype=np.float32), 512), z)
    np.testing.assert_array_equal(x, before)


@pytest.mark.parametrize("kwargs", [{"window": 0}, {"peak_ceiling": 0.0}, {"target_dbfs": float("nan")}])
def test_weighted_normalize_bad_arguments_raise(kwargs):
    args = {"window": 512, **kwargs}
    with pytest.raises(ValueError):
        libspeech.rms_normalize_weighted(_sine(0.1), np.ones(4, dtype=np.float32), **args)


def test_weighted_normalize_rejects_weights_outside_unit_interval_and_2d_signal():
    with pytest.raises(ValueError):
        libspeech.rms_normalize_weighted(_sine(0.1), np.array([1.0, 2.0], dtype=np.float32), 512)
    with pytest.raises(ValueError):
        libspeech.rms_normalize_weighted(np.zeros((2, 512), dtype=np.float32), np.ones(1, dtype=np.float32), 512)


# --- robust level, soft limiter, weighted CMVN (no model) -----------------------------------------


def _clipped_rms(x: np.ndarray, w: np.ndarray, window: int, percentile: float) -> float:
    """Reference from the definition (float64, plain sort)."""
    items = []
    for k, weight in enumerate(w):
        seg = np.asarray(x[k * window : (k + 1) * window], dtype=np.float64)
        if len(seg) and weight > 0:
            items.append((float(np.mean(seg**2)), float(weight) * len(seg)))
    items.sort()
    total = sum(m for _, m in items)
    acc, limit = 0.0, items[-1][0]
    for ms, mass in items:
        acc += mass
        if acc >= total * percentile / 100.0:
            limit = ms
            break
    return float(np.sqrt(sum(m * min(ms, limit) for ms, m in items) / total))


def test_weighted_rms_clip_percentile_matches_the_reference():
    rng = np.random.default_rng(21)
    window = 64
    x = (0.05 * rng.standard_normal(window * 70 - 9) * (1 + 5 * rng.random(window * 70 - 9) ** 2)).astype(np.float32)
    w = np.where(rng.random(70) < 0.25, 0.0, rng.random(70)).astype(np.float32)
    for p in (30.0, 60.0, 90.0, 100.0):
        assert libspeech.weighted_rms(x, w, window, p) == pytest.approx(_clipped_rms(x, w, window, p), rel=1e-9)
    assert libspeech.weighted_rms(x, w, window) == libspeech.weighted_rms(x, w, window, 100.0)


@pytest.mark.parametrize("p", [0.0, -1.0, 100.5, float("nan")])
def test_clip_percentile_out_of_range_raises(p):
    with pytest.raises(ValueError):
        libspeech.weighted_rms(_sine(0.1), np.ones(4, dtype=np.float32), 512, p)


def test_normalize_with_clip_percentile_gives_more_gain_when_a_few_windows_are_loud():
    window = 512
    x = _sine(0.05, 2.0)[: window * 40].copy()
    x[window * 3 : window * 4] *= 8
    w = np.ones(40, dtype=np.float32)
    plain = libspeech.rms_normalize_weighted(x, w, window, peak_ceiling=10.0)
    clipped = libspeech.rms_normalize_weighted(x, w, window, peak_ceiling=10.0, clip_percentile=90.0)
    assert _dbfs(clipped) > _dbfs(plain) + 0.9  # the speech part is louder: the plosive window no longer lowers the gain


def test_limit_peaks_ceiling_locality_and_validation():
    x = _sine(0.2, 1.0)[:10000].copy()
    orig = x.copy()
    x[5000] = 2.0
    y = libspeech.limit_peaks(x, 0.99, 40)
    assert float(np.max(np.abs(y))) <= 0.99
    assert abs(float(y[5000])) == pytest.approx(0.99, abs=1e-5)
    np.testing.assert_array_equal(y[:4960], orig[:4960])  # exactly 1 beyond lookahead - 1 samples from the peak
    np.testing.assert_array_equal(y[5040:], orig[5040:])
    assert abs(float(y[4995])) < abs(float(orig[4995]))  # the gain already falls before the peak
    quiet = _sine(0.3)
    np.testing.assert_array_equal(libspeech.limit_peaks(quiet), quiet)
    np.testing.assert_array_equal(x[5000:5001], np.array([2.0], dtype=np.float32))  # the input is not modified
    with pytest.raises(ValueError):
        libspeech.limit_peaks(np.array([0.1, np.nan], dtype=np.float32))
    with pytest.raises(ValueError):
        libspeech.limit_peaks(quiet, 0.0)


def test_limiter_keeps_the_speech_level_where_the_hard_ceiling_lowers_it():
    window = 512
    x = _sine(0.05, 2.0)[: window * 60].copy()
    x[20000] = 0.9
    w = np.ones(60, dtype=np.float32)
    hard = libspeech.rms_normalize_weighted(x, w, window)
    soft = libspeech.rms_normalize_weighted(x, w, window, limiter_lookahead=80)
    assert _dbfs(hard[2000:8000]) < -26.0
    assert _dbfs(soft[2000:8000]) == pytest.approx(-20.0, abs=0.2)
    assert float(np.max(np.abs(soft))) <= 0.99


def test_weighted_cmvn_uses_only_the_weighted_frames():
    rng = np.random.default_rng(4)
    frames, dim = 300, 6
    f = (rng.standard_normal((frames, dim)) * np.arange(1, dim + 1) + 5 * np.arange(dim)).astype(np.float32)
    f[:100] += 40.0
    w = np.zeros(frames, dtype=np.float32)
    w[100:] = rng.random(200)
    out = libspeech.weighted_cmvn(f, w)
    ww = w.astype(np.float64)
    mean = (ww[:, None] * out).sum(0) / ww.sum()
    var = (ww[:, None] * (out - mean) ** 2).sum(0) / ww.sum()
    np.testing.assert_allclose(mean, 0.0, atol=1e-4)
    np.testing.assert_allclose(var, 1.0, atol=1e-4)
    assert float(np.abs(out[0]).min()) > 3.0  # the silent frames were shifted by the speech statistics
    only_mean = libspeech.weighted_cmvn(f, w, normalize_variance=False)
    shift = only_mean - f
    np.testing.assert_allclose(shift, np.broadcast_to(shift[0], shift.shape), atol=1e-4)  # the same shift for every frame
    np.testing.assert_array_equal(libspeech.weighted_cmvn(f, np.zeros(frames, dtype=np.float32)), f)


def test_weighted_cmvn_bad_arguments_raise():
    f = np.zeros((4, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        libspeech.weighted_cmvn(f, np.ones(5, dtype=np.float32))
    with pytest.raises(ValueError):
        libspeech.weighted_cmvn(f.ravel(), np.ones(12, dtype=np.float32))
    with pytest.raises(ValueError):
        libspeech.weighted_cmvn(f, np.array([1, 2, 0, 0], dtype=np.float32))


def test_frame_weights_maps_windows_to_frames():
    w = np.array([0.0, 1.0, 0.5], dtype=np.float32)
    got = libspeech.frame_weights(w, window=512, hop_length=256, n_frames=8)
    np.testing.assert_array_equal(got, [0, 0, 1, 1, 0.5, 0.5, 0.5, 0.5])  # frame t -> window t * 256 // 512, clamped
    with pytest.raises(ValueError):
        libspeech.frame_weights(np.zeros(0, dtype=np.float32), 512, 256, 4)
    with pytest.raises(ValueError):
        libspeech.frame_weights(w, 0, 256, 4)


def test_limit_peaks_gain_is_a_ramp_that_recovers_after_the_peak():
    lookahead, at = 50, 3000
    x = np.full(6000, 0.2, dtype=np.float32)
    x[at] = 2.0
    y = libspeech.limit_peaks(x, 0.99, lookahead)
    gain = y.astype(np.float64) / np.where(np.arange(6000) == at, 2.0, float(np.float32(0.2)))
    assert gain.min() == pytest.approx(0.495, abs=1e-6)
    assert np.max(np.abs(np.diff(gain))) <= (1 - gain.min()) / lookahead + 1e-6  # a ramp, never a jump
    assert gain[at - lookahead + 1] < 1 - 1e-6
    assert gain[at + lookahead - 1] < 1 - 1e-6  # still releasing after the peak
    assert gain[at + lookahead] == pytest.approx(1.0, abs=1e-9)
    assert gain[at - lookahead] == pytest.approx(1.0, abs=1e-9)
