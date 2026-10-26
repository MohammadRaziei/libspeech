"""libspeech.speech_rms_normalize: the VAD's probabilities weight the RMS (hard and soft threshold).

Needs the real Silero model (downloaded on first use), so it is marked `models` (CMake target
speech_test_python_models_e2e). The audio is synthetic speech-like sound with long pauses, generated here.
"""

from __future__ import annotations

import numpy as np
import pytest

import libspeech

pytestmark = pytest.mark.models

SR = 16000


def _speech_like(pause_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Vowel-like bursts with `pause_s` of quiet noise between them. Returns (signal, mask of the bursts)."""
    scipy_signal = pytest.importorskip("scipy.signal")
    rng = np.random.default_rng(11)

    def vowel(dur, f0, formants):
        n = int(dur * SR)
        t = np.arange(n) / SR
        phase = np.cumsum(f0 * (1 + 0.08 * np.sin(2 * np.pi * 3 * t))) / SR
        y = np.diff(np.r_[0, np.where((phase % 1.0) < 0.08, 1.0, 0.0) - 0.08]) + 0.02 * rng.standard_normal(n)
        for fc, bw in formants:
            r = np.exp(-np.pi * bw / SR)
            y = scipy_signal.lfilter([1 - r], [1, -2 * r * np.cos(2 * np.pi * fc / SR), r * r], y)
        y = y * np.minimum(1, np.minimum(t / 0.03, (dur - t) / 0.04))
        return (0.3 * y / np.abs(y).max()).astype(np.float32)

    def quiet(d):
        return (0.002 * rng.standard_normal(int(d * SR))).astype(np.float32)

    parts, mask = [quiet(0.4)], [False]
    for f0, fm in [(130, [(700, 110), (1200, 120), (2600, 160)]), (150, [(300, 80), (2300, 150), (3000, 200)])]:
        parts += [vowel(0.6, f0, fm), quiet(pause_s)]
        mask += [True, False]
    return np.concatenate(parts), np.concatenate([np.full(len(p), m) for p, m in zip(parts, mask)])


@pytest.fixture(scope="module")
def vad() -> libspeech.SileroVad:
    return libspeech.SileroVad()


def _dbfs(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2))))


@pytest.mark.parametrize("mode", ["hard", "soft"])
def test_speech_level_hits_the_target_whatever_the_pause_length(vad, mode):
    levels = []
    for pause in (0.3, 3.0):
        x, mask = _speech_like(pause)
        y = libspeech.speech_rms_normalize(x, mode=mode, vad=vad)
        levels.append(_dbfs(y[mask]))
        assert float(np.max(np.abs(y))) <= 0.99
    # the speech level is the same for a recording with short and one with long pauses, and near the target
    assert abs(levels[0] - levels[1]) < 1.0
    assert levels[0] == pytest.approx(-20.0, abs=3.0)


def test_plain_normalize_depends_on_the_pause_length_but_gated_does_not(vad):
    short, mask_s = _speech_like(0.3)
    long_, mask_l = _speech_like(3.0)
    plain_gap = abs(_dbfs(libspeech.rms_normalize(short)[mask_s]) - _dbfs(libspeech.rms_normalize(long_)[mask_l]))
    gated_gap = abs(
        _dbfs(libspeech.speech_rms_normalize(short, vad=vad)[mask_s])
        - _dbfs(libspeech.speech_rms_normalize(long_, vad=vad)[mask_l])
    )
    assert plain_gap > 3.0
    assert gated_gap < plain_gap / 3


def test_matches_the_manual_pipeline_and_does_not_modify_the_input(vad):
    x, _ = _speech_like(1.0)
    before = x.copy()
    vad.reset()
    vad.process(x.tolist())
    probabilities = vad.speech_probabilities()
    for mode, threshold in (("hard", 0.5), ("soft", 0.3)):
        weights = libspeech.speech_weights(probabilities, mode, threshold)
        manual = libspeech.rms_normalize_weighted(x, weights, vad.window_samples)
        got = libspeech.speech_rms_normalize(x, mode=mode, threshold=threshold, vad=vad)
        np.testing.assert_array_equal(got, manual)
    np.testing.assert_array_equal(x, before)


def test_default_threshold_is_the_vads_and_noise_without_speech_is_unchanged(vad):
    rng = np.random.default_rng(5)
    noise = (0.003 * rng.standard_normal(SR * 2)).astype(np.float32)
    np.testing.assert_array_equal(libspeech.speech_rms_normalize(noise, vad=vad), noise)
    x, _ = _speech_like(0.5)
    vad.set_threshold(0.5)
    np.testing.assert_array_equal(
        libspeech.speech_rms_normalize(x, vad=vad), libspeech.speech_rms_normalize(x, threshold=0.5, vad=vad)
    )


def test_bad_arguments_raise(vad):
    with pytest.raises(ValueError):
        libspeech.speech_rms_normalize(np.zeros((2, SR), dtype=np.float32), vad=vad)
    with pytest.raises(ValueError):
        libspeech.speech_rms_normalize(np.zeros(SR, dtype=np.float32), mode="medium", vad=vad)
