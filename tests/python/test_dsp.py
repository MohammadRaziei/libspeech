"""Tests for libspeech's speech::dsp bindings (Resample, window, FFT, STFT,
MFCC, dct). Mirrors the correctness properties already validated in the
C++ test suite (tests/dsp/), just exercised through the Python surface.
"""

from __future__ import annotations

import math

import pytest

import libspeech


def make_sine(n: int, freq_hz: float, sample_rate: int) -> list[float]:
    return [math.sin(2 * math.pi * freq_hz * i / sample_rate) for i in range(n)]


def test_resample_changes_length():
    r = libspeech.Resample(44100, 16000)
    signal = make_sine(44100, 440.0, 44100)
    out = r.resample(signal)
    assert abs(len(out) - 16000) < 200


def test_resample_same_rate_is_identity():
    r = libspeech.Resample(16000, 16000)
    signal = make_sine(1000, 440.0, 16000)
    out = r.resample(signal)
    assert out == pytest.approx(signal, abs=1e-6)


def test_window_hann_peaks_at_one():
    w = libspeech.window(libspeech.WindowType.hann, 512)
    assert len(w) == 512
    assert max(w) == pytest.approx(1.0, abs=1e-4)


def test_window_rect_is_all_ones():
    w = libspeech.window(libspeech.WindowType.rect, 100)
    assert all(v == 1.0 for v in w)


def test_fft_impulse_has_flat_spectrum():
    fft = libspeech.FFT(6)  # length 64
    real = [1.0] + [0.0] * 63
    out_real, out_imag = fft.forward(real)
    assert all(v == pytest.approx(1.0, abs=1e-4) for v in out_real)
    assert all(v == pytest.approx(0.0, abs=1e-4) for v in out_imag)


def test_fft_forward_inverse_round_trip():
    fft = libspeech.FFT(6)
    signal = make_sine(64, 3.0, 64)
    freq_real, freq_imag = fft.forward(signal)
    back_real, back_imag = fft.inverse(freq_real, freq_imag)
    assert back_real == pytest.approx(signal, abs=1e-3)


def test_stft_produces_expected_frame_shape():
    stft = libspeech.STFT(9)  # fft_length=512, slide_length=128
    assert stft.fft_length == 512
    assert stft.slide_length == 128

    data_length = 512 + 128 * 9  # 10 frames
    signal = make_sine(data_length, 440.0, 16000)
    real, imag = stft.stft(signal)
    assert len(real) == 10
    assert len(real[0]) == 512


def test_mfcc_produces_expected_shape():
    params = libspeech.MFCCParams()
    params.sample_rate = 16000
    params.num_mel_filters = 26
    params.num_coefficients = 13
    params.radix2_exp = 9

    mfcc = libspeech.MFCC(params)
    data_length = 512 + 128 * 9  # 10 frames
    signal = make_sine(data_length, 440.0, 16000)
    coeffs = mfcc.compute(signal)

    assert len(coeffs) == 10
    assert all(len(frame) == 13 for frame in coeffs)


def test_dct_of_constant_has_energy_only_in_dc_term():
    out = libspeech.dct([3.0] * 32)
    assert abs(out[0]) > 1.0
    assert all(abs(v) < 1e-3 for v in out[1:])


def test_resample_quality_levels():
    np = pytest.importorskip("numpy")
    n = 44100
    x = (0.5 * np.sin(2 * np.pi * 3000 * np.arange(n) / 44100)).astype(np.float32)
    ref = libspeech.Resample(44100, 16000).resample(x)
    assert libspeech.Resample(44100, 16000, "best").resample(x).tolist() == ref.tolist()
    for quality in ("high", "medium", "fast"):
        y = libspeech.Resample(44100, 16000, quality).resample(x)
        assert y.shape == ref.shape
        mid = slice(2000, -2000)  # away from the filter edges
        assert abs(float(np.abs(y[mid]).max()) - 0.5) < 0.01, quality
        assert float(np.abs(y[mid] - ref[mid]).max()) < 0.01, quality  # same tone, same phase
    with pytest.raises(ValueError):
        libspeech.Resample(44100, 16000, "ultra")


def _mfcc(num_mel=40, num_coef=13, exp=9):
    params = libspeech.MFCCParams()
    params.sample_rate = 16000
    params.num_mel_filters = num_mel
    params.num_coefficients = num_coef
    params.radix2_exp = exp
    return libspeech.MFCC(params)


def test_mfcc_log_mel_shape_and_dtype():
    np = pytest.importorskip("numpy")
    x = np.asarray(make_sine(16000, 440.0, 16000), dtype=np.float32)
    mel = _mfcc(num_mel=26).log_mel(x)
    assert mel.dtype == np.float32
    assert mel.ndim == 2 and mel.shape[1] == 26
    assert mel.shape[0] == len(_mfcc(num_mel=26).compute(x))
    assert np.isfinite(mel).all()


def test_mfcc_is_orthonormal_dct_of_log_mel():
    np = pytest.importorskip("numpy")
    scipy_fft = pytest.importorskip("scipy.fft")
    rng = np.random.default_rng(0)
    x = (np.asarray(make_sine(16000, 700.0, 16000)) + 0.05 * rng.standard_normal(16000)).astype(np.float32)
    m = _mfcc(num_mel=40, num_coef=13)
    mel = m.log_mel(x)
    ref = scipy_fft.dct(mel.astype(np.float64), type=2, norm="ortho", axis=1)[:, :13]
    got = np.asarray(m.compute(x))
    assert got.shape == ref.shape
    np.testing.assert_allclose(got, ref, atol=2e-3)


def test_mfcc_log_mel_tone_lands_in_expected_band():
    np = pytest.importorskip("numpy")
    m = _mfcc(num_mel=40, exp=10)
    low = m.log_mel(np.asarray(make_sine(8000, 300.0, 16000), dtype=np.float32))
    high = m.log_mel(np.asarray(make_sine(8000, 4000.0, 16000), dtype=np.float32))
    assert low[low.shape[0] // 2].argmax() < 12
    assert high[high.shape[0] // 2].argmax() > 25
