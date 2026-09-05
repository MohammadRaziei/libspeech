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
