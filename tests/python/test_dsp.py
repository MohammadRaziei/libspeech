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
    assert mel.ndim == 2
    assert mel.shape[1] == 26
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


# ---- delta features (cross-checked against librosa and an independent NumPy reference) -----------------


def _delta_replicate_reference(x, half_window):
    """Straight-from-the-definition float64 reference with replicate padding; x is (frames, features)."""
    np = pytest.importorskip("numpy")
    T = x.shape[0]
    den = 2 * sum(n * n for n in range(1, half_window + 1))
    out = np.zeros(x.shape, dtype=np.float64)
    for t in range(T):
        for n in range(1, half_window + 1):
            out[t] += n * (x[min(t + n, T - 1)] - x[max(t - n, 0)])
    return out / den


@pytest.mark.parametrize("half_window", [1, 2, 4])
def test_delta_replicate_matches_numpy_reference(half_window):
    np = pytest.importorskip("numpy")
    x = np.random.default_rng(1).standard_normal((50, 13)).astype(np.float32)
    got = libspeech.delta(x, half_window=half_window)
    np.testing.assert_allclose(got, _delta_replicate_reference(x.astype(np.float64), half_window), atol=1e-5)


@pytest.mark.parametrize("half_window", [1, 2, 4])
@pytest.mark.parametrize("order", [1, 2])
def test_delta_interp_matches_librosa(half_window, order):
    np = pytest.importorskip("numpy")
    librosa = pytest.importorskip("librosa")
    x = np.random.default_rng(2).standard_normal((60, 13)).astype(np.float32)  # (frames, features)
    # librosa wants (features, frames) and a window width of 2*half_window + 1. Our order=2 is delta-delta in the
    # HTK/Kaldi sense (delta applied twice), so the reference applies librosa's order=1 twice: librosa's own
    # order=2 is a different quantity (second derivative of a quadratic Savitzky-Golay fit).
    want = x.T.astype(np.float64)
    for _ in range(order):
        want = librosa.feature.delta(want, width=2 * half_window + 1, order=1, axis=-1, mode="interp")
    want = want.T
    got = libspeech.delta(x, half_window=half_window, order=order, edge="interp")
    assert got.shape == x.shape
    assert got.dtype == np.float32
    np.testing.assert_allclose(got, want, atol=2e-5)


def test_delta_of_real_mfcc_matches_librosa_on_libspeech_features():
    np = pytest.importorskip("numpy")
    librosa = pytest.importorskip("librosa")
    x = np.asarray(make_sine(16000, 440.0, 16000), dtype=np.float32)
    x += 0.1 * np.random.default_rng(3).standard_normal(x.size).astype(np.float32)
    feats = np.asarray(_mfcc(num_mel=40, num_coef=13).compute(x), dtype=np.float32)
    d1 = librosa.feature.delta(feats.T.astype(np.float64), width=9, order=1, axis=-1, mode="interp")
    want = librosa.feature.delta(d1, width=9, order=1, axis=-1, mode="interp").T
    got = libspeech.delta(feats, half_window=4, order=2, edge="interp")
    np.testing.assert_allclose(got, want, atol=1e-4)


def test_delta_argument_validation():
    np = pytest.importorskip("numpy")
    x = np.zeros((4, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        libspeech.delta(x, edge="nearest")
    with pytest.raises(ValueError):
        libspeech.delta(x, order=0)
    with pytest.raises(ValueError):
        libspeech.delta(x, half_window=0)
    with pytest.raises(ValueError):
        libspeech.delta(x, half_window=2, edge="interp")  # 4 frames < 2*2+1
