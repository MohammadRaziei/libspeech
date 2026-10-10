"""Tests for the NumPy fast paths of the DSP/Audio bindings.

A float32 ndarray goes in, a NumPy array that owns the C++ buffer comes out --
no list conversion, no copies. The list API must keep working unchanged, and
both must agree. NumPy is optional for libspeech itself, so this whole module
is skipped when it is not installed.
"""

from __future__ import annotations

import gc
import math

import pytest

import libspeech

np = pytest.importorskip("numpy")


def speech_like(n: int, sample_rate: int) -> np.ndarray:
    t = np.arange(n) / sample_rate
    return (0.4 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 1900 * t)).astype(np.float32)


def mfcc_params(sample_rate: int = 16000) -> libspeech.MFCCParams:
    p = libspeech.MFCCParams()
    p.sample_rate = sample_rate
    p.num_mel_filters = 26
    p.num_coefficients = 13
    p.radix2_exp = 9
    p.slide_length = 128
    return p


# --- STFT ---------------------------------------------------------------

def test_stft_numpy_returns_float32_arrays_equal_to_list_api():
    x = speech_like(16000, 16000)
    re, im = libspeech.STFT(9, libspeech.WindowType.hann, 128).stft(x)
    re_l, im_l = libspeech.STFT(9, libspeech.WindowType.hann, 128).stft(x.tolist())

    assert isinstance(re, np.ndarray)
    assert re.dtype == np.float32
    assert re.shape == im.shape == (122, 512)
    assert np.array_equal(re, np.asarray(re_l, dtype=np.float32))
    assert np.array_equal(im, np.asarray(im_l, dtype=np.float32))


def test_stft_onesided_is_the_first_half_of_the_full_spectrum():
    x = speech_like(16000, 16000)
    stft = libspeech.STFT(9, libspeech.WindowType.hann, 128)
    assert stft.num_bins() == 512
    assert stft.num_bins(onesided=True) == 257

    re_f, im_f = stft.stft(x)
    re_h, im_h = stft.stft(x, onesided=True)
    assert re_h.shape == im_h.shape == (122, 257)
    assert np.array_equal(re_h, re_f[:, :257])
    assert np.array_equal(im_h, im_f[:, :257])
    # the full spectrum's upper half is the conjugate mirror of the lower
    assert np.array_equal(re_f[:, 1:256][:, ::-1], re_f[:, 257:])
    assert np.array_equal(-im_f[:, 1:256][:, ::-1], im_f[:, 257:])


def test_stft_matches_numpys_rfft_of_the_windowed_frames():
    x = speech_like(8000, 16000)
    n_fft, hop = 512, 128
    re, im = libspeech.STFT(9, libspeech.WindowType.hann, hop).stft(x, onesided=True)

    window = np.hanning(n_fft + 1)[:-1]  # periodic Hann, same window the STFT applies
    frames = np.stack([x[i * hop : i * hop + n_fft] * window for i in range(re.shape[0])])
    ref = np.fft.rfft(frames.astype(np.float64), axis=1)
    got = re.astype(np.float64) + 1j * im.astype(np.float64)
    assert np.abs(got - ref).max() < 1e-3 * max(1.0, np.abs(ref).max())


def test_stft_numpy_accepts_float64_and_strided_input():
    x = speech_like(16000, 16000)
    ref, _ = libspeech.STFT(9).stft(x)
    re64, _ = libspeech.STFT(9).stft(x.astype(np.float64))
    assert np.allclose(re64, ref, atol=1e-5)

    interleaved = np.repeat(x, 2)[::2]  # non-contiguous view of the same samples
    assert not interleaved.flags["C_CONTIGUOUS"]
    re_strided, _ = libspeech.STFT(9).stft(interleaved)
    assert np.array_equal(re_strided, ref)


def test_stft_numpy_short_input_gives_zero_frames():
    re, im = libspeech.STFT(9).stft(np.zeros(100, dtype=np.float32))
    assert re.shape[0] == 0
    assert im.shape[0] == 0


def _spectrum_of_a_temporary_signal():
    """STFT of a signal that no longer exists once this returns."""
    x = speech_like(16000, 16000)
    return libspeech.STFT(9).stft(x)[0]


def test_numpy_result_outlives_the_producing_objects():
    out = _spectrum_of_a_temporary_signal()
    expected = out.copy()
    gc.collect()
    junk = [np.random.rand(10000) for _ in range(50)]  # churn the allocator
    gc.collect()
    assert np.array_equal(out, expected)
    del junk


# --- MFCC ---------------------------------------------------------------

# --- FFT ------------------------------------------------------------------

@pytest.mark.parametrize("radix2_exp", [4, 9, 13, 16])  # 16 uses the heap-scratch path (> 8192)
def test_fft_numpy_forward_matches_numpys_fft(radix2_exp):
    n = 1 << radix2_exp
    x = _noise(n, seed=radix2_exp)
    re, im = libspeech.FFT(radix2_exp).forward(x)

    assert re.dtype == np.float32
    assert im.dtype == np.float32
    assert re.shape == (n,)
    ref = np.fft.fft(x.astype(np.float64))
    scale = np.abs(ref).max()
    assert np.abs(re - ref.real).max() < 1e-4 * scale
    assert np.abs(im - ref.imag).max() < 1e-4 * scale


def test_fft_numpy_equals_the_list_api():
    fft = libspeech.FFT(10)
    x, y = _noise(1024, seed=1), _noise(1024, seed=2)
    for args in ((x,), (x, y)):  # real-only (fast path) and complex
        re_a, im_a = fft.forward(*args)
        re_l, im_l = fft.forward(*(a.tolist() for a in args))
        assert np.abs(re_a - np.asarray(re_l, dtype=np.float32)).max() < 1e-3
        assert np.abs(im_a - np.asarray(im_l, dtype=np.float32)).max() < 1e-3


def test_fft_numpy_inverse_round_trips_and_matches_numpy():
    fft = libspeech.FFT(11)
    x, y = _noise(2048, seed=3), _noise(2048, seed=4)
    re, im = fft.forward(x, y)
    back_re, back_im = fft.inverse(re, im)
    assert np.abs(back_re - x).max() < 1e-4
    assert np.abs(back_im - y).max() < 1e-4

    ref = np.fft.ifft(x.astype(np.float64) + 1j * y)
    inv_re, inv_im = fft.inverse(x, y)
    assert np.abs(inv_re - ref.real).max() < 1e-4 * np.abs(ref).max()
    assert np.abs(inv_im - ref.imag).max() < 1e-4 * np.abs(ref).max()


def test_fft_numpy_dct_round_trips_and_idct_keeps_its_input():
    fft = libspeech.FFT(9)
    x = _noise(512, seed=5)
    for is_norm in (True, False):
        c = fft.dct(x, is_norm)
        assert c.dtype == np.float32
        assert np.abs(np.asarray(fft.dct(x.tolist(), is_norm), dtype=np.float32) - c).max() < 1e-3
        kept = c.copy()
        back = fft.idct(c, is_norm)
        assert np.array_equal(c, kept)  # idct must not modify its input
        assert np.abs(back - x).max() < 1e-4


def test_fft_numpy_accepts_float64_and_non_contiguous_input():
    fft = libspeech.FFT(9)
    x = _noise(512, seed=6)
    ref_re, ref_im = fft.forward(x)
    re64, im64 = fft.forward(x.astype(np.float64))
    assert np.array_equal(re64, ref_re)
    assert np.array_equal(im64, ref_im)
    strided = np.zeros(1024, dtype=np.float32)
    strided[::2] = x
    s_re, _ = fft.forward(strided[::2])
    assert np.array_equal(s_re, ref_re)


def test_fft_numpy_validates_lengths():
    fft = libspeech.FFT(6)
    ok, short = _noise(64), _noise(63)
    with pytest.raises(ValueError):
        fft.forward(short)
    with pytest.raises(ValueError):
        fft.forward(ok, short)
    with pytest.raises(ValueError):
        fft.inverse(ok, short)
    with pytest.raises(ValueError):
        fft.dct(short)
    with pytest.raises(ValueError):
        fft.idct(short)


# --- inverse STFT ---------------------------------------------------------

def _noise(n: int, seed: int = 0) -> np.ndarray:
    return (0.3 * np.random.default_rng(seed).standard_normal(n)).astype(np.float32)


@pytest.mark.parametrize("onesided", [False, True])
@pytest.mark.parametrize("method_type", [0, 1])
def test_istft_numpy_round_trips_the_stft(onesided, method_type):
    x = _noise(512 * 12)
    stft = libspeech.STFT(9, libspeech.WindowType.hann)
    re, im = stft.stft(x, onesided=onesided)
    y = stft.istft(re, im, method_type)

    assert isinstance(y, np.ndarray)
    assert y.dtype == np.float32
    assert y.ndim == 1
    assert y.shape[0] == stft.cal_data_length(re.shape[0])
    # away from the first/last fft_length samples, where the window tapers to zero
    assert np.abs(y[512:-512] - x[512 : len(y) - 512]).max() < 2e-4


def test_istft_numpy_equals_the_list_api():
    x = _noise(256 * 10, seed=1)
    stft = libspeech.STFT(8, libspeech.WindowType.hann, 64)
    re, im = stft.stft(x)
    from_lists = stft.istft(re.tolist(), im.tolist())
    from_arrays = stft.istft(re, im)
    assert isinstance(from_lists, list)
    assert np.abs(np.asarray(from_lists, dtype=np.float32) - from_arrays).max() < 1e-6


def test_istft_numpy_accepts_float64_and_non_contiguous_input():
    x = _noise(512 * 6, seed=2)
    stft = libspeech.STFT(9, libspeech.WindowType.hann)
    re, im = stft.stft(x, onesided=True)
    ref = stft.istft(re, im)
    assert np.array_equal(stft.istft(re.astype(np.float64), im.astype(np.float64)), ref)
    assert np.array_equal(stft.istft(np.asfortranarray(re), np.asfortranarray(im)), ref)


def test_istft_numpy_result_is_independent_of_its_inputs():
    stft = libspeech.STFT(9, libspeech.WindowType.hann)
    re, im = stft.stft(_noise(512 * 6, seed=3), onesided=True)
    y = stft.istft(re, im)
    expected = y.copy()
    del re, im, stft
    gc.collect()
    assert np.array_equal(y, expected)


def test_istft_numpy_validates_its_arguments():
    stft = libspeech.STFT(9, libspeech.WindowType.hann)
    re, im = stft.stft(_noise(512 * 6, seed=4), onesided=True)  # (frames, 257)

    with pytest.raises(ValueError):
        stft.istft(re, im[:-1])  # shapes differ
    with pytest.raises(ValueError):
        stft.istft(re[:, :100], im[:, :100])  # neither 512 nor 257 columns
    with pytest.raises(ValueError):
        stft.istft(re, im, 2)  # method_type must be 0 or 1
    with pytest.raises(ValueError):
        stft.istft(re[:0], im[:0])  # no frames


def test_mfcc_numpy_equals_list_api():
    x = speech_like(16000, 16000)
    m = libspeech.MFCC(mfcc_params()).compute(x)
    m_l = libspeech.MFCC(mfcc_params()).compute(x.tolist())

    assert m.dtype == np.float32
    assert m.shape == (122, 13)
    assert np.array_equal(m, np.asarray(m_l, dtype=np.float32))
    assert np.isfinite(m).all()


def test_mfcc_object_is_reusable_across_calls():
    x = speech_like(16000, 16000)
    mfcc = libspeech.MFCC(mfcc_params())
    assert np.array_equal(mfcc.compute(x), mfcc.compute(x))


# --- Resample -----------------------------------------------------------

def test_resample_numpy_equals_list_api_and_tracks_the_true_signal():
    src, dst = 44100, 16000
    t = np.arange(src) / src
    x = (0.4 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

    y = libspeech.Resample(src, dst).resample(x)
    y_l = libspeech.Resample(src, dst).resample(x.tolist())

    assert y.dtype == np.float32
    assert y.shape == (dst,)
    assert np.array_equal(y, np.asarray(y_l, dtype=np.float32))

    t_out = np.arange(dst) / dst
    expected = 0.4 * np.sin(2 * np.pi * 300 * t_out)
    assert np.abs(y[1000:-1000] - expected[1000:-1000]).max() < 5e-3


def test_resample_numpy_identity_rate_returns_a_copy():
    x = speech_like(8000, 16000)
    y = libspeech.Resample(16000, 16000).resample(x)
    assert np.array_equal(x, y)
    y[0] = 123.0  # must not write through into the caller's array
    assert x[0] != 123.0


# --- Audio --------------------------------------------------------------

def test_audio_to_numpy_matches_data(tmp_path):
    samples = speech_like(8000, 16000)
    audio = libspeech.Audio()
    assert audio.load([samples.tolist()], 16000)
    wav = tmp_path / "a.wav"
    assert audio.save(str(wav))

    loaded = libspeech.Audio()
    assert loaded.load(str(wav))
    arr = loaded.to_numpy(0)

    assert arr.dtype == np.float32
    assert arr.shape == (len(loaded),)
    assert np.array_equal(arr, np.asarray(loaded.data(0), dtype=np.float32))
    assert np.abs(arr - samples).max() < 1e-3  # WAV round-trip quantization only


def test_audio_to_numpy_is_a_copy_and_validates_channel():
    audio = libspeech.Audio()
    assert audio.load([[0.1, 0.2, 0.3]], 16000)
    arr = audio.to_numpy()  # channel 0 by default
    arr[0] = 9.0
    assert audio.to_numpy(0)[0] == pytest.approx(0.1)  # caller's edit did not leak into Audio
    with pytest.raises(IndexError):
        audio.to_numpy(5)


def test_audio_len_does_not_depend_on_copying_data():
    audio = libspeech.Audio()
    assert audio.load([[0.0] * 1234], 16000)
    assert len(audio) == 1234
    assert math.isclose(audio.duration, 1234 / 16000)
