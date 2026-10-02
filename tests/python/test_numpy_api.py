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

    assert isinstance(re, np.ndarray) and re.dtype == np.float32
    assert re.shape == im.shape == (122, 512)
    assert np.array_equal(re, np.asarray(re_l, dtype=np.float32))
    assert np.array_equal(im, np.asarray(im_l, dtype=np.float32))


def test_stft_onesided_is_the_first_half_of_the_full_spectrum():
    x = speech_like(16000, 16000)
    stft = libspeech.STFT(9, libspeech.WindowType.hann, 128)
    assert stft.num_bins() == 512 and stft.num_bins(onesided=True) == 257

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
    assert re.shape[0] == 0 and im.shape[0] == 0


def test_numpy_result_outlives_the_producing_objects():
    x = speech_like(16000, 16000)

    def make():
        return libspeech.STFT(9).stft(x)[0]

    out = make()
    expected = out.copy()
    del x
    gc.collect()
    junk = [np.random.rand(10000) for _ in range(50)]  # churn the allocator
    gc.collect()
    assert np.array_equal(out, expected)
    del junk


# --- MFCC ---------------------------------------------------------------

def test_mfcc_numpy_equals_list_api():
    x = speech_like(16000, 16000)
    m = libspeech.MFCC(mfcc_params()).compute(x)
    m_l = libspeech.MFCC(mfcc_params()).compute(x.tolist())

    assert m.dtype == np.float32 and m.shape == (122, 13)
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

    assert y.dtype == np.float32 and y.shape == (dst,)
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

    assert arr.dtype == np.float32 and arr.shape == (len(loaded),)
    assert np.array_equal(arr, np.asarray(loaded.data(0), dtype=np.float32))
    assert np.abs(arr - samples).max() < 1e-3  # WAV round-trip quantization only


def test_audio_to_numpy_is_a_copy_and_validates_channel():
    audio = libspeech.Audio()
    assert audio.load([[0.1, 0.2, 0.3]], 16000)
    arr = audio.to_numpy()  # channel 0 by default
    arr[0] = 9.0
    assert audio.to_numpy(0)[0] == pytest.approx(0.1)  # caller's edit did not leak into Audio
    with pytest.raises(Exception):
        audio.to_numpy(5)


def test_audio_len_does_not_depend_on_copying_data():
    audio = libspeech.Audio()
    assert audio.load([[0.0] * 1234], 16000)
    assert len(audio) == 1234
    assert math.isclose(audio.duration, 1234 / 16000)
