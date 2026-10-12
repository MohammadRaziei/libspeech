"""The everyday API: libspeech.stft / libspeech.istft, STFT(n_fft=...), method strings.

numpy is a hard requirement of this module; everything is compared against the STFT class (which is
tested against librosa in test_librosa_compat.py) or against numpy itself.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

import libspeech  # noqa: E402


def _signal(n: int, seed: int = 0) -> np.ndarray:
    return (0.3 * np.random.default_rng(seed).standard_normal(n)).astype(np.float32)


# --- stft ------------------------------------------------------------------

def test_stft_shape_dtype_and_layout_follow_librosa():
    y = _signal(16000)
    z = libspeech.stft(y, n_fft=512, hop_length=128)
    assert z.dtype == np.complex64
    assert z.shape == (257, 1 + 16000 // 128)  # (bins, frames), centred
    assert z.flags.f_contiguous  # a transposed view of the kernel's (frames, bins) array: no copy


def test_stft_defaults_are_librosas():
    z = libspeech.stft(_signal(8192))
    assert z.shape == (1025, 1 + 8192 // 512)  # n_fft=2048, hop_length=n_fft // 4


def test_stft_equals_the_class_on_the_padded_signal():
    y = _signal(5000, seed=1)
    ref = libspeech.STFT(n_fft=256, hop_length=64).stft_complex(np.pad(y, 128), onesided=True).T
    assert np.array_equal(libspeech.stft(y, n_fft=256, hop_length=64), ref)


def test_stft_without_centering_analyses_the_signal_as_given():
    y = _signal(4096, seed=2)
    z = libspeech.stft(y, n_fft=512, hop_length=128, center=False)
    assert z.shape == (257, (4096 - 512) // 128 + 1)
    ref = libspeech.STFT(n_fft=512, hop_length=128).stft_complex(y, onesided=True).T
    assert np.array_equal(z, ref)


def test_stft_onesided_false_gives_all_bins_with_a_conjugate_mirror():
    z = libspeech.stft(_signal(4096, seed=3), n_fft=512, hop_length=128, onesided=False)
    assert z.shape[0] == 512
    assert np.array_equal(z[1:256][::-1], np.conj(z[257:512]))


def test_stft_pad_mode_is_numpys():
    y = _signal(3000, seed=4)
    ref = libspeech.STFT(n_fft=256).stft_complex(np.pad(y, 128, mode="reflect"), onesided=True).T
    assert np.array_equal(libspeech.stft(y, n_fft=256, pad_mode="reflect"), ref)
    assert not np.array_equal(libspeech.stft(y, n_fft=256, pad_mode="reflect"), libspeech.stft(y, n_fft=256))


def test_stft_window_by_name_or_enum():
    y = _signal(4096, seed=5)
    hann = libspeech.stft(y, n_fft=512)
    assert np.array_equal(hann, libspeech.stft(y, n_fft=512, window=libspeech.WindowType.hann))
    assert np.array_equal(hann, libspeech.stft(y, n_fft=512, window="HANN"))
    assert not np.array_equal(hann, libspeech.stft(y, n_fft=512, window="hamming"))
    with pytest.raises(ValueError, match="unknown window"):
        libspeech.stft(y, n_fft=512, window="triangle-ish")


def test_stft_accepts_lists_and_float64():
    y = _signal(2048, seed=6)
    ref = libspeech.stft(y, n_fft=256)
    assert np.array_equal(libspeech.stft(y.astype(np.float64), n_fft=256), ref)
    assert np.array_equal(libspeech.stft(y.tolist(), n_fft=256), ref)


def test_stft_validates_its_arguments():
    y = _signal(2048)
    with pytest.raises(ValueError, match="power of two"):
        libspeech.stft(y, n_fft=500)
    with pytest.raises(ValueError, match="hop_length"):
        libspeech.stft(y, n_fft=256, hop_length=0)
    with pytest.raises(ValueError, match="1-D"):
        libspeech.stft(np.zeros((2, 2048), np.float32), n_fft=256)
    with pytest.raises(ValueError, match="fewer than n_fft"):
        libspeech.stft(_signal(100), n_fft=256, center=False)
    with pytest.raises(TypeError):
        libspeech.stft(y, 256)  # n_fft is keyword-only, like librosa


# --- istft -----------------------------------------------------------------

@pytest.mark.parametrize("method", ["weighted", "plain"])
@pytest.mark.parametrize("onesided", [True, False])
def test_istft_round_trips_the_whole_signal(method, onesided):
    y = _signal(8000, seed=7)
    z = libspeech.stft(y, n_fft=512, hop_length=128, onesided=onesided)
    back = libspeech.istft(z, hop_length=128, length=len(y), method=method)
    assert back.dtype == np.float32
    assert back.shape == y.shape
    assert np.abs(back - y).max() < 5e-4


def test_istft_infers_n_fft_and_uses_center_and_length():
    y = _signal(4096, seed=8)
    z = libspeech.stft(y, n_fft=256, hop_length=64)
    full = libspeech.istft(z, hop_length=64, center=False)  # n_fft inferred as 2 * (bins - 1)
    centred = libspeech.istft(z, hop_length=64)  # trims n_fft // 2 on each side
    assert len(centred) == len(full) - 256
    assert np.array_equal(centred, full[128:-128])
    assert len(libspeech.istft(z, hop_length=64, length=5000)) == 5000  # zero-padded past the end
    assert len(libspeech.istft(z, hop_length=64, length=100)) == 100


def test_istft_accepts_any_memory_layout_and_dtype():
    z = libspeech.stft(_signal(4096, seed=9), n_fft=512, hop_length=128)
    ref = libspeech.istft(z, hop_length=128)
    assert np.array_equal(libspeech.istft(np.ascontiguousarray(z), hop_length=128), ref)  # (bins, frames), C order
    assert np.array_equal(libspeech.istft(z.astype(np.complex128), hop_length=128), ref)


def test_istft_validates_its_arguments():
    z = libspeech.stft(_signal(2048), n_fft=256)
    with pytest.raises(ValueError, match="complex"):
        libspeech.istft(np.abs(z))
    with pytest.raises(ValueError, match="complex"):
        libspeech.istft(z[0])  # 1-D
    with pytest.raises(ValueError, match="bins"):
        libspeech.istft(z[:100], n_fft=256)
    with pytest.raises(ValueError, match="method"):
        libspeech.istft(z, method="average")


# --- the STFT class: n_fft constructor and method strings --------------------------------------

def test_stft_class_takes_n_fft_in_samples_or_the_exponent():
    by_exp = libspeech.STFT(9, libspeech.WindowType.hann, 128)
    by_len = libspeech.STFT(n_fft=512, hop_length=128)
    assert (by_len.fft_length, by_len.slide_length) == (by_exp.fft_length, by_exp.slide_length) == (512, 128)
    assert libspeech.STFT(n_fft=512).slide_length == 128  # default hop: a quarter frame
    with pytest.raises(ValueError, match="power of two"):
        libspeech.STFT(n_fft=500)
    with pytest.raises(ValueError, match="EXPONENT"):  # the classic mistake: a length where an exponent goes
        libspeech.STFT(512)


def test_istft_method_accepts_names_and_numbers():
    stft = libspeech.STFT(n_fft=256)
    z = stft.stft_complex(_signal(4096, seed=10), onesided=True)
    assert np.array_equal(stft.istft(z, "weighted"), stft.istft(z, 0))
    assert np.array_equal(stft.istft(z, "Plain"), stft.istft(z, 1))
    assert np.array_equal(stft.istft(z), stft.istft(z, "weighted"))
    with pytest.raises(ValueError, match="weighted"):
        stft.istft(z, "mean")
    with pytest.raises(ValueError):
        stft.istft(z, 2)


# --- thread safety ----------------------------------------------------------------------------

@pytest.mark.parametrize("n_fft", [256, 8])  # 8 samples is outside the fast path: the AudioFlux fallback
def test_one_stft_object_can_be_shared_between_threads(n_fft):
    stft = libspeech.STFT(n_fft=n_fft)
    y = _signal(4000, seed=11)
    expected = stft.stft_complex(y, onesided=True)
    results, errors = [], []

    def work():
        try:
            for _ in range(30):
                results.append(np.array_equal(stft.stft_complex(y, onesided=True), expected))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=work) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert results
    assert all(results)


# --- typing -----------------------------------------------------------------------------------

def test_the_package_is_marked_typed_and_ships_stubs():
    package = Path(libspeech.__file__).parent
    assert (package / "py.typed").exists()
    if not (package / "speech_dsp_py.pyi").exists():
        pytest.skip("stub generation is best-effort (it is skipped, with a warning, where modules cannot be imported)")
    assert "def istft" in (package / "speech_dsp_py.pyi").read_text()
