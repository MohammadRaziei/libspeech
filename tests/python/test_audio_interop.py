"""Audio <-> array interop: Audio(array), np.asarray(audio), DLPack, and the shared planar buffer.

The samples live in ONE planar float32 buffer, channel c at [c * samples, (c + 1) * samples). NumPy and
DLPack views share it (no copy); every load builds a new buffer, so a view stays valid after a reload.
"""

from __future__ import annotations

import numpy as np
import pytest

import libspeech
from libspeech import Audio

RATE = 16000


def stereo(n=100):
    rng = np.random.default_rng(1)
    return rng.uniform(-1, 1, (2, n)).astype(np.float32)


def test_shape_and_channels():
    mono = Audio(np.zeros(50, dtype=np.float32), RATE)
    assert (mono.channels, mono.shape, len(mono)) == (1, (1, 50), 50)
    st = Audio(stereo(80), RATE)
    assert (st.channels, st.shape, len(st)) == (2, (2, 80), 80)


def test_audio_from_2d_array_roundtrips_exactly():
    x = stereo()
    a = Audio(x, RATE)
    assert a.sample_rate == RATE
    out = np.asarray(a)
    assert out.shape == (2, 100)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, x)
    np.testing.assert_array_equal(a.to_numpy(1), x[1])


def test_1d_array_is_mono_and_float64_is_converted():
    x = np.linspace(-1, 1, 64)  # float64
    a = Audio(x, RATE)
    assert a.shape == (1, 64)
    np.testing.assert_allclose(np.asarray(a)[0], x.astype(np.float32))


def test_non_contiguous_input_is_accepted():
    x = stereo(40)
    a = Audio(np.ascontiguousarray(x.T).T, RATE)  # a transposed view: not C-contiguous
    np.testing.assert_array_equal(np.asarray(a), x)


@pytest.mark.parametrize("bad", [np.zeros((2, 3, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)])
def test_bad_shapes_are_rejected(bad):
    with pytest.raises(ValueError):
        Audio(bad, RATE)


@pytest.mark.parametrize("rate", [0, -8000])
def test_bad_sample_rate_is_rejected(rate):
    with pytest.raises(ValueError):
        Audio(np.zeros(8, dtype=np.float32), rate)


def test_numpy_view_shares_memory_with_the_audio():
    a = Audio(stereo(), RATE)
    view = np.asarray(a)
    view[1, 7] = 0.5
    assert a.data(1)[7] == 0.5
    assert np.shares_memory(view, a.numpy())


def test_numpy_copy_and_dtype_do_not_share():
    a = Audio(stereo(), RATE)
    copy = a.numpy(copy=True)
    copy[0, 0] = 9.0
    assert a.data(0)[0] != 9.0
    wide = a.numpy(dtype=np.float64)
    assert wide.dtype == np.float64
    assert not np.shares_memory(wide, a.numpy())


def test_a_view_survives_a_reload_of_the_audio():
    x = stereo()
    a = Audio(x, RATE)
    view = np.asarray(a)
    assert a.load([[0.25] * 10], 8000)  # a new, shorter buffer
    assert a.shape == (1, 10)
    np.testing.assert_array_equal(view, x)  # the old samples, still valid
    np.testing.assert_array_equal(np.asarray(a), np.full((1, 10), 0.25, dtype=np.float32))


def test_a_copy_of_an_audio_does_not_share_samples():
    a = Audio(stereo(), RATE)
    b = Audio(a)
    np.asarray(a)[0, 0] = 7.0
    assert b.data(0)[0] != 7.0
    c = Audio()
    c = a
    assert c.shape == a.shape


def test_dlpack_export_is_zero_copy():
    a = Audio(stereo(), RATE)
    assert a.__dlpack_device__() == (1, 0)
    shared = np.from_dlpack(a)
    assert shared.shape == (2, 100)
    assert shared.dtype == np.float32
    assert np.shares_memory(shared, np.asarray(a))
    assert not shared.flags.writeable  # an unversioned DLPack capsule does not say it is writable
    np.asarray(a)[0, 3] = -0.75  # a write through the writable view shows in the DLPack one
    assert shared[0, 3] == -0.75


def test_dlpack_roundtrip_through_from_dlpack():
    x = stereo()
    a = Audio.from_dlpack(np.from_dlpack(Audio(x, RATE)), RATE)
    np.testing.assert_array_equal(np.asarray(a), x)
    b = Audio.from_numpy(x, RATE)
    np.testing.assert_array_equal(np.asarray(b), x)


def test_empty_audio_exports_nothing():
    a = Audio()
    assert np.asarray(a).shape == (0, 0)
    assert a.shape == (0, 0)
    with pytest.raises(ValueError):
        a.__dlpack__()


def test_reloading_a_shorter_file_into_the_same_audio_updates_the_length(tmp_path):
    long_path, short_path = str(tmp_path / "long.wav"), str(tmp_path / "short.wav")
    Audio(np.linspace(-1, 1, 16000, dtype=np.float32), RATE).save(long_path)
    Audio(np.linspace(-1, 1, 4000, dtype=np.float32), RATE).save(short_path)
    a = Audio()
    assert a.load(long_path)
    assert len(a) == 16000
    assert a.load(short_path)
    assert len(a) == 4000
    assert a.duration == pytest.approx(0.25)
    assert np.asarray(a).shape == (1, 4000)


def test_a_failed_load_keeps_the_old_audio(tmp_path):
    x = stereo()
    a = Audio(x, RATE)
    assert not a.load(str(tmp_path / "missing.wav"))
    np.testing.assert_array_equal(np.asarray(a), x)


def test_channels_of_different_lengths_are_rejected():
    a = Audio()
    assert not a.load([[0.0] * 10, [0.0] * 9], RATE)
    assert a.shape == (0, 0)


def test_raw_bin_on_a_fresh_audio_fails_cleanly(tmp_path):
    path = tmp_path / "raw.bin"
    path.write_bytes(np.zeros(32, dtype=np.float32).tobytes())
    assert Audio().load(str(path)) is False  # used to divide by zero


def test_resample_and_to_mono_still_work_on_the_shared_buffer():
    a = Audio(stereo(1600), RATE)
    mono = a.to_mono()
    np.testing.assert_allclose(np.asarray(mono)[0], stereo(1600).mean(axis=0), atol=1e-6)
    assert mono.channels == 1
    assert a.channels == 2  # the original is untouched
    assert a.resample(8000).sample_rate == 8000
    assert libspeech.Audio is Audio
