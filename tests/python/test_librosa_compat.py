"""Cross-checks of libspeech against librosa (and soundfile / scipy, which librosa builds on).

Covers audio I/O (load for every container/bit depth, save, to_mono, resample) and
DSP (window, STFT, MFCC, array resampling). Tolerances were set from measured
differences and are noted next to each assertion; where libspeech deliberately
differs from librosa (mel filterbank construction, triangular windows, MP3 decoder
delay) the test says so and pins the difference instead of hiding it.

Needs the `test` extra: pip install "libspeech[test]"  (librosa, soundfile, scipy, numpy).
The whole module is skipped when any of them cannot be imported.
"""

from __future__ import annotations

import importlib
import warnings

import pytest

import libspeech


def _optional(name: str):
    """Import an optional test dependency, or skip this whole module.

    Catches more than ImportError on purpose: a half-installed numba/llvmlite (librosa's
    dependencies) fails with OSError/RuntimeError rather than ModuleNotFoundError, and a
    DeprecationWarning at import time is an error under this project's pytest settings.
    Neither is a libspeech problem, so neither may fail the suite.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return importlib.import_module(name)
    except Exception as exc:
        pytest.skip(f"optional test dependency {name!r} is unavailable: {exc!r}", allow_module_level=True)


np = _optional("numpy")
librosa = _optional("librosa")
sf = _optional("soundfile")
scipy_fft = _optional("scipy.fft")

# These tests judge libspeech's output. Warnings raised inside librosa / numpy / scipy (which
# differ between their releases, and the project turns warnings into errors) must not fail them.
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning", "ignore::FutureWarning", "ignore::UserWarning")


# --- helpers -----------------------------------------------------------------


def tones(freqs, sample_rate: int, seconds: float = 1.0, amp: float = 0.3) -> np.ndarray:
    """Band-limited test signal: a sum of sines at `freqs` (Hz) with distinct phases."""
    t = np.arange(int(sample_rate * seconds)) / sample_rate
    x = sum(amp / len(freqs) * np.sin(2 * np.pi * f * t + 0.7 * i) for i, f in enumerate(freqs))
    return np.asarray(x, dtype=np.float32)


def stereo(sample_rate: int, seconds: float = 1.0) -> np.ndarray:
    """(channels, samples) float32."""
    return np.stack([tones([300, 1700], sample_rate, seconds), tones([440, 2300], sample_rate, seconds)])


def noisy_speechlike(seconds: float = 3.0, sample_rate: int = 16000) -> np.ndarray:
    rng = np.random.default_rng(0)
    x = tones([220, 1900], sample_rate, seconds, amp=0.6)
    return (x + 0.01 * rng.standard_normal(x.size)).astype(np.float32)


def libspeech_channels(audio: libspeech.Audio) -> np.ndarray:
    """(channels, samples) float32 view of a libspeech.Audio."""
    return np.stack([audio.to_numpy(c) for c in range(len(audio.data()))])


def load_with_libspeech(path) -> libspeech.Audio:
    audio = libspeech.Audio()
    assert audio.load(str(path)), f"libspeech failed to load {path}"
    return audio


# --- load ---------------------------------------------------------------------


@pytest.mark.parametrize("subtype", ["PCM_16", "PCM_24", "PCM_32", "FLOAT"])
@pytest.mark.parametrize(("sample_rate", "channels"), [(16000, 1), (44100, 2)])
def test_wav_load_matches_librosa(tmp_path, subtype, sample_rate, channels):
    data = stereo(sample_rate) if channels == 2 else stereo(sample_rate)[:1]
    path = tmp_path / f"a_{subtype}.wav"
    sf.write(path, data.T, sample_rate, subtype=subtype)

    audio = load_with_libspeech(path)
    ref, ref_sr = librosa.load(path, sr=None, mono=False)
    ref = np.atleast_2d(ref)

    assert audio.sample_rate == ref_sr == sample_rate
    got = libspeech_channels(audio)
    assert got.shape == ref.shape
    # Measured: bit-identical for every bit depth.
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-7)


def test_flac_load_matches_librosa(tmp_path):
    data = stereo(44100)
    path = tmp_path / "a.flac"
    sf.write(path, data.T, 44100, subtype="PCM_16")

    audio = load_with_libspeech(path)
    ref, ref_sr = librosa.load(path, sr=None, mono=False)

    assert audio.sample_rate == ref_sr
    np.testing.assert_allclose(libspeech_channels(audio), ref, rtol=0, atol=1e-7)


def test_mp3_load_matches_librosa_after_decoder_delay(tmp_path):
    """MP3 decoders legitimately differ in gapless handling: libsndfile (librosa) trims the
    encoder delay and padding, miniaudio (libspeech) returns the raw decoded stream. Once the
    leading delay is aligned away, the two decoders agree to float precision."""
    if "MP3" not in sf.available_formats():
        pytest.skip("this libsndfile build cannot write MP3")
    data = stereo(44100)
    path = tmp_path / "a.mp3"
    sf.write(path, data.T, 44100, format="MP3")

    audio = load_with_libspeech(path)
    ref, ref_sr = librosa.load(path, sr=None, mono=False)
    got = libspeech_channels(audio)

    assert audio.sample_rate == ref_sr
    assert got.shape[0] == ref.shape[0]
    extra = got.shape[1] - ref.shape[1]
    assert 0 < extra < 6000, "expected libspeech to keep the decoder delay/padding samples"

    # Find the delay by cross-correlating channel 0, then compare the interior.
    n = ref.shape[1]
    corr = np.correlate(got[0, : n + 4608], ref[0], mode="valid")
    lag = int(np.argmax(corr))
    assert 0 < lag < 4608
    m = n - 4096
    np.testing.assert_allclose(got[:, lag + 2048 : lag + 2048 + m], ref[:, 2048 : 2048 + m], rtol=0, atol=1e-3)


def test_load_reports_failure_for_a_missing_file(tmp_path):
    assert libspeech.Audio().load(str(tmp_path / "does_not_exist.wav")) is False


# --- to_mono / resample (Audio) ---------------------------------------------------


def test_to_mono_matches_librosa(tmp_path):
    data = stereo(44100)
    path = tmp_path / "s.wav"
    sf.write(path, data.T, 44100, subtype="FLOAT")

    mono = load_with_libspeech(path).to_mono()  # returns a new Audio
    assert len(mono.data()) == 1
    np.testing.assert_allclose(mono.to_numpy(0), librosa.to_mono(data), rtol=0, atol=1e-7)


@pytest.mark.parametrize(
    ("target", "res_type", "atol"),
    [
        (22050, "soxr_hq", 1e-6),  # measured 9e-8
        (48000, "soxr_hq", 1e-6),  # measured 1.5e-7
        (16000, "soxr_hq", 1e-6),  # measured 1.4e-7 (libspeech's windowed-sinc bank is closed-form)
    ],
)
def test_audio_resample_matches_librosa(tmp_path, target, res_type, atol):
    data = stereo(44100)
    path = tmp_path / "s.wav"
    sf.write(path, data.T, 44100, subtype="FLOAT")

    out = load_with_libspeech(path).resample(target)
    ref = librosa.resample(data, orig_sr=44100, target_sr=target, res_type=res_type)
    got = libspeech_channels(out)

    assert out.sample_rate == target
    assert got.shape[0] == ref.shape[0]
    assert abs(got.shape[1] - ref.shape[1]) <= 1  # librosa rounds the length up, libspeech down
    n = min(got.shape[1], ref.shape[1])
    edge = int(0.02 * target)  # filter start-up/tail differ; compare the interior
    np.testing.assert_allclose(got[:, edge : n - edge], ref[:, edge : n - edge], rtol=0, atol=atol)


# --- save ---------------------------------------------------------------------------


@pytest.mark.parametrize(("sample_rate", "channels"), [(16000, 1), (44100, 2), (8000, 2)])
def test_saved_wav_is_read_back_identically_by_librosa(tmp_path, sample_rate, channels):
    data = stereo(sample_rate) if channels == 2 else stereo(sample_rate)[:1]
    audio = libspeech.Audio()
    assert audio.load([c.tolist() for c in data], sample_rate)
    path = tmp_path / "saved.wav"
    assert audio.save(str(path))

    info = sf.info(path)
    assert (info.samplerate, info.channels, info.frames) == (sample_rate, channels, data.shape[1])

    ref, ref_sr = librosa.load(path, sr=None, mono=False)
    assert ref_sr == sample_rate
    # The WAV is written as 32-bit float, so the round trip is exact.
    np.testing.assert_allclose(np.atleast_2d(ref), data, rtol=0, atol=1e-7)


def test_save_then_load_roundtrip_inside_libspeech(tmp_path):
    data = stereo(22050)
    audio = libspeech.Audio()
    assert audio.load([c.tolist() for c in data], 22050)
    path = tmp_path / "rt.wav"
    assert audio.save(str(path))

    again = load_with_libspeech(path)
    assert again.sample_rate == 22050
    assert len(again) == data.shape[1]
    np.testing.assert_allclose(libspeech_channels(again), data, rtol=0, atol=1e-7)


# --- window -------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "librosa_name", "atol"), [
        ("hann", "hann", 1e-6),
        ("hamming", "hamming", 1e-6),
        ("blackman", "blackman", 1e-6),
        ("flattop", "flattop", 1e-6),
        ("rect", "boxcar", 0.0),
        # libspeech's triangular windows use a slightly different end-point convention
        # than scipy's periodic ones (measured 3.9e-3 / 5.8e-3 for N=512): pinned, not hidden.
        ("bartlett", "bartlett", 1e-2),
        ("triang", "triang", 1e-2),
    ],
)
def test_window_matches_librosa(name, librosa_name, atol):
    n = 512
    got = np.asarray(libspeech.window(getattr(libspeech.WindowType, name), n))
    ref = librosa.filters.get_window(librosa_name, n, fftbins=True)  # periodic, like an FFT window
    assert got.shape == ref.shape
    np.testing.assert_allclose(got, ref, rtol=0, atol=atol)


# --- DCT ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "k"),
    [(26, 13), (64, 64), (100, 40), (512, 13), (512, 512), (1000, 1000), (4096, 4096), (8192, 100), (16384, 16384)],
)
@pytest.mark.parametrize("orthonormal", [True, False])
def test_dct_matches_scipy(n, k, orthonormal):
    rng = np.random.default_rng(n + k)
    x = rng.standard_normal(n).astype(np.float32)
    got = np.asarray(libspeech.dct(x.tolist(), k, orthonormal))
    ref = scipy_fft.dct(x.astype(np.float64), type=2, norm="ortho" if orthonormal else None)[:k]
    assert got.shape == ref.shape
    # float32 FFT/accumulation: relative to the largest output
    assert np.abs(got - ref).max() <= 2e-5 * max(1.0, np.abs(ref).max())


# --- STFT ---------------------------------------------------------------------------


@pytest.mark.parametrize(("radix2_exp", "hop"), [(8, 64), (9, 128), (10, 256), (11, 512)])
def test_stft_matches_librosa(radix2_exp, hop):
    x = noisy_speechlike()
    n_fft = 1 << radix2_exp
    stft = libspeech.STFT(radix2_exp, libspeech.WindowType.hann, hop)

    re, im = stft.stft(x, onesided=True)
    ref = librosa.stft(x, n_fft=n_fft, hop_length=hop, window="hann", center=False)  # (bins, frames)
    got = (re + 1j * im).T

    assert got.shape == ref.shape
    # Measured relative to the largest bin: ~2e-7 (float32 round-off).
    assert np.abs(got - ref).max() <= 1e-5 * np.abs(ref).max()


def test_full_stft_is_librosas_spectrum_plus_its_conjugate_mirror():
    x = noisy_speechlike(seconds=1.0)
    stft = libspeech.STFT(9, libspeech.WindowType.hann, 128)
    re, im = stft.stft(x)  # all 512 bins
    ref = librosa.stft(x, n_fft=512, hop_length=128, window="hann", center=False)
    full = (re + 1j * im).T

    assert full.shape == (512, ref.shape[1])
    assert np.abs(full[:257] - ref).max() <= 1e-5 * np.abs(ref).max()
    assert np.abs(full[257:] - np.conj(ref[1:256][::-1])).max() <= 1e-5 * np.abs(ref).max()


@pytest.mark.parametrize(("radix2_exp", "hop"), [(9, 128), (10, 256)])
def test_istft_matches_librosa(radix2_exp, hop):
    x = noisy_speechlike()
    n_fft = 1 << radix2_exp
    stft = libspeech.STFT(radix2_exp, libspeech.WindowType.hann, hop)
    re, im = stft.stft(x, onesided=True)

    ours = stft.istft(re, im)  # weighted overlap-add, the default
    ref = librosa.istft((re + 1j * im).T, hop_length=hop, win_length=n_fft, n_fft=n_fft, window="hann", center=False)

    assert ours.shape == ref.shape
    # Compared away from the first/last n_fft samples: there the window sum-of-squares is ~0 and
    # the two libraries normalize differently (librosa divides by almost nothing, libspeech
    # leaves samples below a 1e-6 floor alone) -- a boundary effect, not a difference in the maths.
    inner = slice(n_fft, -n_fft)
    assert np.abs(ours[inner] - ref[inner]).max() <= 2e-5 * max(1.0, np.abs(ref[inner]).max())


# --- MFCC ---------------------------------------------------------------------------


def _mfcc_params(sample_rate: int) -> libspeech.MFCCParams:
    p = libspeech.MFCCParams()
    p.sample_rate = sample_rate
    p.num_mel_filters = 26
    p.num_coefficients = 13
    p.radix2_exp = 9
    p.slide_length = 128
    return p


def _libspeech_mel_filterbank(sample_rate: int, n_fft: int, n_mels: int) -> np.ndarray:
    """libspeech's documented mel filterbank: HTK mel scale, breakpoints rounded DOWN to an
    integer FFT bin, unit-height triangles over those integer bins."""

    def hz2mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel2hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mels = np.linspace(hz2mel(0.0), hz2mel(sample_rate / 2.0), n_mels + 2)
    pts = np.clip(np.floor((n_fft + 1) * mel2hz(mels) / sample_rate).astype(int), 0, n_fft // 2)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(n_mels):
        lo, mid, hi = pts[m], pts[m + 1], pts[m + 2]
        fb[m, lo:mid] = (np.arange(lo, mid) - lo) / max(mid - lo, 1)
        fb[m, mid:hi] = (hi - np.arange(mid, hi)) / max(hi - mid, 1)
    return fb


def test_mfcc_equals_the_documented_recipe_applied_to_librosas_stft():
    x = noisy_speechlike()
    got = libspeech.MFCC(_mfcc_params(16000)).compute(x)

    power = np.abs(librosa.stft(x, n_fft=512, hop_length=128, window="hann", center=False)) ** 2
    fb = _libspeech_mel_filterbank(16000, 512, 26)
    ref = scipy_fft.dct(np.log(fb @ power + 1e-10), axis=0, type=2, norm="ortho")[:13].T

    assert got.shape == ref.shape == (372, 13)
    # Measured 1.1e-5 on values up to ~9.4: libspeech's MFCC is exactly this recipe.
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-3)


def test_mfcc_tracks_librosas_own_mfcc_pipeline():
    """librosa's mel filterbank uses exact fractional triangle edges; libspeech rounds the edges
    to FFT bins. The two MFCCs therefore are NOT numerically equal (mean |diff| ~0.7 on values up
    to ~9) -- but every coefficient follows the same trajectory over time."""
    x = noisy_speechlike()
    got = libspeech.MFCC(_mfcc_params(16000)).compute(x)

    mel_power = librosa.feature.melspectrogram(
        y=x, sr=16000, n_fft=512, hop_length=128, center=False, power=2.0,
        n_mels=26, htk=True, norm=None, fmin=0.0, fmax=8000.0, window="hann",
    )
    ref = scipy_fft.dct(np.log(mel_power + 1e-10), axis=0, type=2, norm="ortho")[:13].T

    assert got.shape == ref.shape
    for c in range(13):
        corr = np.corrcoef(got[:, c], ref[:, c])[0, 1]
        assert corr > 0.95, f"coefficient {c}: correlation {corr:.3f}"  # measured 0.967-0.988


# --- array resampling -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "target", "atol"),
    [
        # In-band tones agree with soxr_hq to float rounding for every ratio: both are within
        # ~2e-7 of the true signal (libspeech designs its polyphase filter in closed form).
        (44100, 22050, 1e-6),
        (44100, 48000, 1e-6),
        (48000, 44100, 1e-6),
        (44100, 16000, 1e-6),
        (48000, 16000, 1e-6),
    ],
)
def test_resample_array_matches_librosa(source, target, atol):
    x = tones([300, 1200, 2500], source, seconds=1.0)
    got = libspeech.Resample(source, target).resample(x)
    ref = librosa.resample(x, orig_sr=source, target_sr=target, res_type="soxr_hq")

    assert abs(got.shape[0] - ref.shape[0]) <= 1
    n = min(got.shape[0], ref.shape[0])
    edge = int(0.02 * target)
    np.testing.assert_allclose(got[edge : n - edge], ref[edge : n - edge], rtol=0, atol=atol)
