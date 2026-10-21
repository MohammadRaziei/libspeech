"""Streaming denoiser (Denoiser.stream() -> DenoiserStream) with the real DPDFNet models.

Marked `models`: runs only with `pytest -m models` (CMake target speech_test_python_models_e2e); the 10 MB models are
downloaded on first use. The core property is the same as for GTCRN: feeding the audio in chunks of any size gives
exactly the audio process() returns for the whole signal, tail included. DPDFNet has one more thing to get right: the
network answers four hops late, and process() compensates for it, so a stream has to as well.
"""

from __future__ import annotations

import numpy as np
import pytest

import libspeech

pytestmark = pytest.mark.models

SR = 16000
# dpdfnet2 (16 kHz): n_fft 320, hop 160. The STFT needs n_fft - 1 samples of look-ahead, and the network answers
# four hops (640 samples) late, which process() compensates for: 319 + 640.
LATENCY = 959


def _noisy_tone(seconds: float, seed: int, sr: int = SR) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    voice = sum(np.sin(2 * np.pi * f * t) / k for k, f in enumerate([180, 360, 540, 720], start=1))
    voice *= 0.5 + 0.5 * np.sin(2 * np.pi * 2.5 * t) ** 2
    return (0.15 * voice + 0.05 * rng.standard_normal(t.size)).astype(np.float32)


@pytest.fixture(scope="module")
def den():
    return libspeech.Denoiser("dpdfnet")


@pytest.fixture(scope="module")
def audio() -> np.ndarray:
    return _noisy_tone(2.5, seed=1)


@pytest.fixture(scope="module")
def reference(den, audio) -> np.ndarray:
    return den.process(audio)


def _stream_all(den, x, chunk_sizes):
    stream = den.stream()
    pieces, i, k = [], 0, 0
    while i < len(x):
        n = chunk_sizes[k % len(chunk_sizes)]
        pieces.append(stream.feed(x[i : i + n]))
        i += n
        k += 1
    pieces.append(stream.finish())
    return np.concatenate(pieces), stream


def test_dpdfnet_supports_streaming(den):
    assert den.supports_streaming


@pytest.mark.parametrize("chunks", [[1], [100], [159], [160], [161], [320], [1000], [4000], [7, 600, 33, 160, 5], [10**6]])
def test_stream_equals_process_for_any_chunking(den, audio, reference, chunks):
    out, stream = _stream_all(den, audio, chunks)
    assert out.dtype == np.float32
    assert out.shape == reference.shape
    np.testing.assert_array_equal(out, reference)  # same frames, same math: identical, not just close
    assert stream.samples_in == stream.samples_out == len(audio)


@pytest.mark.parametrize("length", [1, 100, 159, 160, 161, 319, 320, 500, 959, 960, 1500])
def test_short_streams_equal_process(den, length):
    x = _noisy_tone(1.0, seed=2)[:length]
    out, _ = _stream_all(den, x, [64])
    np.testing.assert_array_equal(out, den.process(x))


def test_latency_is_bounded_and_the_bound_is_reached(den, audio):
    stream = den.stream()
    assert stream.latency_samples == LATENCY
    worst = 0
    for i in range(len(audio)):
        stream.feed(audio[i : i + 1])
        worst = max(worst, stream.samples_in - stream.samples_out)
        assert stream.samples_in - stream.samples_out <= LATENCY
    assert worst == LATENCY
    stream.finish()


def test_the_tail_of_the_input_is_real_output_not_silence(den):
    # The last 4 hops only exist because finish() flushes the network with silence; they must carry the signal at
    # the level of the audio just before them (this synthetic input is mostly removed by the model, so the level is
    # low in absolute terms, but a lost tail would be exactly 0).
    x = _noisy_tone(1.0, seed=7)
    out, _ = _stream_all(den, x, [500])

    def rms(a):
        return float(np.sqrt(np.mean(a.astype(np.float64) ** 2)))

    assert rms(out[-640:]) > 0.25 * rms(out[-1280:-640]) > 0.0


def test_two_streams_are_independent_even_when_interleaved(den):
    a, b = _noisy_tone(1.5, seed=3), _noisy_tone(1.5, seed=4)
    sa, sb = den.stream(), den.stream()
    out_a, out_b = [], []
    for i in range(0, len(a), 700):
        out_a.append(sa.feed(a[i : i + 700]))
        out_b.append(sb.feed(b[i : i + 700]))
    out_a.append(sa.finish())
    out_b.append(sb.finish())
    np.testing.assert_array_equal(np.concatenate(out_a), den.process(a))
    np.testing.assert_array_equal(np.concatenate(out_b), den.process(b))


def test_new_stream_starts_with_fresh_state(den, audio, reference):
    _stream_all(den, _noisy_tone(1.0, seed=5), [500])  # leave the model's recurrent state "used"
    out, _ = _stream_all(den, audio, [500])
    np.testing.assert_array_equal(out, reference)


def test_feed_after_finish_raises_and_finish_is_idempotent(den, audio):
    stream = den.stream()
    stream.feed(audio[:2000])
    tail = stream.finish()
    assert stream.finished
    assert len(stream.finish()) == 0
    assert stream.samples_out == 2000
    assert len(tail) > 0
    with pytest.raises(RuntimeError):
        stream.feed(audio[:10])


def test_empty_feed_and_list_input(den, audio, reference):
    stream = den.stream()
    assert len(stream.feed(np.zeros(0, dtype=np.float32))) == 0
    assert stream.samples_in == 0
    pieces = [stream.feed(audio[:3000].tolist()), stream.feed(audio[3000:]), stream.finish()]
    np.testing.assert_array_equal(np.concatenate(pieces), reference)


def test_stream_outlives_denoiser_close():
    local = libspeech.Denoiser("dpdfnet")
    x = _noisy_tone(1.0, seed=6)
    expected = local.process(x)
    stream = local.stream()
    first = stream.feed(x[:8000])
    local.close()
    with pytest.raises(RuntimeError):
        local.stream()  # a closed denoiser starts no new sessions
    out = np.concatenate([first, stream.feed(x[8000:]), stream.finish()])  # the running one carries on
    np.testing.assert_array_equal(out, expected)


def test_the_48khz_model_streams_too():
    den48 = libspeech.Denoiser("dpdfnet", "dpdfnet2_48khz_hr", sample_rate=48000)
    assert den48.supports_streaming
    x = _noisy_tone(1.5, seed=8, sr=48000)
    out, stream = _stream_all(den48, x, [3000, 777])
    np.testing.assert_array_equal(out, den48.process(x))
    assert stream.latency_samples > LATENCY  # a longer window at 48 kHz
