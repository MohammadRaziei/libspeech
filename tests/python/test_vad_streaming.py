"""Streaming SileroVad (feed / finish / pop_speech_timestamps).

These tests need the real Silero model (downloaded on first use), so they are marked `models` and only run with
`pytest -m models` (CMake target speech_test_python_models_e2e). The audio is synthetic speech-like sound (a
pulse train through formant filters, with pauses), generated here so the repository needs no audio fixture.

The core property: splitting the audio into chunks never changes the result. process() is reset() + feed(all) +
finish(), so every chunking must give exactly the same segments as the batch call.
"""

from __future__ import annotations

import numpy as np
import pytest

import libspeech

pytestmark = pytest.mark.models

SR = 16000


def _speech_like_audio() -> np.ndarray:
    scipy_signal = pytest.importorskip("scipy.signal")
    rng = np.random.default_rng(7)

    def vowel(dur, f0, formants):
        n = int(dur * SR)
        t = np.arange(n) / SR
        f = f0 * (1 + 0.08 * np.sin(2 * np.pi * 3 * t))
        phase = np.cumsum(f) / SR
        src = np.where((phase % 1.0) < 0.08, 1.0, 0.0) - 0.08
        y = np.diff(np.r_[0, src]) + 0.02 * rng.standard_normal(n)
        for fc, bw in formants:
            r = np.exp(-np.pi * bw / SR)
            th = 2 * np.pi * fc / SR
            y = scipy_signal.lfilter([1 - r], [1, -2 * r * np.cos(th), r * r], y)
        env = np.minimum(1, np.minimum(t / 0.03, (dur - t) / 0.04)) * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t) ** 2)
        y = y * env
        return (0.5 * y / np.abs(y).max()).astype(np.float32)

    def silence(d, amp=0.002):
        return (amp * rng.standard_normal(int(d * SR))).astype(np.float32)

    parts = [silence(0.6)]
    for i, fm in enumerate([[(700, 110), (1200, 120), (2600, 160)], [(300, 80), (2300, 150), (3000, 200)],
                            [(500, 100), (1500, 120), (2500, 150)]]):
        parts += [vowel(0.35, 130 + 10 * i, fm), silence(0.04)]
    parts.append(silence(0.9))
    for fm in [[(800, 110), (1100, 120), (2500, 160)], [(350, 80), (2000, 150), (2800, 200)]]:
        parts += [vowel(0.4, 140, fm), silence(0.03)]
    parts.append(silence(0.7))
    return np.concatenate(parts).astype(np.float32)


@pytest.fixture(scope="module")
def audio() -> np.ndarray:
    return _speech_like_audio()


@pytest.fixture(scope="module")
def vad():
    return libspeech.SileroVad(model_path="silero-vad.onnx")


def _pairs(timestamps):
    return [(t.start, t.end) for t in timestamps]


def _batch(vad, audio):
    vad.reset()
    vad.process(audio)
    return _pairs(vad.get_speech_timestamps())


def _stream(vad, audio, chunk_sizes):
    """Feeds `audio` in chunks cycling through `chunk_sizes`; returns (all popped segments, per-feed pops)."""
    vad.reset()
    popped, i, k = [], 0, 0
    while i < len(audio):
        n = chunk_sizes[k % len(chunk_sizes)]
        vad.feed(audio[i:i + n])
        popped += _pairs(vad.pop_speech_timestamps())
        i += n
        k += 1
    vad.finish()
    popped += _pairs(vad.pop_speech_timestamps())
    return popped


def test_synthetic_audio_is_detected_as_speech(vad, audio):
    # Sanity for the fixture itself: without it the equivalence tests below would compare empty lists.
    segments = _batch(vad, audio)
    assert len(segments) >= 2
    assert all(0 <= s < e <= len(audio) for s, e in segments)


@pytest.mark.parametrize("chunks", [[1], [100], [511], [512], [513], [1600], [4000], [7, 1000, 33, 512, 5]])
def test_streaming_equals_batch_for_any_chunking(vad, audio, chunks):
    expected = _batch(vad, audio)
    assert _stream(vad, audio, chunks) == expected


def test_pop_returns_each_segment_once_and_in_order(vad, audio):
    expected = _batch(vad, audio)
    vad.reset()
    seen = []
    for i in range(0, len(audio), 800):
        vad.feed(audio[i:i + 800])
        batch = _pairs(vad.pop_speech_timestamps())
        assert not (set(batch) & set(seen))  # never reported twice
        seen += batch
        assert _pairs(vad.pop_speech_timestamps()) == []  # nothing new until more audio arrives
    vad.finish()
    seen += _pairs(vad.pop_speech_timestamps())
    assert seen == expected
    assert _pairs(vad.get_speech_timestamps()) == expected  # the full list is not consumed by pop


def test_segments_are_reported_while_streaming_not_only_at_the_end(vad, audio):
    # The silence after the first words ends their segment long before the audio ends.
    expected = _batch(vad, audio)
    vad.reset()
    first_end = expected[0][1]
    got_before_end = []
    for i in range(0, len(audio), 512):
        vad.feed(audio[i:i + 512])
        got_before_end += _pairs(vad.pop_speech_timestamps())
        if vad.samples_fed > first_end + SR:  # a second of silence after the first segment
            break
    assert got_before_end
    assert got_before_end[0] == expected[0]


def test_in_speech_flag_and_finish_closes_open_segment(vad, audio):
    expected = _batch(vad, audio)
    start, end = expected[0]
    cut = (start + end) // 2  # stop in the middle of the first speech segment
    vad.reset()
    vad.feed(audio[:cut])
    assert vad.in_speech
    assert _pairs(vad.pop_speech_timestamps()) == []  # still open: nothing to report yet
    vad.finish()
    assert not vad.in_speech
    closed = _pairs(vad.pop_speech_timestamps())
    assert len(closed) == 1
    assert closed[0][0] == start
    assert closed[0][1] == cut == vad.samples_fed


def test_reset_makes_the_instance_reusable(vad, audio):
    first = _stream(vad, audio, [512])
    vad.feed(audio[:3000])  # leave junk behind: pending samples + state
    second = _stream(vad, audio, [333])  # _stream() starts with reset()
    assert first == second
    assert vad.samples_fed == len(audio)


def test_empty_and_tiny_chunks(vad, audio):
    vad.reset()
    vad.feed(np.zeros(0, dtype=np.float32))
    assert vad.samples_fed == 0
    vad.feed(audio[:100])  # less than one window: buffered, no inference yet
    assert vad.samples_fed == 100
    assert not vad.in_speech
    vad.finish()
    assert _pairs(vad.pop_speech_timestamps()) == []
