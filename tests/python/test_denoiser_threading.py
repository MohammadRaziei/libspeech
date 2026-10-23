"""A Denoiser used from several Python threads: concurrent process() and streams, and the GIL.

Marked `models`: runs only with `pytest -m models` (CMake target speech_test_python_models_e2e); the models are
downloaded on first use. Two properties are checked for the two streaming backends that are small enough to run
here (GTCRN and DPDFNet):

* one Denoiser shared by several threads gives each thread exactly the result it would get alone, with
  process() and with streams (every thread has its own stream; the model's recurrent state is per stream);
* process() and feed() release the GIL while the model runs, so other Python threads keep going. If they did
  not, a thread that only sleeps and counts would stand still for the whole call.
"""

from __future__ import annotations

import threading
import time
from typing import Callable

import numpy as np
import pytest

import libspeech

pytestmark = pytest.mark.models

SR = 16000
THREADS = 4


def _noisy_tone(seconds: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    voice = sum(np.sin(2 * np.pi * f * t) / k for k, f in enumerate([180, 360, 540, 720], start=1))
    voice *= 0.5 + 0.5 * np.sin(2 * np.pi * 2.5 * t) ** 2
    return (0.15 * voice + 0.05 * rng.standard_normal(t.size)).astype(np.float32)


def _run_together(work: Callable[[int], np.ndarray], n: int = THREADS) -> list[np.ndarray]:
    """Runs work(0..n-1) in n threads that all start at the same moment; re-raises the first exception."""
    start = threading.Barrier(n)
    results: list = [None] * n
    errors: list[BaseException] = []

    def target(i: int) -> None:
        try:
            start.wait()
            results[i] = work(i)
        except BaseException as exc:  # handed to the main thread below
            errors.append(exc)

    threads = [threading.Thread(target=target, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]
    return results


@pytest.fixture(scope="module", params=["gtcrn", "dpdfnet"])
def den(request):
    return libspeech.Denoiser(request.param)


@pytest.fixture(scope="module")
def clips() -> list[np.ndarray]:
    # Different lengths and content per thread, so a mix-up between threads cannot cancel out.
    return [_noisy_tone(1.0 + 0.37 * i, seed=10 + i) for i in range(THREADS)]


@pytest.fixture(scope="module")
def expected(den, clips) -> list[np.ndarray]:
    return [den.process(c) for c in clips]


def test_concurrent_process_calls_equal_the_serial_results(den, clips, expected):
    for _ in range(3):
        got = _run_together(lambda i: den.process(clips[i]))
        for out, want in zip(got, expected):
            np.testing.assert_array_equal(out, want)


def test_concurrent_streams_on_one_denoiser_equal_the_serial_results(den, clips, expected):
    def stream_clip(i: int) -> np.ndarray:
        stream = den.stream()
        step = 160 + 37 * i  # every thread feeds in chunks of its own size
        pieces = [stream.feed(clips[i][s : s + step]) for s in range(0, len(clips[i]), step)]
        pieces.append(stream.finish())
        return np.concatenate(pieces)

    for _ in range(3):
        got = _run_together(stream_clip)
        for out, want in zip(got, expected):
            np.testing.assert_array_equal(out, want)


def _ticks_while(call: Callable[[], object]) -> tuple[int, float]:
    """(ticks a sleeping Python thread made during call(), how long call() took in seconds)."""
    ticks = 0
    stop = threading.Event()

    def ticker() -> None:
        nonlocal ticks
        while not stop.is_set():
            time.sleep(0.001)
            ticks += 1

    thread = threading.Thread(target=ticker)
    thread.start()
    try:
        time.sleep(0.05)  # let the ticker get going
        before, t0 = ticks, time.perf_counter()
        call()
        elapsed = time.perf_counter() - t0
        gained = ticks - before
    finally:
        stop.set()
        thread.join()
    return gained, elapsed


def _assert_python_threads_keep_running(call: Callable[[], object]) -> None:
    gained, elapsed = _ticks_while(call)
    assert elapsed > 0.1, f"the call took only {elapsed:.3f} s, too short to tell (use a longer clip)"
    # A free thread that sleeps 1 ms per round makes hundreds of ticks per second; with the GIL held it makes none
    # until the call returns. 50 per second leaves a wide margin for a loaded machine.
    assert gained >= 50 * elapsed, f"only {gained} ticks in {elapsed:.2f} s: the GIL was held"


def test_process_releases_the_gil(den):
    clip = _noisy_tone(8.0, seed=1)
    _assert_python_threads_keep_running(lambda: den.process(clip))


def test_stream_feed_releases_the_gil(den):
    clip = _noisy_tone(8.0, seed=2)
    stream = den.stream()
    _assert_python_threads_keep_running(lambda: stream.feed(clip))
    stream.finish()
