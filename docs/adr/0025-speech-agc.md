# ADR-0025: Streaming speech AGC (`speech_agc` / `SpeechAgc`)

- **Status:** accepted
- **Date:** 2026-10-26
- **Source:** `include/libspeech/dsp/agc.h`, `src/dsp/agc.cpp`, `src/bindings/python/bind_dsp.cpp`, `src/bindings/python/libspeech/_normalize.py`, `tests/dsp/test_agc.cpp`, `tests/python/test_normalize.py`, `tests/python/test_speech_normalize.py`, CHANGELOG (Unreleased)

## Context

The normalizers of ADR-0020, 0022, 0023 and 0024 apply one gain to a whole recording, so a speaker who moves away from the
microphone, or two speakers at different distances, still come out at different levels inside it. The owner left the choice of
the next normalization method to the implementer; a streaming, time-varying AGC driven by the VAD probabilities was the one
named as next (it is what `pop_speech_probabilities()` was made for).

## Decision

`speech::dsp::speech_agc` (Python `libspeech.SpeechAgc`, offline helper `libspeech.speech_agc()`), in `speech_dsp`, no model dependency:

- **Level:** exponential average of the sample mean square, weighted per window by the speech weights from `speech_weights()`
  (ADR-0022). It is cumulative (a plain mean) until the accumulated weight reaches `level_ms`, then exponential, so it is
  right from the first speech sample. During weight 0 it is frozen: pauses and noise do not move it, so the noise floor is
  not amplified.
- **Gain:** `target / level`, clamped to `[min_gain_db, max_gain_db]` (defaults -24 / +24 dB), smoothed per sample with two one-pole
  filters: attack (gain falls, 20 ms) and release (gain rises, 400 ms). The gain is 1 until `warmup_ms` (250 ms) of speech has
  been seen. Defaults: target -20 dBFS, level time constant 1 s.
- **Peaks:** no look-ahead, so no latency: a sample still above `peak_ceiling` after the gain is scaled to the ceiling on its own
  (a one-sample hard limit; distortion only on such samples). A soft limit over a buffer is `limit_peaks()` (ADR-0023).
- **Streaming contract:** `process(x, n, weights, n_weights, window)` in place, state continues between calls; samples after the
  last window get weight 0. For chunks that are a multiple of the window (512 for Silero) the result is bit-identical for any
  chunking (tested). Validation happens before anything is changed. One instance per stream, not thread-safe.
- **Python:** `SpeechAgc(sample_rate, target_dbfs, level_ms, attack_ms, release_ms, max_gain_db, min_gain_db, peak_ceiling,
  warmup_ms)` with `process(signal, weights, window)` (returns a copy), `reset()`, `gain()`, `level_dbfs()`, `sample_rate`;
  `speech_agc(signal, sample_rate=16000, mode, threshold, vad, ...)` runs the VAD and then the AGC over a whole signal.

## Rejected alternatives

- A look-ahead limiter inside the AGC: it adds latency and buffering; `limit_peaks()` is available offline.
- Smoothing the gain in dB: for the small corrections an AGC makes the linear one-pole is indistinguishable, and cheaper.
- Compression (a level-dependent gain curve within one utterance): this is levelling, not dynamics processing.

## Consequences

- Speech at different levels is brought to about the target after the gain has settled (within the time constants); the first
  250 ms of speech are untouched and the gain then moves at the release rate, so the start of a recording is not corrected at
  once. An offline one-gain normalizer is still the better tool when latency does not matter.
- A wrong VAD decision moves the level; the cap of +24 dB limits how much noise misread as speech can be amplified.
- Not verified: real recordings and several real speakers, Windows and macOS, 8 kHz VAD, long streams (hours; the state is a few
  doubles, so no growth is expected).
