# ADR-0022: Speech-weighted RMS normalization: hard and soft thresholds on the VAD probabilities

- **Status:** accepted
- **Date:** 2026-10-26
- **Source:** `include/libspeech/dsp/normalize.h`, `src/dsp/normalize.cpp`, `src/bindings/python/libspeech/_normalize.py`, `tests/dsp/test_normalize.cpp`, `tests/python/test_normalize.py`, `tests/python/test_speech_normalize.py`, CHANGELOG (Unreleased)

## Context

ADR-0020's normalizer measures the RMS over every sample. A recording with long pauses reads quieter than its speech is,
so the gain comes out too high and the speech ends up louder than the target. `SileroVad` already gives a speech
probability per window (`speech_probabilities()`); the user asked to apply a hard and a soft threshold to those
probabilities and to use the result in the normalizer.

## Decision

- **Layer:** the weighting is pure DSP and stays in `speech::dsp` (no model dependency, ADR-0001/0002):
  - `speech_weights(prob, n, speech_gate, threshold, out)`: `hard` gives weight 1 when `p >= threshold` (the rule the VAD
    uses to open a segment), else 0; `soft` gives `(p - threshold) / (1 - threshold)` clamped to [0, 1] (0 up to the
    threshold, a linear ramp to 1 at `p = 1`). Hard needs `0 < threshold <= 1`, soft `0 < threshold < 1`; a probability
    outside [0, 1] or NaN throws.
  - `weighted_rms()` and `rms_normalize_weighted()`: window `k` (samples `[k * window, (k + 1) * window)`) counts with weight
    `weights[k]`; level = `sqrt(sum w_k e_k / sum w_k len_k)`. Samples after the last window (the VAD drops a trailing
    partial window) have weight 0.
- **Gain:** still one number for the whole signal, and the peak ceiling is checked on the whole signal, so a peak inside a
  weight-0 window still limits the gain (nothing clips). When the total weight is 0 (no speech) or the signal has a
  NaN/inf sample, the signal is returned unchanged.
- **Python:** `libspeech.speech_weights(probabilities, mode="hard", threshold=0.5)` and
  `libspeech.rms_normalize_weighted(signal, weights, window, target_dbfs, peak_ceiling)` (1-D) are the bindings;
  `libspeech.speech_rms_normalize(signal, *, sample_rate=16000, mode="hard", threshold=None, target_dbfs=-20.0,
  peak_ceiling=0.99, vad=None)` runs `SileroVad`, turns its probabilities into weights and normalizes. `threshold=None`
  means the VAD's own threshold; `vad` reuses a loaded model.
- `Audio.normalize_rms` and `rms_normalize` are **not** changed.

## Rejected alternatives

- One mode that combines the two (a hard mask, then a soft weight): two separate modes are easier to explain and test.
- Per-sample weights: a probability covers a whole window, so per-window weights are smaller and exact.
- The C++ `Audio` method or a VAD call inside `speech_dsp`: `speech_dsp` must not depend on the models.
- Multi-channel input: the windows of a planar block do not align per channel; only 1-D is supported for now.

## Consequences

- The speech level is the same for a recording with short and one with long pauses; the plain RMS is not.
- A wrong VAD decision moves the level (noise counted as speech lowers it, missed speech raises it).
- `speech_rms_normalize` passes the signal to the VAD as a Python list, which is slow for very long inputs.
- Not verified: Windows and macOS, OpenMP, 8 kHz VAD, real recordings (only synthetic speech-like audio was used).
