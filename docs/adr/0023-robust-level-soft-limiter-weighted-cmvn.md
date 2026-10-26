# ADR-0023: Robust speech level, soft peak limiter and weighted CMVN

- **Status:** accepted
- **Date:** 2026-10-26
- **Source:** `include/libspeech/dsp/normalize.h`, `src/dsp/normalize.cpp`, `src/bindings/python/bind_dsp.cpp`, `src/bindings/python/libspeech/_normalize.py`, `tests/dsp/test_normalize.cpp`, `tests/python/test_normalize.py`, `tests/python/test_speech_normalize.py`, CHANGELOG (Unreleased)

## Context

ADR-0022 measures the level of the speech windows only. Three weak points were left, and the owner asked for the
normalization ideas that looked good to be implemented (no choice was made between the alternatives, so the choices are
recorded here):

1. A few very loud windows (plosives, clicks) still pull the gain down.
2. The peak ceiling lowers the gain of the **whole** signal, so one stray peak makes all the speech too quiet.
3. Features (MFCC, log-mel) for ASR or speaker models need mean and variance normalization over the speech frames only.

## Decision

All three are opt-in; with the defaults the behaviour of ADR-0022 is unchanged.

- **`clip_percentile`** (0 < p <= 100, default 100 = off) in `weighted_rms()`, `rms_normalize_weighted()` and Python
  `speech_rms_normalize()`: the mean square of every speech window above the weighted p-th percentile of the window mean
  squares is replaced by that percentile (winsorizing), then the usual weighted RMS is taken. It stays an RMS, so the
  target keeps its meaning (a plain percentile of window levels would put the output RMS below the target).
  Rejected: ITU-T P.56 active speech level (a heavier algorithm that does not combine with soft weights), a trimmed mean
  (no better, and it throws away windows).
- **`limit_peaks(x, n, ceiling, lookahead)`** and `limiter_lookahead` in `rms_normalize_weighted()` (Python
  `limiter_ms` in `speech_rms_normalize()`): the needed gain per sample is `min(1, ceiling / |x|)`, its minimum over the
  next `lookahead` samples is taken and averaged over `lookahead` samples. This guarantees `|y| <= ceiling`, the gain falls
  before a peak and recovers after it, and it is exactly 1 more than `lookahead - 1` samples away from any peak. With
  `limiter_lookahead > 0` the gain is the one for the target and the limiter handles the peaks; with 0 the old rule
  (ceiling lowers the whole gain) stays. `output_rms_dbfs` is then the level before limiting.
- **`weighted_cmvn(features, frames, dim, weights, normalize_variance, eps)`**: per dimension, weighted mean (and
  variance) from the frames with weight, applied to every frame. Row-major `(frames, dim)`. Python `weighted_cmvn()` returns
  a new array; `frame_weights()` maps per-window VAD weights to feature frames (`center=True` convention).
- Python also exposes `weighted_rms()` and `limit_peaks()`. All C++ code is in `speech_dsp` (no model dependency).

## Consequences

- One stray peak no longer lowers the speech level when the limiter is on; the price is a (small) gain modulation around the
  peak, which is not transparent for a sustained overload.
- `clip_percentile < 100` raises the gain relative to the plain weighted RMS, so the mean RMS of the output is above the
  target when the speech has loud windows.
- Not verified: real recordings, Windows and macOS, 8 kHz, features other than synthetic ones. `limit_peaks` allocates
  two `double` vectors of the signal length.
