# ADR-0020: RMS normalizer: dsp functions, `Audio.normalize_rms`, Python `rms_normalize` / `rms_dbfs`

- **Status:** accepted
- **Date:** 2026-10-21
- **Source:** `include/libspeech/dsp/normalize.h`, `src/dsp/normalize.cpp`, `tests/dsp/test_normalize.cpp`, `tests/python/test_normalize.py`, CHANGELOG (Unreleased)

## Context

`checklist.md` listed an RMS normalizer for audio as open, with the name and the place of the API undecided. ADR-0009 had
already put SpeechBrain's output at the level of its input with a private helper (`models::detail::match_level`).

## Decision

- **Layer:** `speech::dsp` (ADR-0001/0002): flat functions on `float*`, no dependency on `Audio` or the models.
  `rms()`, `rms_dbfs()`, `scale_to_rms()` (linear target) and `rms_normalize()` (target in dBFS, default -20), both in
  place, returning `rms_normalize_result` (gain, input and output level, `peak_limited`).
- **Gain:** one gain for the whole signal: `target_rms / rms`; if that would put the peak above `peak_ceiling` (linear,
  default 0.99, as in ADR-0009) the gain becomes `peak_ceiling / peak` (the ceiling wins, so nothing clips). Silent
  input and input with a NaN/inf are left unchanged. Bad arguments throw `std::invalid_argument` (`ValueError` in Python).
- **Channels:** one call over a planar block gives one gain for all channels, which keeps their balance.
- **Entry points:** C++ `speech::io::audio::normalize_rms()` (returns a new audio, like `resample`/`to_mono`); Python
  `libspeech.rms_normalize(array, target_dbfs=-20.0, peak_ceiling=0.99)` (1-D or (channels, samples); returns a new
  float32 array), `libspeech.rms_dbfs(array)` and `Audio.normalize_rms(target_dbfs, peak_ceiling)`.
- `models::detail::match_level` was **not** changed: `speech_models` does not link `speech_dsp`, and sharing the ~10 lines
  would add a dependency between the layers.

## Rejected alternatives

- Per-channel gains (changes the stereo balance). Peak normalization and LUFS (ITU-R BS.1770): different measures, still
  open in `checklist.md`.
- A method on `Audio` only: the function on arrays is what NumPy users and the denoiser pipeline need.

## Consequences

- **Streaming / running RMS is not done.** A stream normalizer needs a decision (cumulative or sliding window, gain
  smoothing, how it behaves in the first seconds) and cannot equal the one-shot result; it is open in `checklist.md`.
- RMS of the whole signal is not perceptual loudness: a clip with long silences is raised more than one without.
- Not verified: Windows and macOS builds of the new code, OpenMP.
