# ADR-0009: A denoiser returns audio at the level of its input (SpeechBrain is level-matched)

- **Status:** accepted
- **Date:** 2026-10-20
- **Source:** commit 9ee1427, `src/models/level_match.h`, `tests/models/test_level_match.cpp`, CHANGELOG (Fixed)

## Context

SpeechBrain's SepFormer is trained with a scale-invariant loss (SI-SNR), so the absolute level of its output is arbitrary. On real speech its raw output was about 15x louder than the input, with peaks of 6 to 27, and 14 to 43% of the samples clipped once saved as 16-bit.

## Decision

- `speechbrain_denoiser::process` scales its output to the RMS of the input, then lowers the gain if the peak would exceed max(0.99, input peak). Silent or non-finite input is left alone (`speech::models::detail::match_level`).
- Measured on one real clip scaled by 0.02, 1.0 and 6.0: output/input RMS 1.00. `gtcrn`, `dpdfnet` and `facebook` were already at the input's level (0.94 to 1.01); only `facebook` on very quiet input (peak 0.01) returned 0.61, which is the model removing more of a very quiet signal, not clipping, and was left as it is.
- Decided by: the project owner agreed it should be automatic; the design is the maintainer's assistant's.

## Rejected alternatives

Leaving the level to the user (everyone who saves the output gets clipping).

## Consequences

- Not a benchmark: one clip, one speaker.
- A general RMS normalizer for `Audio` is a separate open item in `checklist.md`.
