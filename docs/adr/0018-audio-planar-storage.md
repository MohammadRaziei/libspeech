# ADR-0018: `Audio` stores samples in one planar shared block; miniaudio is the only decoder

- **Status:** accepted
- **Date:** 2026-09-29 to 2026-10-18
- **Source:** commits af12930, 87fa0ec

## Context

A vector of vectors cannot be shown to NumPy or DLPack without copying, and stale lengths were a bug source. Several decoders meant several code paths.

## Decision

- Samples live in a single planar block shared with views (`shared_samples()`), so `np.asarray(audio)` and DLPack are zero-copy. `Audio(array, sample_rate)`, `from_numpy` and `from_dlpack` build one.
- `src/audio.cpp` decodes WAV, MP3 and FLAC through miniaudio's bundled codecs only.

## Rejected alternatives

A vector-of-vectors buffer; separate dr_libs decoders (ADR-0003).

## Consequences

- A view can outlive the `Audio`; the shared block keeps the memory alive.
- Behavior beyond the commit messages is not re-verified in this ADR.
