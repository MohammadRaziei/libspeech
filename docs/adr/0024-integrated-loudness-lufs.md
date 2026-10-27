# ADR-0024: Integrated loudness (LUFS, ITU-R BS.1770-4) and `loudness_normalize`

- **Status:** accepted
- **Date:** 2026-10-26
- **Source:** `include/libspeech/dsp/normalize.h`, `src/dsp/normalize.cpp`, `src/bindings/python/bind_dsp.cpp`, `tests/dsp/test_normalize.cpp`, `tests/python/test_normalize.py`, CHANGELOG (Unreleased)

## Context

RMS (ADR-0020) and the VAD-weighted RMS (ADR-0022) measure signal energy. Broadcast and podcast targets use LUFS, which
weights frequencies the way hearing does (K-weighting) and ignores silence and much quieter parts by gating. The owner
asked for the normalization ideas that looked good to be implemented; LUFS was the one named as next and needs no model.

## Decision

- **`speech::dsp::integrated_loudness(x, n, sample_rate)`** returns LUFS: the K-weighting filter (a +4 dB high shelf, then a
  38 Hz high-pass), the mean square over 400 ms blocks every 100 ms, an absolute gate at -70 LUFS, a relative gate 10 LU below
  the mean loudness of the blocks that passed, and `-0.691 + 10 log10(mean square of the blocks left)`. Only complete blocks
  count; a signal shorter than 400 ms is one block of what there is; silence gives -infinity.
- **Filter coefficients:** from the analogue prototype (shelf: 1681.97 Hz, +3.9998 dB, Q 0.70718, `Vb = Vh^0.49967`;
  high-pass: 38.135 Hz, Q 0.50033) through the bilinear transform with pre-warping. These formulas reproduce the standard's
  published 48 kHz coefficients to about 1e-13 (checked in the Python tests) and work for any rate >= 8 kHz. Away from 48 kHz the
  gain near 1 kHz differs from the ideal analogue filter by a few hundredths of a dB (0.04 dB at 16 kHz); this is inherent to the
  bilinear transform and is not corrected. A first version that used the shelf formulas of an audio-EQ cookbook was 0.2 dB off
  at 997 Hz and was dropped; pyloudnorm uses those formulas too, so it is not used as a reference.
- **`loudness_normalize(x, n, sample_rate, target_lufs = -23, peak_ceiling = 0.99, limiter_lookahead = 0)`:** one gain for the
  whole signal; peaks are handled as in ADR-0023 (ceiling lowers the gain, or with `limiter_lookahead > 0` the soft limiter
  `limit_peaks()` lowers only the peaks). Silence and signals with NaN/inf are left unchanged.
- **Python:** `libspeech.integrated_loudness(signal, sample_rate)` and `libspeech.loudness_normalize(signal, sample_rate,
  target_lufs, peak_ceiling, limiter_lookahead)`, 1-D float32.

## Rejected alternatives

- Multi-channel loudness (channel weights 1.41 for surround, sum of channel powers): only mono for now, like the other
  normalizers.
- Combining LUFS with the VAD (loudness of the speech blocks only, as "dialogue-gated" loudness does): the standard gating
  already ignores silence; a VAD mask on 100 ms hops would need a mapping from the 32 ms VAD windows and was left out.
- Loudness range (LRA) and true-peak (4x oversampling) measurement: not asked for.

## Consequences

- The result follows perceived loudness and is independent of the amount of silence, so two recordings of one speaker
  normalized to the same LUFS sound equally loud, which equal RMS does not guarantee.
- The absolute gate is fixed at -70 LUFS, so after scaling the gating can differ from before for signals with blocks close to
  the gate; the output loudness is then not exactly the target (tests keep clear of the gate).
- Not verified: real recordings, Windows and macOS, a comparison with a certified meter (EBU Tech 3341 test files). The
  997 Hz sine value and an independent re-implementation from the definition are the references.
