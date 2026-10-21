# ADR-0002: DSP is flat, non-virtual C++ with its own public types

- **Status:** accepted
- **Date:** 2026-09/10
- **Source:** CONTRIBUTING.md (style notes), CHANGELOG 0.4.0 (Breaking), `src/third_party/audioflux/ISSUES.md`, commit 65d08cf

## Context

DSP code runs in tight loops and sits on a vendored C library (AudioFlux) whose types and macros must not leak to users.

## Decision

- `src/dsp/` has no `virtual` methods and no inheritance hierarchies; the vtable cost in a hot loop was judged real.
- No AudioFlux header is public. `window_type` (an `enum class`) replaced AudioFlux's `WindowType` / `Window_*` in public headers; other AudioFlux names stay inside `src/`.
- Types are our own (`stft`, `resample`, `complex_spectrogram`, `dct_ii_plan`, `span<T>`, `frame_view<T>`).
- `speech::models` is the opposite case on purpose: it uses interfaces (`denoiser`) because a handful of calls per signal makes the abstraction free (ADR-0007).

## Rejected alternatives

Wrapping AudioFlux headers directly in the public API (replaced in 0.4.0, commit 65d08cf).

## Consequences

- Fast and predictable DSP; the price is some duplicated per-operator code.
- Bugs found in the vendored C are documented with repro steps in `src/third_party/audioflux/ISSUES.md`, not hidden.
