# ADR-0007: Denoiser: one interface, pimpl backends, non-virtual process() and close(), thread-safe

- **Status:** accepted
- **Date:** 2026-10-09 to 2026-10-18
- **Source:** commits a5de729, 4af5bdd, CHANGELOG 0.4.0, `include/libspeech/models/`

## Context

Four backends compete (facebook, speechbrain, gtcrn, dpdfnet) and users call `process()` from several threads and must be able to free memory early.

## Decision

- `denoiser` is a single-inheritance interface; backends are pimpl classes (`facebook_denoiser`, `speechbrain_denoiser`, `gtcrn_denoiser`, `dpdfnet_denoiser`). `process()` and `close()` are non-virtual in the base (NVI): argument validation and the closed-state check live there, once.
- Concurrent `process()` calls are allowed; `close()` releases the session safely (Python: context manager, `closed`). Construction and copying are restricted.
- `Denoiser("<backend>")` uses that backend's standard model; `model=` picks another. `num_threads` defaults to 1 (0 means automatic). The GIL is released while running. A NumPy overload exists.
- `facebook` and `speechbrain` are whole-signal models. `gtcrn` and `dpdfnet` work frame by frame and carry state (ADR-0008).

## Rejected alternatives

None recorded in the history.

## Consequences

- A backend must implement validation-free hooks only; do not put validation in a backend.
- Memory in Python is released by `close()` or the context manager, not by garbage collection timing.
