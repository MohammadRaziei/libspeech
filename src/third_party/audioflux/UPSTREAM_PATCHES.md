# Upstream Patches — Vendored AudioFlux Code

See **[`/audioflux_issues.md`](../../../../audioflux_issues.md)** (project root)
for the full write-up of every AudioFlux issue found and fixed while
vendoring this code, including repro steps and before/after diffs — kept
there instead of duplicated here so there's a single source of truth.

## Additive accessors (no behaviour change)

This adds read-only access for the C++ wrapper; no existing function changed:

- `resampleObj_getTables()` (`src/dsp/resample_algorithm.{c,h}`): exposes the
  interpolation tables and rate state so `speech::dsp::Resample` can build an
  exact rational polyphase filter bank from the very same tables.
