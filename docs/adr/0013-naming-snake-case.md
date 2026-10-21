# ADR-0013: C++ names are STL-style snake_case; Python classes keep PascalCase

- **Status:** accepted
- **Date:** 2026-10-14
- **Source:** CHANGELOG 0.4.0 (Breaking), CONTRIBUTING.md (Style notes), `docs/api-renames.md`, commit 65d08cf

## Context

The API mixed camelCase and PascalCase; the Python API mirrors the C++ one.

## Decision

- Types, enumerators, functions, methods, variables and constants are `snake_case`; private members end in `_`; macros are `UPPER_CASE`; namespaces and files are lowercase.
- Python class names stay `PascalCase` (`libspeech.STFT`): they are strings in the bindings.
- A type cannot share a method's name, so a few names changed beyond case (`STFT::stft` to `stft::compute`, `Resample::resample` to `resample::process`, `Spectrogram` to `complex_spectrogram`, `DctII` to `dct_ii_plan`, `Denoiser::Create` to `denoiser::create`). The full table is `docs/api-renames.md`.
- Names from other libraries keep their own spelling and must not leak into public headers.

## Rejected alternatives

Keeping the old names (breaking change taken in 0.4.0).

## Consequences

- Old C++ code does not compile against 0.4.0 without the rename table.
