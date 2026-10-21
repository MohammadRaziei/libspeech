# ADR-0001: Three layers (dsp, io, models) behind one `speech` library

- **Status:** accepted
- **Date:** 2026-08/09
- **Source:** `git log` (CMake restructuring), checklist.md sections on namespacing and `speech::speech`, commit 62f3cf9

## Context

The project grew three domains with very different dependencies: DSP (pure C++ over vendored C), audio I/O (miniaudio) and models (ONNX Runtime, network).

## Decision

- Three static libraries and namespaces: `speech_dsp` / `speech::dsp`, `speech_io` / `speech::io`, `speech_models` / `speech::models`. A namespace names a domain, not the main class in it.
- One shared library, `speech`, links all three and is what users and the Python package load (`speech::speech` is the "everything" alias).
- `-DBUILD_MODELS=OFF` configures `speech::dsp` and its tests only, with no ONNX Runtime and no network.
- On MSVC the three archives are linked with `/WHOLEARCHIVE`, because `speech.dll`'s only own source calls nothing and the linker would otherwise drop every object, leaving nothing to export (commit 62f3cf9).

## Rejected alternatives

None recorded in the history.

## Consequences

- A DSP-only build is cheap and works offline; tests are split the same way (`speech_test_cpp_dsp` needs no network).
- Anything that must work in a DSP-only build has to live in headers (see ADR-0014: the logger state is inline for this reason).
- Not verified here: the MSVC `/WHOLEARCHIVE` fix runs only in the project's CI.
