# ADR-0016: Testing: real behavior first, real references, a separate gate for heavy model tests

- **Status:** accepted
- **Date:** 2026-08 to 2026-10-20
- **Source:** CONTRIBUTING.md (Testing philosophy, Building and testing), pyproject.toml (`models` marker), commit a5de729

## Context

DSP and model code is easy to get subtly wrong; mocks hide that.

## Decision

- Test first, against real behavior (impulse response, round trip, energy compaction), with `tests/utest/utest.h` (one shared `main()`); no mocking of internals. A bug found while adding a feature gets a regression test before the fix.
- Cross-checks use independent references: librosa (needs the `test` extra; skipped on Python 3.13+ and free-threaded builds where its dependencies lack wheels), sherpa-onnx for GTCRN and DPDFNet, and all SIMD backends against each other.
- Targets are `speech_test_<language>_<module>`, each both a ctest entry and a buildable-and-run target; `speech_test` runs all. `speech_test_python_models_e2e` (pytest marker `models`, real downloads, about 1 GB RAM) is **not** part of `speech_test`.
- Python code is checked with `ruff`; the C++ docs (Doxygen) are kept warning-free.
- Every new test is mutation-checked: a deliberate bug is introduced, the test must fail, and then the bug is reverted. (A ceiling test for `match_level` passed with its check deleted and was rewritten because of this.)
- Coverage (`BUILD_COVERAGE`) is off by default and needs `lcov`.

## Rejected alternatives

None recorded in the history.

## Consequences

- Skipped or unverified, stated plainly: Windows and macOS for the 2026-10-20 logging and SpeechBrain changes, multi-core and OpenMP numbers, microphone input.
- Heavy tests only run when asked for.
