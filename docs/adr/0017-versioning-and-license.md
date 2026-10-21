# ADR-0017: One version source (`version.h`) and Apache-2.0

- **Status:** accepted
- **Date:** 2026-10-17
- **Source:** CONTRIBUTING.md (Versioning), CHANGELOG 0.4.0 (Changed), commit 05bea82

## Context

The version was duplicated, and the license was declared as MIT in the README and `pyproject.toml` while `LICENSE` held pybind11's BSD-3 text.

## Decision

- `include/libspeech/version.h` is the only place with the numbers; CMake and `pyproject.toml` read it. Use `python version.py` (`minor +`, `tag create patch`), never edit the numbers by hand.
- The license is Apache-2.0: `LICENSE` is the standard text, `NOTICE` names third-party code and says downloaded models keep their own licenses. Copies already distributed under MIT stay MIT.

## Rejected alternatives

None recorded in the history.

## Consequences

- Release notes live in `CHANGELOG.md`; new behavior and breaking changes are written there in the same change.
