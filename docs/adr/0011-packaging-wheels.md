# ADR-0011: Wheels: one cp312-abi3 wheel, separate free-threaded wheels, no musllinux, no delvewheel on Windows

- **Status:** accepted
- **Date:** 2026-09 to 2026-10-18
- **Source:** pyproject.toml comments, commits 66b46b1, e7d9faf, 7ba1ab7; CONTRIBUTING.md (Packaging / CI)

## Context

Wheels must cover many platforms while carrying ONNX Runtime and `speech` next to the extension modules.

## Decision

- One `cp312-abi3` wheel (forward compatible with 3.12 and newer) built with nanobind's `STABLE_ABI`; older interpreters (3.9 to 3.11) get their ordinary per-version wheels.
- Free-threaded CPython (3.14t, 3.15t) gets its own wheels built in a separate step, because cibuildwheel 4.x no longer builds `cp313t`.
- No musllinux wheels (ADR-0006). No 32-bit Windows.
- On Windows `delvewheel` repair is disabled: the wheel already places `speech.dll` and `onnxruntime.dll` next to the modules and `__init__.py` loads them (commit e7d9faf).
- Rule from experience: when install rules, `pyproject.toml` or the wheels workflow change, build a wheel and inspect its contents. Twice a bug was invisible in the config: a duplicated ~20 MB ONNX Runtime binary, and a silently missing `abi3` tag.

## Rejected alternatives

None recorded in the history.

## Consequences

- Wheel builds run only in the project's CI; nothing about them was re-run for this ADR.
