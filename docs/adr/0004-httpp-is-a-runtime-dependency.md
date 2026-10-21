# ADR-0004: httpp is an external runtime dependency (downloads, SHA-256, progress), not vendored

- **Status:** accepted
- **Date:** 2026-09-16 to 2026-10-17
- **Source:** commits cfc7b96, a6851f8, be00157; CMakeLists.txt `find_package(httpp)` comment; pyproject.toml

## Context

Model downloads need an HTTPS client, a progress bar, a download lock and SHA-256. An earlier setup vendored cpp-httplib + Mbed TLS and a custom progress bar.

## Decision

- `httpp` (a sibling project) provides `httpp::download`, `httpp::progress::bar` and `httpp::sha256_hex`. It is `pip install`ed, found with `find_package(httpp)` at configure time (`python -c "import httpp; httpp.get_cmake_dir()"`), and is a build requirement and a runtime requirement of the wheel.
- The version range is deliberately narrow: `httpp>=0.7,<0.8`, because the download lock and SHA-256 functions are not in 0.6.x and httpp's API is still changing.
- The wheel does not bundle httpp's shared library; `__init__.py` loads it from the installed httpp package (`httpp.get_lib_dir()`).

## Rejected alternatives

Vendoring or patching an HTTPS stack in tree; keeping the custom progress bar.

## Consequences

- Security fixes in the HTTPS stack arrive through httpp, not through this repository.
- Every httpp minor release needs an explicit bump and a test run here.
