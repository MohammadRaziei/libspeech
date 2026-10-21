# ADR-0003: Vendoring: small slices copied in, patches documented; large or security-relevant code is a dependency

- **Status:** accepted
- **Date:** 2026-08 to 2026-10
- **Source:** CONTRIBUTING.md (Vendoring), `src/third_party/README.md`, commits c013b31, 87fa0ec, be00157

## Context

The project needs a slice of AudioFlux, a logger (AixLog), miniaudio's codecs and a SHA-256 function, and has had cpp-httplib, Mbed TLS, picosha2 and dr_libs in tree at different times.

## Decision

- Copy only the files actually needed into `src/third_party/<name>/` (AudioFlux subset, AixLog). They are not submodules.
- Every patch to copied code is documented (what, why, before and after): `src/third_party/audioflux/ISSUES.md`, and for AixLog `src/third_party/aixlog/README.md` (one patch: it no longer redirects `std::clog`, ADR-0014). Patches are written so they can become upstream PRs.
- `miniaudio` is a git submodule and brings its own copies of dr_wav, dr_mp3 and dr_flac, so the separate `dr_libs` submodule was removed (2026-09-29).
- Security- and network-critical code is not vendored: see ADR-0004.
- Sources moved from `src/vendor/` to `src/third_party/` (2026-09-26).

## Rejected alternatives

Vendored cpp-httplib + Mbed TLS (replaced by httpp), vendored picosha2 (replaced by `httpp::sha256_hex`), a separate dr_libs submodule.

## Consequences

- A fresh clone needs `git submodule update --init src/third_party/miniaudio`.
- Upgrading AixLog means redoing its one patch.
