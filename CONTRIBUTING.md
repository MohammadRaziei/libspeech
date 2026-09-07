# Contributing to libspeech

Thanks for considering a contribution! This document covers the project's
conventions -- most of them exist because of a real problem hit during
development, not arbitrary style preference, so a short "why" is included
where it helps.

## Getting the source

```bash
git clone https://github.com/MohammadRaziei/libspeech.git
cd libspeech

# Small, lightweight submodules:
git submodule update --init src/third_party/miniaudio src/third_party/dr_libs src/third_party/indicators

# Mbed TLS (pinned to v3.6.2, kept shallow -- see .gitmodules) has its own
# nested submodule ("framework") that also needs initializing:
git submodule update --init src/third_party/mbedtls
git submodule update --init --depth 1 src/third_party/mbedtls/framework
```

`src/vendor/` (AudioFlux subset, aixlog, httplib) is **not** a submodule --
those are copied directly into the repo. See `src/vendor/README.md` for why.

## Building and testing

```bash
mkdir build && cd build
cmake ..                       # downloads ONNX Runtime on first configure
cmake --build . -j$(nproc)
ctest                          # or: cmake --build . --target speech_test
```

Working only on `speech::dsp`? Skip ONNX Runtime/network entirely:
```bash
cmake .. -DBUILD_MODELS=OFF
```

Test targets are named `speech_test_<language>_<module>` and are all
discoverable via `cmake --build build --target help`:

```
speech_test
├── speech_test_cpp
│   ├── speech_test_cpp_dsp       # cheap, no network/ONNX deps
│   └── speech_test_cpp_models
└── speech_test_python
    ├── speech_test_python_audio
    ├── speech_test_python_dsp
    └── speech_test_python_models
```

Each leaf is both a `ctest` entry and a directly buildable+runnable target
(`cmake --build build --target speech_test_cpp_dsp` builds *and* runs it).
Run just the module you're working on rather than the whole tree while
iterating.

## Where things live

| Namespace | CMake target | Python module | What it is |
|---|---|---|---|
| `speech::dsp` | `speech_dsp` | `speech_dsp_py` | Resample, window, FFT, STFT, MFCC, DCT |
| `speech::io` | `speech_io` | `speech_io_py` | `Audio`: load/save/play/resample/to_mono |
| `speech::models` | `speech_models` | `speech_models_py` | `Denoiser`, `SileroVad`, ONNX plumbing |

A namespace names a *domain*, not the primary class in it (e.g. `speech::io`
holds `Audio`, not a class literally named `Io` -- `speech::io::Audio` reads
better than the `speech::audio::Audio` stutter a same-named namespace would
produce). Follow this when adding a new module.

Add a Python binding for a new module under `src/binding/bind_<name>/main.cpp`;
it's picked up automatically and named `speech_<name>_py` (mirroring the
C++ target). Add a matching `tests/python/test_<name>.py` and one line in
`tests/python/CMakeLists.txt` (`add_python_test(<name>)`).

## Testing philosophy: TDD with real behavior, not mocks

Every DSP operator in this project (Resample, Window, FFT, STFT, MFCC) was
built test-first with `tests/utest/utest.h` (a vendored single-header
framework -- see below), asserting real mathematical properties (impulse
response, energy compaction, round-trip identity) rather than mocking
internals. This caught several real bugs before they shipped -- see
`audioflux_issues.md` for concrete examples with repro steps. Please follow
the same approach for new DSP/model code: write the test against the
expected real behavior first, then make it pass.

`utest.h`/`utest_main.cxx` live in `tests/utest/`; a new `.cpp` test file
just needs `#include "utest.h"` and `UTEST(Suite, Name) { ... }` blocks --
no `UTEST_MAIN()` needed in each file, there's exactly one shared `main()`
in `tests/utest/utest_main.cxx`.

## Vendoring third-party C code

If a DSP operator needs more of AudioFlux (or another small C library) than
what's already vendored under `src/vendor/`:

1. Copy only the specific `.c`/`.h` files actually needed (not the whole
   upstream project) into `src/vendor/<name>/`.
2. If you need to patch the vendored code, document the patch --
   what/why/before-after -- in `audioflux_issues.md` (or a new file
   following that structure for a different vendored project), so it can
   become an upstream PR later.
3. Security-critical code (crypto, TLS) is the exception: keep those as
   full git submodules (see Mbed TLS above), not cherry-picked/vendored,
   so security patches stay easy to pull in.

## Style notes

- DSP code (`src/dsp/`) stays flat/non-virtual C++ -- no `virtual` methods,
  no inheritance hierarchies. It's called in tight loops; polymorphism's
  vtable indirection has a real, measured cost there. Models code
  (`src/models/`) uses interfaces (`Denoiser`) where multiple backends
  genuinely compete -- that's a different situation (a handful of calls,
  not millions) where the cost doesn't matter and the abstraction helps.
- Logging goes through `aixlog` (`LOG(DEBUG) << TAG("speech::module::Class") << ... << std::endl;`).
  Use `DEBUG` for normal lifecycle/results, `ERROR` for failures. Reach for
  `TRACE` only where it exposes an internal computation someone would
  actually want to inspect when debugging a wrong result -- not on every
  call; it's easy to make `TRACE` noise that nobody reads.
- Prefer a small, real regression test over a comment saying "this used to
  be broken." If you find a bug while adding a feature, add the test that
  would have caught it before fixing the code.

## Versioning

`include/libspeech/version.h` is the single source of truth (both CMake
and `pyproject.toml`'s package version read from it). Don't hand-edit the
numbers there -- use `version.py`:

```bash
python version.py                    # show the current version
python version.py minor +            # bump the minor version
python version.py tag create patch   # bump patch, commit, and tag a release
```

## Current project status

See [`checklist.md`](checklist.md) for the up-to-date, detailed state of
the project -- what's done, in progress, or blocked, including several
real bugs found (and fixed) along the way with enough detail to understand
what happened and why.
