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
git submodule update --init src/third_party/miniaudio src/third_party/dr_libs

# speech::models (BUILD_MODELS=ON, the default) needs httpp on
# CMAKE_PREFIX_PATH for model downloads (HTTPS client + progress bar):
pip install httpp
```

A few entries under `src/third_party/` (AudioFlux subset, aixlog) are
**not** submodules -- those are copied directly into the repo, alongside
the actual submodules. See `src/third_party/README.md` for why and which
ones.

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
what's already vendored under `src/third_party/`:

1. Copy only the specific `.c`/`.h` files actually needed (not the whole
   upstream project) into `src/third_party/<name>/`.
2. If you need to patch the vendored code, document the patch --
   what/why/before-after -- in `audioflux_issues.md` (or a new file
   following that structure for a different vendored project), so it can
   become an upstream PR later.
3. Security/network-critical code is a different story: `httpp`
   (HTTPS client + progress bar for model downloads) is a genuine,
   separately-`pip install`able runtime dependency, not something to vendor
   or patch in-tree -- see the `find_package(httpp)` comment block in
   `CMakeLists.txt` for why (this replaced an earlier vendored
   cpp-httplib + Mbed TLS setup for exactly this reason).

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

## Packaging / CI changes

If you touch `CMakeLists.txt`'s install rules, `pyproject.toml`, or
`.github/workflows/wheels.yml`, actually build a wheel locally and inspect
its *contents* before opening a PR -- don't just read the config and
assume it's right:

```bash
python -m build --wheel -o /tmp/wheelcheck
python -c "import zipfile; z = zipfile.ZipFile('/tmp/wheelcheck/<name>.whl'); [print(i.filename, i.file_size) for i in z.infolist()]"
```

This project has twice shipped a packaging bug invisible from reading the
config alone -- a duplicated ~20MB ONNX Runtime binary (symlinks get
dereferenced both by `install(FILES ...)` and by the wheel-zip step), and a
wheel silently missing its intended `abi3` tag (`CIBW_ENVIRONMENT_CP312`
looked like a real cibuildwheel option by analogy with
`CIBW_ENVIRONMENT_LINUX`, but isn't -- cibuildwheel only recognizes
platform suffixes there, not Python-version ones, and silently no-ops
instead of erroring). Both only showed up by actually opening the built
artifact.

## Current project status

See [`checklist.md`](checklist.md) for the up-to-date, detailed state of
the project -- what's done, in progress, or blocked, including several
real bugs found (and fixed) along the way with enough detail to understand
what happened and why.
