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
git submodule update --init src/third_party/miniaudio

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
speech_test                               # everything
├── speech_test_cpp
│   ├── speech_test_cpp_dsp               # cheap, no network/ONNX deps
│   └── speech_test_cpp_models
└── speech_test_python
    ├── speech_test_python_audio          # Audio load/save/resample/to_mono
    ├── speech_test_python_audio_interop  # np.asarray(audio), DLPack, Audio(array)
    ├── speech_test_python_dsp            # FFT / STFT / MFCC / Resample classes
    ├── speech_test_python_functional     # libspeech.stft / istft (librosa-style)
    ├── speech_test_python_numpy_api      # NumPy in/out fast paths
    ├── speech_test_python_simd           # SIMD backend selection and agreement
    ├── speech_test_python_logging
    ├── speech_test_python_models         # Denoiser / SileroVad bindings
    ├── speech_test_python_model_files    # checksums, downloads (local HTTP server, offline)
    ├── speech_test_python_free_threading # GIL stays off on free-threaded CPython (skips elsewhere)
    ├── speech_test_python_readme_examples # runs the python blocks of README.md
    └── speech_test_python_librosa_compat # needs the `test` extra; skips itself without
```

`speech_test_python_models_e2e` is not part of `speech_test`: it needs the network (real model
downloads) and runs the tests marked `models` (`pytest -m models`), including the model example in
`README.md`.

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

## SIMD and portability

Kernels that benefit from explicit SIMD (today the polyphase resampler, ~2.3x over its portable
twin on AVX2) have a portable C++ implementation plus vectorized twins that are **selected at run
time from the CPU**, never from build flags, so one wheel or one default CMake build is correct on
every machine of its architecture and fast on the capable ones. No `-march=native`, no per-ISA build
variants. The real-input FFT behind STFT/MFCC is plain portable C++ that the compiler vectorizes for
the baseline ISA: a hand-targeted AVX2 clone of it was measured (n_fft 64-8192) and was never faster,
so it was dropped.

| Target | Fast path | Compilers |
|---|---|---|
| x86 / x86-64 | AVX2 + FMA (CPUID and OS YMM-state checked) | GCC, Clang, MSVC |
| ARM64 (Linux, macOS, Windows) | NEON | GCC, Clang, MSVC |
| anything else (32-bit ARM, RISC-V, ...) | portable C++ | any C++17 compiler |

```python
import libspeech
libspeech.simd_backend()            # 'avx2+fma' | 'neon' | 'generic'
libspeech.set_simd_enabled(False)   # run-time switch (objects/calls created afterwards)
```

- Environment: `LIBSPEECH_SIMD=off` forces the portable kernels (`LIBSPEECH_DISABLE_AVX2=1` still works).
- All backends are cross-checked against each other and against independent references in
  `tests/dsp/test_fast_paths.cpp`. The suite was run on x86-64 (GCC and Clang), 32-bit x86,
  AArch64 (under QEMU), and with SIMD disabled at run time.
- Support is decided in the sources, not the build system. `src/dsp/simd_internal.h` holds every
  architecture/compiler macro (`LS_HAVE_X86_AVX2`, `LS_HAVE_ARM_NEON`, `LS_TARGET_AVX2`, ...):
  `#if LS_HAVE_X86_AVX2` the AVX2 kernel is compiled and used when the CPU has it, otherwise the
  portable one is. Kernels never test `__x86_64__`/`_MSC_VER` directly.

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

- **Naming follows the standard library.** Everything we define is `snake_case`:
  types (`stft`, `frame_view`, `complex_spectrogram`), enumerators
  (`window_type::hann`), functions and methods (`cal_time_length()`,
  `stft_into()`), variables and parameters. Private data members end in an
  underscore (`fft_length_`); constants are plain `snake_case` too (`max_length`);
  macros are `UPPER_CASE`; namespaces and file names are lowercase. This mirrors
  the Python API one-to-one (`stft.cal_time_length` / `stft::cal_time_length`).
  Python class names stay `PascalCase` there (`libspeech.STFT`, per PEP 8): they
  are strings in the bindings, not C++ identifiers. Names that come from other
  libraries keep their own spelling (`Ort::Session`, AudioFlux's `stftObj_new`,
  miniaudio's `deviceConfig.sampleRate`) -- do not rename those, and do not let
  them leak into public headers (see `window_type`, which replaced AudioFlux's
  `WindowType` for exactly that reason). A type that shares its name with a method
  does not compile (`stft::stft`), so pick a verb for the method (`compute`).
  The old camelCase/`PascalCase` names are listed in
  [`docs/api-renames.md`](docs/api-renames.md).
- DSP code (`src/dsp/`) stays flat/non-virtual C++ -- no `virtual` methods,
  no inheritance hierarchies. It's called in tight loops; polymorphism's
  vtable indirection has a real, measured cost there. Models code
  (`src/models/`) uses interfaces (`denoiser`) where multiple backends
  genuinely compete -- that's a different situation (a handful of calls,
  not millions) where the cost doesn't matter and the abstraction helps.
- Logging goes through libspeech's own logger (`libspeech/detail/log.h`):
  `SPEECH_LOG(DEBUG) << TAG(log_tag) << ... << std::endl;` (`TAG`/`COND` are the
  `aixlog` helpers; the output side is ours and writes to stderr). Use `DEBUG` for
  normal lifecycle/results, `ERROR` for failures. Reach for `TRACE` only where it
  exposes an internal computation someone would actually want to inspect when
  debugging a wrong result -- not on every call; it's easy to make `TRACE` noise
  that nobody reads. Users control the level with `speech::utils::set_log_level()`,
  `libspeech.set_log_level()` or `LIBSPEECH_LOG`; the default is `warning`.
- Prefer a small, real regression test over a comment saying "this used to
  be broken." If you find a bug while adding a feature, add the test that
  would have caught it before fixing the code.

## Type stubs

The package ships `py.typed` and a `.pyi` stub per compiled module, generated like ctoon's
(`python -m nanobind.stubgen`) in `src/bindings/python/CMakeLists.txt`: the `speech_stubs` target
writes them into the staged package in the build tree, and a wheel installs those files next to the
modules. The modules can only be imported as `libspeech.<module>` (they need the libraries
`__init__.py` preloads), so stubgen is run through the package. A module that cannot be imported
fails the build. Pure-Python additions (`_functional.py`) carry their own inline annotations.

## Versioning

`include/libspeech/version.h` is the single source of truth (both CMake
and `pyproject.toml`'s package version read from it). Don't hand-edit the
numbers there -- use `version.py`:

```bash
python version.py                    # show the current version
python version.py minor +            # bump the minor version
python version.py tag create patch   # bump patch, commit, and tag a release
```

## Docs and coverage

`-DBUILD_DOCS=ON` gives you `speech_docs`, laid out like ctoon's `docs/`: the C++ reference
(Doxygen, warnings-free -- keep it that way when you touch a public header), the Python
reference (Sphinx; needs `pip install sphinx doxygen-awesome-css markdown-it-py`, `BUILD_PYTHON`
and a local build, because it imports the package staged in the build tree), the route index
(`docs/supported_langs.json` plus one fragment folder per language) and the Markdown pages
(changelog, API renames, this file). Output: `<build>/docs/out/`; `speech_docs_cpp` and
`speech_docs_python` build one part. Footer links to the benchmark and coverage pages are
dropped when those pages are not in the output (`docs/FinishPages.cmake`).
`-DBUILD_COVERAGE=ON` (needs
`lcov`; add `-DBUILD_PYTHON=ON` and `pip install coverage` for the Python
side) gives you `speech_coverage`, which runs every component's tests
instrumented, merges them, and writes a dashboard to
`build/coverage/index.html`. It mirrors ctoon's setup (one target per component, a merged total,
a dashboard tying the reports together):

| Target | Covers | Output (under `build/coverage/`) |
|---|---|---|
| `speech_coverage_cpp_dsp` | `speech::dsp` (utest.h suite) | `cpp_dsp/` |
| `speech_coverage_cpp_models` | `speech::models` + download utils | `cpp_models/` |
| `speech_coverage_python` | the Python package's `.py` files (coverage.py) | `python/` |
| `speech_coverage_total` | lcov merge of all of the above | `total/` |

Each component (and the total) leaves a plain `coverage.lcov` next to its HTML, which is what CI
uploads to Codecov. `BUILD_COVERAGE` is off by default: it instruments `speech_dsp`/`speech_models`
with `--coverage`, which you do not want in a normal build. Without `BUILD_PYTHON` you get only the
C++ components. A new C++ module with its own test target
gets a coverage target by calling `speech_enable_coverage(<lib>)` and
`speech_add_cpp_coverage_target(...)` next to its tests (see
`tests/models/CMakeLists.txt` for a three-line example).

Two lcov 2.0 quirks worth knowing before you debug them yourself: its
`--list` output is wrong (use `--summary`, as the targets do), and an
`--extract`/`--exclude` pattern that matches nothing is a hard error unless
you pass `--ignore-errors unused`.

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
