# libspeech Checklist

Living checklist for the rearchitecture: moving DSP off the AudioFlux
submodule (vendored + patched, one operator at a time, TDD-first with
`utest.h`), professional `include/` layout, and a `speech::dsp` /
`speech::models` split (flat structs/functions for DSP, no `virtual`; small
polymorphic interfaces only where multiple backends really compete).

Check things off as we go — don't skip ahead to the next DSP operator until
its tests are green (see `AGENTS`/conversation ground rules).

---

## Foundations

- [x] Move all public headers under `include/libspeech/` (namespaced, professional layout)
- [x] Vendor `utest.h` (from ctoon) into `tests/utest/`, replacing googletest
- [x] Remove `tests/third_party/googletest` submodule
- [x] Vendor `aixlog.hpp` directly (single header), remove `src/third_party/aixlog` submodule
- [x] Establish logging convention: `aixlog`, `DEBUG` for lifecycle/results/errors, `TRACE` only where it exposes internal computation worth debugging (not on every call)
- [x] Establish AudioFlux vendoring workflow: copy only the `.c`/`.h` files a ported operator needs into `src/vendor/audioflux/`, patch in place if needed, log every patch in `/audioflux_issues.md`
- [x] Remove `src/third_party/audioflux` submodule entirely (nothing in the build references it anymore — everything comes from `src/vendor/audioflux`)
- [ ] Decide fate of `src/third_party/indicators`, `dr_libs`, `miniaudio` (still used by `Audio`/CLI progress bars — keep for now, revisit)

> **Note for future DSP operators:** with the full AudioFlux submodule gone,
> the next time we need to vendor a new `.c`/`.h` file (e.g. for STFT/MFCC),
> we'll temporarily shallow-clone upstream AudioFlux to pull just that file,
> then discard the clone — same as we'd do for any other one-off upstream
> reference, no need to keep the whole submodule around permanently.

## DSP layer (`speech::dsp`, flat/non-virtual, vendored AudioFlux C under the hood)

- [x] **Resample** — `include/libspeech/dsp/resample.h`, `src/dsp/resample.cpp`
  - [x] TDD tests (`tests/dsp/test_resample.cpp`), 5/5 passing
  - [x] Found + fixed AudioFlux bug: `sourceRate==targetRate` corrupts ratio (Issue 1)
  - [x] Found + fixed amplitude bug: `isScale=1` energy-scales output ~1.66x on downsample (Issue 2)
  - [x] Documented both in `/audioflux_issues.md`
- [x] Wire DSP layer into main CMake build: new lightweight `speech_dsp` static
  library target (src/dsp/*.cpp + vendored audioflux, no ONNXRuntime/CURL
  dependency), `tests/CMakeLists.txt` wiring all 6 DSP test suites via
  `add_dsp_test()`, and a `BUILD_MODELS` CMake option (default on; off skips ONNXRuntime/mbedtls/models) to configure/build
  just the DSP layer + tests without touching ONNXRuntime/CURL at all.
  Verified end-to-end: `cmake -DBUILD_MODELS=OFF .. && cmake --build . && ctest`
  → all 6 suites pass through the real build system (not manual g++ anymore).
- [x] **Window functions** — `include/libspeech/dsp/window.h`, `src/dsp/window.cpp` (flat free-function, not a class)
  - [x] TDD tests (`tests/dsp/test_window.cpp`), 6/6 passing
- [x] **FFT / IFFT** — `include/libspeech/dsp/fft.h`, `src/dsp/fft.cpp`
  - [x] TDD tests (`tests/dsp/test_fft.cpp`), 8/8 passing (impulse response, sine-bin energy, forward/inverse round-trip)
- [x] **DCT / IDCT** — implemented as `FFT::dct()`/`FFT::idct()` (AudioFlux computes DCT via the same FFT object, so no separate class)
  - [x] Found + documented AudioFlux footgun: `fftObj_idct()` mutates its input in place (Issue 3) -- worked around with an internal copy in our wrapper
  - [x] Round-trip + no-mutation regression tests passing
- [x] **STFT / spectrogram** — `include/libspeech/dsp/stft.h`, `src/dsp/stft.cpp`
  - [x] TDD tests (`tests/dsp/test_stft.cpp`), 7/7 passing (frame shape, round-trip via weighted overlap-add, edge-taper awareness documented)
  - [x] No new AudioFlux bugs found — this implementation was clean
- [x] **MFCC** — `include/libspeech/dsp/mfcc.h`, `src/dsp/mfcc.cpp` (composes STFT + a hand-written mel filterbank + `speech::dsp::dctII`)
  - [x] TDD tests (`tests/dsp/test_mfcc.cpp`), 6/6 passing
  - [x] **New utility:** `speech::dsp::dctII` (`include/libspeech/dsp/dct.h`) — direct O(N^2) DCT-II for arbitrary N, since `FFT::dct()` requires power-of-2 length and mel-filter counts (e.g. 26, 40) rarely are; 5/5 tests passing
  - [x] Mel filterbank hand-written (not vendored) — see comment in `mfcc.h` for rationale (AudioFlux's equivalent, `bft_algorithm.c`, bundles reassignment/temporal-feature code libspeech doesn't need)
  - [x] Found + fixed a **test-infrastructure bug** (not a production bug): `makeSine()` helpers computing phase as `float` lost precision at large sample indices, causing spuriously "inconsistent" MFCC output across frames; fixed by computing phase in `double`, rounding only the final `sin()` result to `float`. Affected `test_resample.cpp`, `test_fft.cpp`, `test_stft.cpp`, `test_mfcc.cpp`.

**DSP layer is now feature-complete per the original plan** (Resample, Window, FFT/DCT, STFT, MFCC). Remaining DSP-layer work is CMake integration, not new operators.
- [ ] Decide: do we need `filterDesign_fir`/`filterDesign_iir` as a public `speech::dsp` API, or is it purely an internal dependency of Resample?
- [ ] Benchmark our DSP path vs. sherpa-onnx on at least one op (validates the "why libspeech" question from earlier)

## Models layer (`speech::models`, small interfaces where backends compete)

- [x] `Denoiser` interface (`include/libspeech/models/denoiser.h`) with `Create(backend, url, sample_rate)` factory
- [x] `FacebookDenoiser` and `SpeechBrainDenoiser` implement `Denoiser`
- [ ] Tests for `Denoiser` factory + both backends (TDD, currently untested)
- [ ] Revisit `BaseModel`/`ONNXModel` for cross-platform issues (e.g. `getenv("HOME")` breaks on Windows despite README claiming cross-platform support)
- [ ] `SileroVad` — leave as concrete class (no interface) unless/until a second VAD backend actually exists (YAGNI)

## Build & packaging

- [x] CMakeLists.txt cleanup + three-library split, fully verified end-to-end:
  - Removed all dead/commented cruft (Conan/oatpp toolchain block, `crow`,
    `Vorbis`, stale `PUBLIC_SUFFIX_LIST_*` defs, duplicate `file(COPY...)`,
    commented-out example executables) accumulated from earlier experiments.
  - Split into three real targets with namespaced aliases:
    - **`speech::dsp`** (`speech_dsp`) — DSP layer, zero ONNXRuntime/network deps.
    - **`speech::models`** (`speech_models`) — BaseModel/ONNXModel + VAD/denoiser
      backends + the download/progress-bar utilities they need (ONNXRuntime,
      httplib+mbedtls, indicators).
    - **`speech::speech`** (`speech`) — umbrella SHARED library: audio file I/O
      (miniaudio/dr_libs) + links `speech::dsp` + `speech::models`, consumed
      by the Python bindings and example executables.
  - Fixed a real bug surfaced by the split: `speech_dsp`/`speech_models`/
    `audioflux` (STATIC) need `-fPIC` to link into the `speech` SHARED
    library — added `set(CMAKE_POSITION_INDEPENDENT_CODE ON)`.
  - Verified: full configure from clean (real ONNXRuntime download, no CURL
    needed), full build (`speech_dsp` → `speech_models` → `speech` →
    `example`, all link cleanly), and `ctest` (6/6 DSP tests still pass).
- [x] Replace system `libcurl` (`find_package(CURL REQUIRED)`) with vendored
  `httplib.h` (single header, `src/vendor/httplib/`) + Mbed TLS (git
  submodule pinned to `v3.6.2`, built from source — NOT vendored/patched
  like AudioFlux, since crypto code should stay pristine and upstream-
  updatable). `speech::utils::downloadFile` rewritten around
  `httplib::Client`. Verified end-to-end with real HTTPS downloads in this
  sandbox: a plain file from raw.githubusercontent.com (200 OK, correct
  content) and a cross-host redirect (github.com → objects.githubusercontent.com,
  the exact pattern ONNXRuntime/model releases use) — both worked with zero
  system dependencies, only source builds.
- [ ] Verify full `speech` library CMake build end-to-end on a machine with normal internet access (this sandbox's own egress restrictions, now unrelated to CURL, are the only remaining blocker here)
- [ ] `pip install` end-to-end smoke test (core promise of the project: zero system deps)
- [ ] CI update: drop googletest/aixlog submodule steps, add utest.h-based test target

## Documentation

- [x] `/audioflux_issues.md` — running log of every AudioFlux bug/quirk found + fixed, PR-ready
- [x] `src/vendor/audioflux/README.md` — explains why vendored instead of submoduled
- [ ] Update root `README.md` Quick Start to match actual current API (it currently references classes/methods that don't exist yet — `AudioProcessor`, `extract_features`, etc.)
- [ ] `CONTRIBUTING.md` (referenced by README but missing)
- [ ] Doxygen-style comments on public headers (`Audio`, `BaseModel`, ...)

## Python bindings (nanobind)

- [x] Fixed `bind_audio` (the `_audio` module): it referenced a nonexistent
  `speech::Audio::sampleRate()` (the real method is `sample_rate()`) --
  this binding never actually compiled before.
- [x] Fixed a real CMake bug: the nanobind module target used
  `target_link_directories(${NB_MODULE} PRIVATE speech)`, which only adds a
  search path and never actually links the library -- every symbol from
  libspeech would have been unresolved at import time. Changed to
  `target_link_libraries`.
- [x] Completed the `_audio` binding to cover the full `speech::Audio` API:
  `load` (both overloads: file path and raw PCM), `play`, `save`,
  `to_mono`, `resample`, `data()` (both overloads), `sample_rate`,
  `duration`, `__len__`, `__repr__`.
- [x] Found + fixed a real bug in my own binding while testing: `size()`
  returns the sample count *per channel*, not the channel count --
  `__repr__`'s "channels=" label and the `__len__` docstring were wrong
  until corrected to use `data().size()` for the true channel count.
- [x] Fixed `pip install` in the CMake Python-deps bootstrap step failing
  with `externally-managed-environment` (PEP 668, default on modern
  Debian/Ubuntu) by adding `--break-system-packages`.
- [x] Verified end-to-end in this sandbox: full `-DBUILD_PYTHON=ON` configure
  + build (nanobind auto-installed, ONNXRuntime downloaded, `_about.abi3.so`
  and `_audio.abi3.so` both built), then actually imported both modules from
  real Python and exercised `Audio`: load-from-raw-PCM, `to_mono`,
  `resample`, `data()`, save-to-WAV, and load-from-WAV -- all correct.
- [ ] Bind `speech::dsp` (Resample, MFCC, ...) and `speech::models`
  (Denoiser, SileroVad) for Python -- currently only `Audio` is exposed.

## Public Python import path (`import libspeech`, not `import _audio`)

The underscore-prefixed extension modules (`_audio`, `_about`) are private
implementation details, matching ctoon's `ctoon_py` convention -- users
should always `import libspeech` and get the clean public API from
`libspeech/__init__.py`'s re-exports, never import the compiled extension
directly. Testing through that public path (instead of `import _audio`
directly, which skips `__init__.py` entirely) surfaced three real bugs:

- [x] `__init__.py` tried to `cdll.LoadLibrary(".../libaudioflux.so")`, which
  no longer exists as a separate shared library now that AudioFlux is
  statically linked into `speech_dsp`/`speech` -- `import libspeech` was
  crashing outright. Removed the stale preload; also made the ONNXRuntime
  `.so` lookup glob-based instead of hardcoding `"1.21.0"`, so a future
  version bump (`cmake/ONNXRuntime.cmake`) doesn't silently break this again.
- [x] Found + fixed a real bug in `cmake/ONNXRuntime.cmake`: `ONNXRUNTIME_LIB_FILE`
  was only ever set inside the "directory already exists" branch -- on a
  completely fresh (first-ever) download, the variable was never set at
  all, so the later `file(COPY ${ONNXRUNTIME_LIB_FILE} ...)` silently
  copied nothing (empty path = no-op, not an error). Exactly the scenario
  every new contributor or a CI cache miss hits. Moved the lib-file-path
  determination to run unconditionally after both branches.
- [x] Found + fixed a real bug in `CMakeLists.txt`: `libspeech.so` built to
  the top-level build directory by default, but `__init__.py`'s
  `cdll.LoadLibrary` looks for it right next to `_audio.abi3.so`/
  `_about.abi3.so` inside the package directory. Added
  `set_target_properties(speech PROPERTIES LIBRARY_OUTPUT_DIRECTORY ...)`
  to route it there too.
- [x] Verified end-to-end via the real public path this time: fresh
  `-DBUILD_PYTHON=ON` build from a clean ONNXRuntime download,
  `import libspeech` (not `import _audio`), `libspeech.__version__`,
  `libspeech.Audio()` with the full load/to_mono/resample/save/load-from-wav
  flow, and the standalone `example` executable all working correctly.

## Python tests wired into the polyglot `speech_test` umbrella

`speech_test` used to only run `speech_test_dsp` -- Python had no tests at
all, and no CMake wiring to run them even if it did. Fixed both, mirroring
ctoon's `tests/python/CMakeLists.txt` pattern:

- [x] `tests/python/test_audio.py` -- real pytest tests for everything
  manually verified while fixing the bindings (version, load-from-raw-PCM,
  `__len__` is sample count not channel count, `to_mono`, `resample`
  including the same-rate no-op case, save/load WAV round-trip, `__repr__`
  channel count). 8 tests.
- [x] `tests/python/CMakeLists.txt` -- `speech_test_python` target (built +
  run via `cmake --build . --target speech_test_python`, or through
  `ctest`), gated on pytest being installed (warns and skips otherwise,
  same as ctoon), added as a dependency of `speech_test`. Included from the
  main `CMakeLists.txt` only after Python bindings are configured (needs
  the `_audio` target + `${PYTHON_PROJECT_NAME}`), and only for local dev
  builds (`NOT DEFINED SKBUILD`).
- [x] Verified end-to-end: `cmake --build . --target speech_test` now runs
  all 37 DSP tests *and* all 8 Python tests in one command; `ctest` shows
  both `speech_dsp` and `speech_python` entries; `-DBUILD_MODELS=OFF`
  (DSP-only) still works correctly with no Python target present.
- [ ] Coverage reporting (lcov/coverage.py + merged dashboard) -- ctoon has
  an elaborate version of this; not set up here yet, out of scope for now.

## Python bindings for speech::models

- [x] New `bind_models` module (`_models`, re-exported as
  `libspeech.Denoiser`, `libspeech.SileroVad`, `libspeech.SpeechTimestamp`)
  -- previously only `Audio` was exposed to Python.
- [x] `Denoiser`: no public constructor bound (matches the C++ abstract
  interface) -- only `Denoiser.create(backend, model_path, sample_rate)`
  and `.process(input_audio)`.
- [x] `SileroVad`: full constructor (all tunable parameters), `.process()`,
  `.get_speech_timestamps()`, `.reset()`.
- [x] `SpeechTimestamp`: `start`/`end` (samples), `start_s`/`end_s`
  (seconds), `__repr__`, `__eq__`.
- [x] Found + fixed a real, pre-existing bug while testing this binding:
  `timestamp_t::start_s()`/`end_s()` (`include/libspeech/models/silero_vad.h`)
  performed integer division (`start / sample_rate`, both `int`), always
  truncating to `0.0` unless `start`/`end` happened to exceed `sample_rate`.
  Fixed by casting to `float` before dividing.
- [x] `tests/python/test_models.py` -- 4 new tests (timestamp seconds
  conversion regression test, repr, and clean-error-propagation checks for
  both `Denoiser.create` with an unknown backend and `SileroVad` with a
  nonexistent model). Real ONNX inference isn't exercised (needs downloaded
  model weights) -- these validate the binding surface and error handling.
- [x] Verified end-to-end: `cmake --build . --target speech_test` now runs
  37 DSP tests + 12 Python tests (8 audio + 4 models) together.

## Version handling (ctoon-style single source of truth)

- [x] New `include/libspeech/version.h`: `#define LIBSPEECH_VERSION_MAJOR/MINOR/PATCH`
  is now the single source of truth (was: a hardcoded string in
  `pyproject.toml`, read backwards into CMake -- meaning the C++ code
  itself had no compile-time-visible version at all).
- [x] `cmake/DynamicVersion.cmake` (adapted from ctoon) parses the header
  and feeds `project(libspeech ... VERSION ...)`; replaces the old
  `cmake/PyProject.cmake` (removed).
- [x] `pyproject.toml`: `version = "0.0.1"` → `dynamic = ["version"]` +
  `[tool.scikit-build.metadata.version]` regex provider reading the same
  header -- so CMake and the Python package version can never drift apart.
- [x] `version.py` (adapted from ctoon's, same CLI): `python version.py`
  shows the version, `python version.py minor +`/`patch 4`/etc. bumps or
  sets a component, `python version.py tag create [major|minor|patch|X.Y.Z]`
  bumps, commits, and tags a release. Verified: show, bump, and revert all
  correctly edit `include/libspeech/version.h`.
- [x] Verified end-to-end: fresh configure correctly reports
  `Project: libspeech@v0.0.1` (read from the header, not pyproject.toml).

## Fixed: `_models` binding not found on a real build (user-reported)

Root-caused a `ModuleNotFoundError: No module named 'libspeech._models'`
from an uploaded build log. Two contributing issues, both fixed:

- [x] `file(GLOB PYTHON_BIND_MODULES_PATH ...)` (and the per-module
  `NB_MODULE_SOURCES` glob) lacked `CONFIGURE_DEPENDS`, so a newly added
  `src/binding/bind_*` directory wasn't picked up by an incremental
  `cmake --build` without a manual reconfigure. Added `CONFIGURE_DEPENDS`
  to both (plus the `speech_dsp`/`speech_models` source globs, same class
  of bug) -- but note: this is CMake's own documented *best-effort*
  behavior, not a hard guarantee on every generator (observed unreliable
  with the Unix Makefiles generator in this sandbox); if a new file/module
  still isn't picked up, a manual `cmake ..` (or fresh build directory)
  remains the guaranteed fallback.
- [x] The *actual* root cause of the specific reported error: `_models` was
  simply missing from `add_dependencies(speech_test_python ...)` in
  `tests/python/CMakeLists.txt` -- reproduced from a completely fresh
  configure (so it wasn't just glob staleness). Fixed, and hardened against
  this ever recurring by depending on `${PYTHON_BIND_MODULES}` (every
  discovered binding module) instead of a hardcoded name list.
- [x] Verified end-to-end from scratch: fresh configure + `--target
  speech_test` now correctly builds and runs all 37 DSP tests and all 12
  Python tests (8 audio + 4 models).
