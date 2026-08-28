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
- [x] Establish AudioFlux vendoring workflow: copy only the `.c`/`.h` files a ported operator needs into `src/dsp/vendor/audioflux/`, patch in place if needed, log every patch in `/audioflux_issues.md`
- [ ] Remove remaining unused submodules once nothing depends on them (`src/third_party/audioflux` full submodule — keep until every needed DSP op is vendored)
- [ ] Decide fate of `src/third_party/indicators`, `dr_libs`, `miniaudio` (still used by `Audio`/CLI progress bars — keep for now, revisit)

## DSP layer (`speech::dsp`, flat/non-virtual, vendored AudioFlux C under the hood)

- [x] **Resample** — `include/libspeech/dsp/resample.h`, `src/dsp/resample.cpp`
  - [x] TDD tests (`tests/dsp/test_resample.cpp`), 5/5 passing
  - [x] Found + fixed AudioFlux bug: `sourceRate==targetRate` corrupts ratio (Issue 1)
  - [x] Found + fixed amplitude bug: `isScale=1` energy-scales output ~1.66x on downsample (Issue 2)
  - [x] Documented both in `/audioflux_issues.md`
  - [ ] Wire into main CMake build (`tests/dsp/CMakeLists.txt` or equivalent), not just standalone compile
- [x] **Window functions** — `include/libspeech/dsp/window.h`, `src/dsp/window.cpp` (flat free-function, not a class)
  - [x] TDD tests (`tests/dsp/test_window.cpp`), 6/6 passing
- [x] **FFT / IFFT** — `include/libspeech/dsp/fft.h`, `src/dsp/fft.cpp`
  - [x] TDD tests (`tests/dsp/test_fft.cpp`), 8/8 passing (impulse response, sine-bin energy, forward/inverse round-trip)
- [x] **DCT / IDCT** — implemented as `FFT::dct()`/`FFT::idct()` (AudioFlux computes DCT via the same FFT object, so no separate class)
  - [x] Found + documented AudioFlux footgun: `fftObj_idct()` mutates its input in place (Issue 3) -- worked around with an internal copy in our wrapper
  - [x] Round-trip + no-mutation regression tests passing
- [ ] **STFT / spectrogram** (composes Window + FFT — next up)
- [ ] **MFCC** (composes STFT + DCT — only after STFT is solid)
- [ ] Decide: do we need `filterDesign_fir`/`filterDesign_iir` as a public `speech::dsp` API, or is it purely an internal dependency of Resample?
- [ ] Benchmark our DSP path vs. sherpa-onnx on at least one op (validates the "why libspeech" question from earlier)

## Models layer (`speech::models`, small interfaces where backends compete)

- [x] `Denoiser` interface (`include/libspeech/models/denoiser.h`) with `Create(backend, url, sample_rate)` factory
- [x] `FacebookDenoiser` and `SpeechBrainDenoiser` implement `Denoiser`
- [ ] Tests for `Denoiser` factory + both backends (TDD, currently untested)
- [ ] Revisit `BaseModel`/`ONNXModel` for cross-platform issues (e.g. `getenv("HOME")` breaks on Windows despite README claiming cross-platform support)
- [ ] `SileroVad` — leave as concrete class (no interface) unless/until a second VAD backend actually exists (YAGNI)

## Build & packaging

- [ ] Verify full CMake build end-to-end (blocked in this sandbox by `libcurl4-openssl-dev` install failure — needs verification on a real machine)
- [ ] `pip install` end-to-end smoke test (core promise of the project: zero system deps)
- [ ] CI update: drop googletest/aixlog submodule steps, add utest.h-based test target

## Documentation

- [x] `/audioflux_issues.md` — running log of every AudioFlux bug/quirk found + fixed, PR-ready
- [x] `src/dsp/vendor/audioflux/README.md` — explains why vendored instead of submoduled
- [ ] Update root `README.md` Quick Start to match actual current API (it currently references classes/methods that don't exist yet — `AudioProcessor`, `extract_features`, etc.)
- [ ] `CONTRIBUTING.md` (referenced by README but missing)
- [ ] Doxygen-style comments on public headers (`Audio`, `BaseModel`, ...)
