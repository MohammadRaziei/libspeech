# libspeech: hand-off to the next chat (2026-10-21)

Read first: `docs/agent.md` (rules, set-up, pitfalls), `checklist.md` (state), `docs/adr/README.md` (decisions).
Reports to the owner are in **Persian**; everything written in the repository is **English** with no em dashes.
Owner's standing preferences: no diff or patch files (send the whole project as a zip without `.git`, build trees and caches
after every step); build and test through CMake targets and `ctest`; run `ruff` on the Python code (no lint errors).

## What was done in this session

1. **DPDFNet streaming** (`Denoiser("dpdfnet").stream()`), ADR-0019. `dpdfnet_stream` in `src/models/dpdfnet_denoiser.cpp`;
   shared `step()`; model held by `shared_ptr`. Latency `n_fft - 1 + 4 hops`: 959 samples (16 kHz models), 2879 (48 kHz).
   Output equals `process()` for any chunking (tests in `tests/python/test_dpdfnet_streaming.py`, `models` marker).
2. **RMS normalizer**, ADR-0020. `include/libspeech/dsp/normalize.h`, `src/dsp/normalize.cpp`; `Audio::normalize_rms`;
   Python `libspeech.rms_normalize`, `libspeech.rms_dbfs`, `Audio.normalize_rms`. One gain for the whole signal and all channels,
   peak ceiling wins over the target, silent or non-finite input unchanged. Tests: `tests/dsp/test_normalize.cpp` (14),
   `tests/python/test_normalize.py` (28, registered as `python_normalize`).
3. **Demo re-run** (generator from the owner's `libspeech-demo-generator.zip`, 4 release samples): VAD and GTCRN streaming equal
   one-shot on all 4; RTF on one core: gtcrn about 0.05 to 0.07, dpdfnet about 0.17, facebook about 0.2, speechbrain about 3.0 to 3.4.
   `alex-noisy` speechbrain output has RMS 0.57x of the input because of the 0.99 peak ceiling (ADR-0009), cause on that clip not investigated.
   Delivered as `libspeech-demo.zip`. The demo does not check DPDFNet streaming yet.
4. **Speed and quantization study**, ADR-0021: libspeech is at raw ONNX Runtime speed on all four backends (one core); no code change.
   Dynamic int8 recipes with a real gain: facebook LSTM-only 1.87x at 47.3 dB, speechbrain MatMul+Gemm 1.72x at 33.5 dB.
   Delivered as `libspeech-quantized.zip` (models, `make_quantized.py`, `listen/` samples, README). **Nothing shipped, no API change.**
5. Docs: CHANGELOG (Unreleased), `checklist.md` ("Performance and quantization"), ADR-0019, 0020, 0021 and the ADR index,
   `docs/agent.md` (fresh-sandbox set-up, new pitfalls), `docs/python/source/dsp.rst`, README feature line.

## Verified in this session (one core, through CMake)

- `ctest` in `build-py`: 16 of 16 passed. `speech_test_cpp_dsp`: 124 tests passed. `speech_test_python_models_e2e`: 82 passed.
- `ruff check . --exclude 'build-*'` clean.
- Mutation checks: normalizer (no peak ceiling, wrong gain, silence not skipped) and DPDFNet stream (shift set to 0) each made the
  tests fail; the code was restored and re-verified.

## NOT run or not verified

Windows, macOS, OpenMP, multi-core (`num_threads=0` never measured), microphone or socket streaming, DPDFNet streaming speed,
`dpdfnet_baseline` / `dpdfnet4` / `dpdfnet8` streaming, the Doxygen and Sphinx build (tools not installed), a multi-threaded Python
test, any listening test of the int8 models, a newer ONNX Runtime than the bundled 1.21.0.

## Open decisions for the owner (ask first, format in `docs/agent.md` section 1)

1. Ship int8 models and an API to select them (needs a listening test, release files with checksums, ADR-0005).
2. Default `num_threads` (1 now); needs a multi-core measurement.
3. Streaming (running RMS) normalizer: cumulative or sliding window, gain smoothing, behavior in the first seconds; cannot equal
   the one-shot result. Also peak and LUFS normalization.
4. Streaming for `facebook` and `speechbrain` (not frame-based, ADR-0008) and GPU provider selection: unchanged, still open.

## Practical notes

- A fresh sandbox has no tools: follow "Fresh sandbox set-up" in `docs/agent.md` (httpp from PyPI, not from git).
- `build-py` is not in the zip; the first configure downloads ONNX Runtime and takes a while; models download on first use.
- Local model files: pass an absolute path or `file://` path; a bare file name means "from the models release".
- Do not edit a source while a build runs (stale object files), see `docs/agent.md` section 6.
- `match_level` (SpeechBrain level matching) still has its own copy of the gain logic because `speech_models` does not link `speech_dsp`.
