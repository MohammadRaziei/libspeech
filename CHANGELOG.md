# Changelog

All notable changes to libspeech. The version lives in `include/libspeech/version.h`.
This file starts at 0.3.1; older history is in `git log`.

## Unreleased (since 0.3.1)

### Breaking
- **C++ API renamed to snake_case, STL style** (types, enumerators, functions, methods,
  constants). A few names changed beyond case because a type cannot share a method's name:
  `STFT::stft` → `stft::compute`, `Resample::resample` → `resample::process`,
  `Spectrogram` → `complex_spectrogram`, `DctII` → `dct_ii_plan`,
  `Denoiser::Create` → `denoiser::create`. Full old → new table: `docs/api-renames.md`.
  Python class names (`libspeech.STFT`, ...) are unchanged.
- **`window_type`** (`enum class`, `include/libspeech/dsp/window_type.h`) replaces AudioFlux's
  `WindowType`/`Window_*` in the public headers; no AudioFlux header is public any more.
- **`denoiser` is single-inheritance** (pimpl backends `facebook_denoiser`, `speechbrain_denoiser`);
  `process()`/`close()` are non-virtual in the base.
- **Default log level is now `warning`** (it used to print everything). Use
  `libspeech.set_log_level("debug")`, `speech::utils::set_log_level` or `LIBSPEECH_LOG`.
- **`istft` with an invalid `method` raises** (only 0/1, or `"weighted"`/`"plain"` in Python).
- **`nanobind>=2.0.0`** is required to build the Python bindings.

### Added
- Python `libspeech.stft(...)` / `libspeech.istft(...)` (librosa style, NumPy in/out), and
  `STFT.stft_complex`, `STFT.istft`, `FFT.fft`/`FFT.ifft` returning `complex64`.
- C++ `istft` (inverse real FFT, one-sided or full input, AudioFlux-compatible normalisation),
  `fft::forward_into/inverse_into/dct_into/idct_into`, `span<T>`/`frame_view<T>` and complex
  views (`as_complex`, `split_complex_span`, `complex_spectrogram::frames()`).
- Real-FFT fast path for real input up to 2^17 points.
- `Denoiser`: `close()`, `closed`, context manager, `sample_rate`, `num_threads`
  (default 1, 0 = automatic), NumPy overload, GIL released while running, `Denoiser("facebook")`
  with the default model, `denoiser(x)`.
- Logger with a public API (`set_log_level`, `get_log_level`, `LIBSPEECH_LOG`), output
  `time [level] (tag) message`; it no longer takes over `std::clog`.
- Type stubs: `py.typed` and one `.pyi` per compiled module, generated with
  `python -m nanobind.stubgen` in `src/bindings/python/CMakeLists.txt` and installed with the
  wheel.
- **`resample_quality`** (`best` default, `high`, `medium`, `fast`; Python
  `Resample(src, dst, "medium")`) trades passband width and alias rejection for speed. Python,
  44.1 → 16 kHz, one core, min of 60 runs, 10 s / 60 s of audio, in ms: best 2.5 / 15.4,
  high 1.9 / 11.9, medium 1.35 / 8.2, fast 0.77 / 4.7, against librosa 2.3 / 13.5, soxr MQ
  2.0 / 11.7, soxr LQ 1.8 / 10.5. Quality, measured with pure tones: aliases rejected by
  > 150 dB (best), 135 (high), 124 (medium), 107 (fast) at 12 kHz; passband loss at 7 kHz
  0.00 / 0.00 / 0.07 / 2.3 dB. soxr keeps a flatter passband close to 8 kHz (-0.02 dB at
  7.4 kHz against -1.1 dB for `best` and -2.0 dB for `medium`), so the levels are not an
  equal-quality comparison with soxr.
- **Denoiser backends `gtcrn` and `dpdfnet`** (`gtcrn_denoiser`, `dpdfnet_denoiser`;
  `Denoiser("gtcrn")`, `Denoiser("dpdfnet", model="dpdfnet4")`). Small streaming models that work
  on the STFT one frame at a time and carry their state, so any input length works. GTCRN (0.5 MB,
  16 kHz) and DPDFNet (`dpdfnet_baseline`, `dpdfnet2` (default), `dpdfnet4`, `dpdfnet8` at 16 kHz,
  `dpdfnet2_48khz_hr` at 48 kHz; pass `sample_rate=48000` for that one) are Ceva's / the GTCRN
  authors' public ONNX exports, hosted in the `models` release (each with a `.readme` and `.sha256`)
  and downloaded on first use. Checked against sherpa-onnx 1.13.8 on its
  9.8 s sample clip: max |difference| 1.5e-7 (float precision) for GTCRN and the four 16 kHz
  DPDFNet models, 1.4e-5 for the 48 kHz one. Two deliberate differences: the output always has the
  input's length (sherpa-onnx drops the last `len % hop` samples), and DPDFNet's 4-hop latency is
  compensated by flushing the model with silence, so its last 40 ms are real output instead of
  zeros. Single thread, 9.8 s of audio (not compared to anything else): GTCRN 0.8 s,
  `dpdfnet_baseline` 1.4 s, `dpdfnet2` 2.9 s, `dpdfnet4` 4.0 s, `dpdfnet8` 6.3 s. DPDFNet's
  optional attenuation limit is not exposed.
- `libspeech.available_providers()` and `libspeech.is_cuda_available()` (C++:
  `speech::models::available_providers/is_cuda_available`): the ONNX Runtime providers of the loaded
  build, and whether CUDA can really be created. Models still run on the CPU; selecting a provider
  for a session is not implemented yet.
- Test marker `models` (real-model tests, run by `cmake.yml` only: `pytest -m models`).

### Changed
- Resample output is numerically more accurate than before (up to ~6e-3 different from the
  old kernel). The default (`best`) is still ~8–18% slower than librosa/soxr on one core; use
  `resample_quality` to trade accuracy for speed.
- **Documentation site rebuilt like ctoon's**: `-DBUILD_DOCS=ON` + `--target speech_docs` builds the
  C++ reference (Doxygen, with doxygen-awesome-css), the Python reference (Sphinx, from the package
  staged in the build tree), a route index (`docs/supported_langs.json`) and Markdown pages
  (changelog, C++ API renames, contributing) into `<build>/docs/out/`; `speech_docs_cpp` and
  `speech_docs_python` build one part. Needs `pip install sphinx doxygen-awesome-css markdown-it-py`.
- The Python bindings' CMake code moved from the top-level `CMakeLists.txt` to
  `src/bindings/python/CMakeLists.txt`; `cmake/generate_stubs.cmake` is gone.

### Fixed
- **Windows: `SPEECH_LOG(ERROR)` messages were never printed.** `<windows.h>` defines `ERROR` as 0,
  so the macro mapped it to the trace level. The severity is now pasted into a name
  (`SPEECH_LOG_LEVEL_ERROR`) instead of being macro-expanded. Regression test on every platform
  (`log.error_severity_is_not_hijacked_by_a_windows_style_error_macro`).
- **Windows wheel build: stub generation failed** with `No module named 'libspeech._about'`.
  Multi-config generators (Visual Studio, Xcode, Ninja Multi-Config) append `Release/` to a plain
  `LIBRARY_OUTPUT_DIRECTORY`, so the staged package was missing its modules; the staging directory is
  now a generator expression, which switches that suffix off.
- `audio_callback` initialised a local `output` from itself (it shadowed the `output` parameter),
  which is undefined behaviour (MSVC C4700).
- `Denoiser.create(...)` failed with "Unable to convert function return value".
- Destroying a `Denoiser` aborted the process (`free(): invalid pointer`, multiple-inheritance
  pointer offset).
- **Facebook denoiser silently dropped everything after ~32 s of input** (the exported graph
  works on a fixed window); longer input is now processed in overlapping, cross-faded windows
  and the output has the input's length.
- **The Facebook models (`facebook-denoiser-dns48.onnx`, `-dns64.onnx`, `-master64.onnx`) take 4 to 7 s per call whatever the input length.** The graph
  was traced with a 512000-sample example, so its `valid_length` was baked in as the constant
  512085 and every input was zero-padded to ~32 s. A re-published model computes `valid_length`
  from the input length inside the graph (same weights; 38 nodes replace the constant), measured
  on one core, dns64: 0.1 s → 0.03 s, 1 s → 0.23 s, 5 s → 1.0 s, 32 s → ~7 s (dns48 1 s → 0.14 s,
  master64 1 s → 0.17 s). All three had the same baked constant and are patched the same way. Output equals the old
  model for 32 s inputs and differs by ~1e-3 (peak, at the tail) for 1 to 5 s inputs; not
  compared against the PyTorch reference or on real speech. Existing installs keep the old file
  in `~/.libspeech/` until it is deleted.
- `tests/python/test_logging.py` no longer relies on AudioFlux rejecting `FFT(99)` to produce an
  error line (it failed on Windows CI).

### Known issues
- Not measured or run on Windows/macOS/ARM or with multiple cores in this round of work; check CI.
- `STFT` outside n_fft 16..8192 uses the AudioFlux fallback, which is mutex-locked; `FFT` is not
  thread-safe.
