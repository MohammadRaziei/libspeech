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
- Test marker `models` (real-model tests, run by `cmake.yml` only: `pytest -m models`).

### Changed
- Resample output is numerically more accurate than before (up to ~6e-3 different from the
  old kernel). The default (`best`) is still ~8–18% slower than librosa/soxr on one core; use
  `resample_quality` to trade accuracy for speed.
- The Python bindings' CMake code moved from the top-level `CMakeLists.txt` to
  `src/bindings/python/CMakeLists.txt`; `cmake/generate_stubs.cmake` is gone.

### Fixed
- `Denoiser.create(...)` failed with "Unable to convert function return value".
- Destroying a `Denoiser` aborted the process (`free(): invalid pointer`, multiple-inheritance
  pointer offset).
- **Facebook denoiser silently dropped everything after ~32 s of input** (the exported graph
  works on a fixed window); longer input is now processed in overlapping, cross-faded windows
  and the output has the input's length.
- **`facebook-denoiser-dns64.onnx` takes ~7 s per call whatever the input length.** The graph
  was traced with a 512000-sample example, so its `valid_length` was baked in as the constant
  512085 and every input was zero-padded to ~32 s. A re-published model computes `valid_length`
  from the input length inside the graph (same weights; 38 nodes replace the constant), measured
  on one core: 0.1 s → 0.03 s, 1 s → 0.23 s, 5 s → 1.0 s, 32 s → ~7 s. Output equals the old
  model for 32 s inputs and differs by ~1e-3 (peak, at the tail) for 1 to 5 s inputs; not
  compared against the PyTorch reference or on real speech. Existing installs keep the old file
  in `~/.libspeech/` until it is deleted.
- `tests/python/test_logging.py` no longer relies on AudioFlux rejecting `FFT(99)` to produce an
  error line (it failed on Windows CI).

### Known issues
- Not measured or run on Windows/macOS/ARM or with multiple cores in this round of work; check CI.
- `STFT` outside n_fft 16..8192 uses the AudioFlux fallback, which is mutex-locked; `FFT` is not
  thread-safe.
