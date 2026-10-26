# Changelog

All notable changes to libspeech. The version lives in `include/libspeech/version.h`.
This file starts at 0.3.1; older history is in `git log`.

## Unreleased

### Added
- **Streaming `dpdfnet` denoiser.** `Denoiser("dpdfnet").stream()` works like the GTCRN one: any chunk size, output equal to
  `process()` on the whole signal (tail included), exactly as many samples out as in. The stream compensates the
  network's four-hop delay like `process()` does, so the latency is `n_fft - 1 + 4 hops`: 959 samples (60 ms) for the
  16 kHz models, 2879 (60 ms) for `dpdfnet2_48khz_hr`. Tested with the real `dpdfnet2`, `dpdfnet_baseline`, `dpdfnet4`, `dpdfnet8` and the 48 kHz model (chunks from 1
  to 10^6 samples, short inputs, interleaved streams, `close()` while streaming). ADR-0019.
- **RMS normalizer.** `speech::dsp::rms_normalize()` / `scale_to_rms()` / `rms()` / `rms_dbfs()` (C++),
  `speech::io::audio::normalize_rms()`, and in Python `libspeech.rms_normalize(x, target_dbfs=-20.0, peak_ceiling=0.99)`,
  `libspeech.rms_dbfs(x)` and `Audio.normalize_rms(...)`: scale a signal to a target RMS level in dBFS with one gain
  (so shape and channel balance are kept), lowering the gain if the peak would pass the ceiling so nothing clips. Silent
  or non-finite input is returned unchanged. ADR-0020.
- **Speech-weighted RMS normalization (VAD probabilities with a hard or a soft threshold).** `libspeech.speech_rms_normalize(x, mode="hard"|"soft", threshold=None, ...)`
  runs `SileroVad` and measures the RMS only where the speech probabilities say there is speech, so a recording with long
  pauses is no longer raised too much. `hard`: weight 1 when `p >= threshold`, else 0. `soft`: `(p - threshold) / (1 - threshold)`
  clamped to [0, 1]. Building blocks: C++ `speech::dsp::speech_weights()`, `weighted_rms()`, `rms_normalize_weighted()` and Python
  `libspeech.speech_weights()`, `libspeech.rms_normalize_weighted()`. One gain for the whole signal, the peak ceiling still wins,
  and a signal with no speech is returned unchanged. 1-D only. ADR-0022.
- **`docs/adr/` (18 architecture decision records) and `docs/agent.md`** (the guide for an agent or developer: when to consult, how to build
  and test through CMake, working rules, delivery). The ADRs were reconstructed from the git history, the changelog and the code.
- **The native log reaches Python's `logging`, and the Python logger controls it.** Every line libspeech's C++ code
  logs arrives in `logging.getLogger("libspeech")` with no setup, and that logger's level decides which native
  lines are even built: `setLevel` on it or on the root logger, `logging.basicConfig(level=...)`, `dictConfig`,
  pytest's `caplog.set_level`, `libspeech.set_log_level()` and `LIBSPEECH_LOG` all end up at the same place. The
  native level is the smallest one that lets through what the logger keeps, never left at `trace` for Python to
  filter. New: `speech::utils::set_log_callback()` / `set_log_file()` (C++) and `libspeech.set_log_callback()` /
  `set_log_file()` (Python); one output is active at a time, so setting one replaces the bridge. Records carry the
  native tag or function as `record.where`. `libspeech.set_log_level()` adds a stderr handler when the
  application configured no logging, so asking for `debug` still prints. Output now goes through AixLog's own
  sinks; its severity names show as `Debug`/`Warn`/`Error` in the native stderr/file format, and in Python the
  usual `DEBUG`/`WARNING`/`ERROR`.
- **Log-mel filterbank features.** `mfcc::compute_log_mel_flat()` (C++) and `MFCC.log_mel(x)` (Python) return the
  `(num_frames, num_mel_filters)` log-mel energies the MFCC pipeline already computed internally, without the DCT
  (the usual "fbank" model input). MFCCs are exactly the orthonormal DCT-II of these rows; tests check that against
  an independent DCT (`dct_ii_plan` in C++, `scipy.fft.dct` in Python).

- **Streaming denoiser (GTCRN).** `denoiser.create_stream()` (C++) / `Denoiser.stream()` (Python) start a
  `denoiser_stream` / `DenoiserStream`: `feed(chunk)` takes audio in chunks of any size and returns the enhanced samples
  that became final, `finish()` returns the tail, so the total output is exactly as long as the input. Latency is at
  most 511 samples (32 ms at 16 kHz). The result is bit-identical to `process()` on the whole signal, however the
  audio is chunked (tested with the real model, chunks from 1 sample to 10^6). Each stream has its own recurrent
  state; streams of one denoiser are independent and a running stream survives `close()` of the denoiser. Backends
  that cannot stream (`facebook`, `speechbrain`) report `supports_streaming == False` and `stream()` raises
  a clear error. The incremental STFT/ISTFT behind it (`detail::stft_streamer`, same frames and arithmetic as
  `stft_apply_istft`, bounded memory, supports DPDFNet-style `shift`) is generic.

- **Streaming `SileroVad`.** `feed(chunk)`, `pop_speech_timestamps()`, `finish()`, plus `in_speech` and `samples_fed`
  (C++ and Python) let audio arrive in chunks of any size (microphone callback, socket) and report each speech
  segment as soon as the silence that ends it has been seen. `process()` is now implemented as `reset()` +
  `feed(all)` + `finish()`, so every chunking gives exactly the segments the batch call gives (tested with
  chunk sizes from 1 sample to 4000, against the real model).

- **Delta / delta-delta features.** `speech::dsp::delta()` / `delta_into()` (C++, `dsp/delta.h`) and
  `libspeech.delta(features, half_window=2, order=1, edge="replicate")` (Python) compute the least-squares slope of each
  feature column over a `2*half_window+1` frame window (the HTK/Kaldi regression formula); `order=2` applies it twice
  (delta-delta). `edge="replicate"` repeats the edge frame (HTK, Kaldi, torchaudio); `edge="interp"` matches
  `librosa.feature.delta(width=2*half_window+1, mode="interp")` on the whole matrix. Note that librosa's own `order=2`
  is a different quantity (second derivative of a quadratic fit), not delta applied twice.

- **Free-threaded Python wheels (`cp313t`, `cp314t`, `cp315t`).** Until now `pip install libspeech` on a free-threaded
  interpreter found no wheel (the `cp312-abi3` wheel does not exist for them: abi3 is not available on free-threaded
  CPython) and fell back to a source build. The extension modules are built with nanobind's `FREE_THREADED` flag, so
  importing libspeech keeps the GIL off. cibuildwheel 4 no longer builds 3.13t, so `cp313t` has its own
  cibuildwheel 3.x step in `wheels.yml`.
  `tests/python/test_free_threading.py` checks that `import libspeech` does not re-enable the GIL (skips on
  regular interpreters).

- **Audio <-> array interop.** `Audio` has `channels`, `shape` (`(channels, samples)`), `__array__` / `numpy()`
  (a float32 `(channels, samples)` view that shares the Audio's memory, no copy), `__dlpack__` /
  `__dlpack_device__` (`np.from_dlpack(audio)`, `torch.from_dlpack(audio)`, ...; NumPy gets a read-only view), and
  can be built from an array: `Audio(array, sample_rate)`, `Audio.from_numpy(...)`, `Audio.from_dlpack(...)`
  (1-D mono or 2-D `(channels, samples)`, copied once; other dtypes are converted). New C++
  `audio::shared_samples()`. `to_numpy(channel)` is unchanged (a 1-D copy).
- **`SileroVad` speech probability and threshold.** `speech_probabilities()` returns the model's speech probability
  (0..1) for every complete window since `reset()` (`window_samples` samples each; a segment starts at the first window
  that reaches the threshold), `pop_speech_probabilities()` returns the new ones while streaming, and
  `threshold()` / `set_threshold(value)` read and change the threshold (`0 < value <= 1`, else `ValueError` /
  `std::invalid_argument`). A new threshold applies from the next window on and survives `reset()`.
- **Tests and benchmarks.** `tests/python/test_denoiser_threading.py` (marked `models`): one `Denoiser` shared by several
  threads, with `process()` and with streams, gives each thread its serial result, and `process()` / `feed()` release the
  GIL. `log_mel` and `delta` are in the Python benchmark suite (`benchmarks/python/`).
- **`labs/`**: experiments with their notebooks and scripts, one folder per story. `labs/quantize_denoiser/` has int8 quantization tools for the Facebook denoisers (`onnx_ptq.py`: static int8 for the convolutions, dynamic int8 for the LSTM, per-node sensitivity, evaluation; `distill_qat.py`: teacher-student weight-quantization-aware fine-tuning in PyTorch) and a Colab notebook. Its tests are in `labs/quantize_denoiser/tests/` (run by hand; not part of CMake or CI).

### Fixed
- **`SileroVad` with a threshold below 0.16 never ended a segment.** The silence level was `threshold - 0.15`, which is
  negative there, so no window counted as silence and the segment ran to the end of the audio. It is now
  `max(threshold - 0.15, 0.01)`, as in the reference Silero code; thresholds of 0.16 and above behave exactly as before.
- **`logging.disable()` and a second stderr handler.** `logging.disable(level)` now reaches the native side (the native
  level follows what the logger lets through, so libspeech no longer builds lines Python would drop). The stderr handler
  that `set_log_level()` adds when logging is unconfigured stays silent once the application configures logging
  afterwards (`basicConfig`, a handler on the root or the library's logger), so a line is no longer printed twice.
- **`Denoiser("speechbrain")` returned audio about 15x louder than its input.** The SepFormer was trained with a
  scale-invariant loss (SI-SNR), so the absolute level of its output is arbitrary: on real speech it came out at a
  peak of 6 to 27 with 14 to 43% of the samples beyond full scale, i.e. clipped as soon as it was saved as 16-bit
  PCM. The output is now scaled to the RMS level of the input and lowered if that would push its peak above
  max(0.99, input peak); silent or non-finite input is left alone. `facebook`, `gtcrn` and `dpdfnet` were already at
  the input's level and are unchanged.
- `SileroVad.process()` failed on every released model: the code looked up the output tensor as `state_n` (the models
  call it `stateN`) and always fed an `sr` input (`silero-vad-half` has none). The tensor names are now read from the
  loaded model, and a model with another signature is rejected with an error instead of being driven blindly.
  Found by running the README example; nothing exercised `process()` before.
- Loading a file into an `Audio` that already held audio with the same channel count kept the old length (a 4000
  sample file loaded after a 16000 sample one reported 16000 samples). The samples now live in one planar buffer that
  every load replaces as a whole (a failed load leaves the old audio untouched; views keep the buffer they were made
  from). `Audio.load` of channels with different lengths now fails (it used to accept them and later read out of
  bounds), a raw `.bin` on a fresh `Audio` fails cleanly (it divided by zero), and `Audio` assignment no longer breaks
  on self-assignment.

### Changed
- **Breaking: class getters lost the `get_` prefix.** `SileroVad.get_speech_timestamps()` is now `speech_timestamps()` and
  `get_threshold()` is `threshold()` (C++ `silero_vad_model` and Python; `set_threshold()` is unchanged, the C++ data member is now
  `threshold_`). No aliases are kept. Free functions keep their names (`get_log_level`, `get_temp_directory`,
  `get_default_model_cache_dir`), and so do other projects' names (httpp's `get_lib_dir`, CMake commands, `src/third_party`).
  See `docs/api-renames.md`.
- README rewritten: shorter, aimed at someone installing the package for the first time. The Python examples in it
  are run by `tests/python/test_readme_examples.py` (the model block under `-m models`). Coverage targets, the SIMD
  layout and the test-target tree moved to `CONTRIBUTING.md`.
- The `test` extra installs librosa/soundfile/scipy only on Python < 3.13 (they have no wheels for every free-threaded
  build, e.g. no scipy for `cp313t`, and there is no environment marker for free-threading). Where they are absent,
  `test_librosa_compat.py` skips itself.
- Free-threaded wheel builds set `PIP_PREFER_BINARY=1` (cibuildwheel override), so the wheel tests take numpy 2.4.6
  (which has `cp313t` wheels) instead of compiling the numpy 2.5 sdist, which has none for 3.13t.
- Build requirements: `nanobind>=2.2.0` (first version with `FREE_THREADED`), `scikit-build-core>=0.10.7`.

### Removed
- `libspeech.speech_audio_py`: a stale copy of `speech_io_py` (left over from the `bind_audio` -> `bind_io` rename,
  without `Audio.to_numpy`). Nothing in the package imported it, and importing it made nanobind warn that `Audio` was
  already registered. Use `libspeech.Audio`.

## 0.4.0 (2026-10-17)

Needs **httpp 0.7.x** (`httpp>=0.7,<0.8`, in `pyproject.toml`): the download lock and the SHA-256 functions used
here are not in 0.6.x, and httpp's API is still changing, so the range is deliberately narrow.

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

### Changed
- **License is now Apache-2.0** (was declared as MIT in the README/pyproject, while the `LICENSE` file actually held
  pybind11's BSD-3 text). `LICENSE` is the standard Apache-2.0 text, `NOTICE` names the third-party code and says
  that the downloaded models keep their own licenses. Copies of libspeech already distributed under MIT stay MIT.

### Added
- **Verified model downloads.** A model now always comes with two small files next to it, and all three
  are downloaded: `<model>.onnx.sha256` (one line, `<sha256>  <file name>`, as `sha256sum` writes it) and
  `<model>.onnx.readme` (what the model is, who made it, license, provenance). The checksum file is fetched
  **first** and must be exactly that one line (an HTML error page is rejected); then the readme (so a missing
  one fails before the big download); then the model, which httpp checks against the checksum and deletes if it
  does not match. The checksum and readme are installed next to the model only after it arrived and matched,
  so a failed or interrupted download no longer leaves a broken file behind that later calls would trust
  (`download_file` skipped any existing file). A cached model is hashed on every load and downloaded again if
  it does not match; a model cached by an older version just gets its two files. A bare model name (the
  `models` release) must have all three; an explicit URL may lack the sidecar files and is then used unverified
  with a warning (a malformed checksum file is still an error).
- **Several processes, one download.** The lock is httpp's: `httpp::download` always takes `<file>.lock/`,
  whose `alive` file the downloading process keeps rewriting. Another process (or thread) that wants the same
  model waits and then finds the finished file instead of downloading it again; a lock that stops being
  updated belongs to a dead process and is taken over. A download that was killed is resumed from its `.part`
  file and verified against the checksum. The checksum and readme are fetched under per-download temporary
  names (`*.<tag>.new`), so a failed refresh (`ignore_cache`) leaves the old cache as it was. A cache hit
  never takes the lock.
- **`ignore_cache`**: `Denoiser(..., ignore_cache=True)`, `SileroVad(..., ignore_cache=True)`,
  `libspeech.fetch_model(..., ignore_cache=True)`; C++ `denoiser::create(..., ignore_cache)`, the model
  constructors and `fetch_options::ignore_cache`. Downloads everything again; if that fails the old cache is untouched.
- **Readme API**: `Denoiser.readme()`, `SileroVad.readme()`, `libspeech.model_readme(name)`; C++
  `denoiser::readme()`, `base_model::readme()`, `speech::models::model_readme(name)`. Also `model_path`
  and `sha256` on loaded models, `libspeech.fetch_model(name)` (returns the verified path without loading
  the model) and C++ `speech::models::fetch_model` / `fetch_options` / `parse_checksum`.
- **SHA-256**: `libspeech.sha256(bytes)`, `libspeech.sha256_file(path)`, C++ `speech::utils::sha256_hex` /
  `sha256_file` (chunked, any file size). The hashing itself is httpp's (`httpp::sha256_hex`), which also
  verifies the downloads; no SHA-256 code is vendored.
- **`data/models.csv`**: every model in the `models` release (name, file, backend, sample rate, size, sha256,
  license, `commercial_use`, source, notes). A test checks it against the release's `.sha256` files.
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
- `benchmarks/profile/`: profiling README and C++ timing scripts for DCT/FFT (2026-10-05).
- DSP loops link `OpenMP_CXX`; the README documents the SIMD backends and the architecture-agnostic
  vectorization logic (2026-10-02).

### Changed
- **httpp 0.7.x is required** (see above); libspeech no longer has its own download lock or SHA-256 code.
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
- Wheels: cibuildwheel 3.4.1 → 4.2.0 with explicit CPython version constraints; delvewheel repair is
  disabled on Windows (it caused a missing `speech.dll`); the test dependencies exclude Intel macOS and
  Python 3.15+ (numba → llvmlite wheel availability).
- The Python bindings bake the runtime library file names (ONNX Runtime, httpp) into `_about` at build time
  instead of guessing them at import time (which failed because `dlopen` ignores the RPATH).
- Benchmarks import NumPy in isolation, so the timings are fair.

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
- The real-FFT STFT path takes its analysis window from `window_calFFTWindow()`, the same function the
  vendored AudioFlux uses, instead of reading it out of the opaque `stftObj_`; regression test for every
  supported window type (`tests/dsp/test_fast_paths.cpp`).
- Python binding memory leak of `Denoiser` (the C++ object is held through a `shared_ptr` now); `close()`
  frees the model's memory explicitly.

### Known issues
- Not run on Windows/macOS/ARM, or with several cores, while this version was written (Linux, x86-64 only);
  the new download code has never been compiled with MSVC. Check CI.
- GPU is not used: models run on the CPU (`available_providers()` / `is_cuda_available()` only report).
- The Facebook denoiser models are licensed CC-BY-NC-4.0 (non-commercial); `data/models.csv` says so
  (`commercial_use=no`), and so does the model's `.readme`.
- A process killed during a download leaves its `*.new` files in the cache directory (harmless, not reused).
  If two processes take over the same dead lock at the same moment they can write the same `.part` file;
  the checksum catches the damage and the download fails instead of installing it.
- `STFT` outside n_fft 16..8192 uses the AudioFlux fallback, which is mutex-locked; `FFT` is not
  thread-safe.

## 0.3.1 (2026-10-01; the version was bumped, no git tag)

### Added
- Python benchmark suite (`benchmarks/`) comparing libspeech with librosa, audioflux, scipy and soundfile for
  time, memory and install size, with a standalone HTML report (`report/`); the CMake targets skip Python when
  there is no interpreter.
- `benchmarks/README.md` documents the measured results and the optimized paths: flat spectrogram, block-wise
  MFCC with sparse mel filters, AVX2+FMA resampling dispatch, memory-efficient `load`/`to_numpy`.

### Changed
- README: table of contents, dependency text now names `httpp` (not httplib + Mbed TLS), the STFT/OpenMP
  benchmark explanation separates single-core from multi-core results; the benchmarking section was removed
  from the README (the raw data is kept locally).

## 0.3.0 (2026-09-29)

### Added
- Build options for documentation and coverage (`-DBUILD_DOCS=ON`, `-DBUILD_COVERAGE=ON`); coverage of all
  components is merged into one `speech_coverage_total` report; CI builds with `BUILD_PYTHON=ON` so the tests
  are installed and reported.

### Removed
- The `dr_libs` submodule: `miniaudio` already embeds `dr_wav`, `dr_mp3` and `dr_flac`.
- The command-line tool, and the unused Python files `libspeech/core.py` and `libspeech/__main__.py`
  (0 % coverage, leftovers of an earlier split).

## 0.2.0 (2026-09-27)

### Fixed
- The CPython 3.12 wheel got the wrong ABI tag: the `CIBW_ENVIRONMENT_CP312` variable did nothing
  (cibuildwheel ignores Python-version suffixes) and was removed; `tool.cibuildwheel.overrides` now forces
  `cp312-abi3`.
- Clang's `-Wc23-extensions` warning caused by the vendored miniaudio is silenced.

## 0.1.1 (2026-09-26)

### Fixed
- Linux wheels ship and load ONNX Runtime under its SONAME (`libonnxruntime.so.1`) instead of the
  fully-versioned file name; CMake's `install()` followed the symlinks and shipped duplicate ~20 MB copies.

## 0.1.0 (2026-09-26)

First tagged release. Everything from 2026-08-27 up to the first release:

### Added
- `speech::dsp` as a standalone static library (`speech_dsp`), a direct DCT-II implementation, and an STFT
  of its own; AudioFlux is no longer a submodule (a subset of its C sources lives in `src/third_party/`,
  moved there from `src/vendor/`); `fftObj_idct()`'s input mutation is documented; `resampleObj_setSamplate`
  handles identity resamples.
- `BUILD_MODELS` (replaces `SPEECH_DSP_ONLY`) to build the DSP layer alone, and a unit-test build option.
- Python bindings for the speech models (`bind_models`), Python tests, version handling
  (`include/libspeech/version.h`), `CONTRIBUTING.md`, a standalone benchmarking project for STFT.
- Model downloads use `httpp` (HTTP client with a progress bar), replacing httplib, Mbed TLS and indicators.
- CI: a matrix build for CMake and an orchestrator for the wheels and the release.

### Fixed
- Windows/MSVC: compilation (`NOMINMAX`, `M_PI`, `setenv`), AudioFlux and ONNX Runtime compatibility,
  public API symbols exported from the DLL, static library members forced into `speech.dll`, ONNX Runtime and
  httpp DLLs copied beside executables, `speech.dll` installed in the wheel, DLL load failures diagnosed.
- macOS: arm64 suffix for ONNX Runtime, universal2 ONNX Runtime and dylib names, deployment target 13.3,
  wheel repair.
- Wheels: `$ORIGIN` RPATH and the ONNX Runtime SONAME shipped; musllinux and win32 wheels are skipped (ONNX
  Runtime is glibc-only); the runtime library file names are baked into `_about`; httpp is required
  unconditionally and loaded from pip instead of being bundled; the `pefile` dependency was replaced by an
  inline PE import-table parser and then removed; Python bindings consolidated with explicit install rules for
  the nested package.

## Before 0.1.0 (2025-03-14 to 2025-03-31)

The first prototype, in the order of the commits:
- Audio I/O with miniaudio and dr_libs, a `speech` namespace for `Audio`, a fixed `play()`, Python audio
  bindings, `seconds`, a format function and `operator=`.
- ONNX Runtime support with a base model class, model download with a progress bar (indicators, then a
  generated progress-bar source), and Silero VAD (fixed on 2025-03-17).
- AudioFlux (first as a submodule, then static) and `resample`; glog replaced by AixLog.
- A GitHub workflow, a different way to install onnxruntime for CPython, a simple HTML test, README and the
  libspeech logo (many iterations).
