# libspeech benchmarks

Two independent suites, both driven by CMake:

| Suite | Directory | What it measures |
|---|---|---|
| C++ | `cpp/` | `speech::dsp::STFT` in native C++ (OpenMP on/off) |
| Python | `python/` + `report/` | libspeech vs librosa, audioflux, scipy, soundfile: time, peak memory, install size, rendered into one HTML report |

## Running the Python suite

```bash
cmake -S benchmarks -B build-bench -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-bench --target libspeech_benchmarks_python
# -> benchmarks/results/report.html
```

- The venv is created at **configure** time (`build-bench/python/venv`). The first configure is slow (minutes) because libspeech is built from source there. Reconfiguring does not reinstall; delete `build-bench/python/.venv_ready` to force it.
- libspeech is installed from GitHub by default, never from this checkout. To benchmark local changes:
  `-DLIBSPEECH_BENCH_GITHUB_URL=/path/to/checkout`.
- `-DLIBSPEECH_BENCH_REPEATS=N` sets timed repeats per cell (default 7, best-of-N is reported).
- `-DLIBSPEECH_BENCH_RESULTS_DIR=...` moves where `report.html` is written.

Individual stages: `libspeech_bench_corpus`, `_throughput`, `_memory`, `_sizes`, `_system_info`, `_report`.

## Methodology

- **Corpus** (`python/corpus.py`): deterministic synthetic speech-like signal, 16-bit mono WAV, so every library reads identical bytes. STFT/MFCC run on the 16 kHz files, resample on the 44.1 kHz files, load on all.
- **What each operation means** is defined once in `python/ops.py` (n_fft 512, hop 128, 13 MFCCs from 26 mels, resample to 16 kHz). Speed and memory both import it.
- **Timing**: one untimed warm-up, then N timed calls with GC paused; min is the headline, median is stored. Transform objects (STFT, MFCC, Resample) are constructed inside the timed call for every library.
- **Memory**: one fresh process per data point (`bench_op_memory_one.py`), peak RSS reset via `/proc/self/clear_refs`, reported as the extra memory the call needed on top of the loaded interpreter, library and input. Linux only; elsewhere it falls back to `ru_maxrss`, which can read 0.
- **Install size**: `pip-size` against PyPI, so it reflects the latest release, not your checkout.

## What is compared (read this before quoting a number)

Every row is a **Python-level call**: libspeech through its nanobind bindings, `audioflux` (PyPI package, not the copy vendored inside libspeech), `librosa`, `scipy`, `soundfile`. No C++ timing is ever put next to a Python one in these tables.

- Same inputs and parameters for everyone (see `ops.py`): n_fft 512, hop 128, Hann, no centering; 13 MFCCs from 26 mels; 44.1 kHz -> 16 kHz.
- The STFT outputs have the same shape for libspeech (NumPy, `onesided=True`), librosa and audioflux (257 bins); only `libspeech_list` returns 512.
- **MFCC is each library's own pipeline.** Mel filterbank edges, log and DCT conventions differ (libspeech vs librosa agree in trajectory but not numerically, see `tests/python/test_librosa_compat.py`), so that row measures "the MFCC call with these parameters", not an identical computation.
- **Resample quality tiers differ**: libspeech Kaiser (zeroNum 64), librosa `soxr_hq`, audioflux "Best" tables, scipy `resample_poly`. libspeech and `soxr_hq` agree to ~1e-6 on in-band tones.
- **Warm-up primes libspeech's filter-bank cache.** The timed call constructs `Resample` every time like the others, but after the warm-up call that construction is a cache hit (~0.07 ms). A cold first call for a new rate pair also designs the filter (~0.5 ms more). Over many files at the same rates the cached number is the representative one; for a single one-off resample use the cold one.
- `load`: audioflux has no file loader, so that row is libspeech vs librosa vs soundfile.
- Single-core sandbox numbers; run-to-run noise is 5-15%. OpenMP paths (STFT/MFCC/resample) are untested for speed here.

## Reading the results

- **Two libspeech rows.** `libspeech` is the NumPy path: float32 arrays are read in place and results come back as NumPy arrays that own the C++ buffer (no list conversion, no copies). `libspeech_list` is the original `list[float]` in / nested-list out API, kept so its cost stays visible. A caller holding lists or float64/strided arrays pays one conversion (nanobind copies those once).
- **Outputs are not bit-identical** across libraries (window and filterbank conventions differ), and libspeech's STFT returns all `fftLength` bins per frame by default while librosa and the audioflux package return `fftLength/2 + 1`; the benchmark therefore asks libspeech for `onesided=True` (257 bins, same shape as the other two). Compare times as "the same call a user would make", not as identical arithmetic.
- **Ratios** in the report are relative to the fastest library in that row, at that input size only.
- **Single-core hosts** hide OpenMP gains in libspeech's STFT. Include the machine info footer when sharing numbers.

## Where time went, and what changed (single-core sandbox, treat as indicative)

Before optimization the Python API was dominated by building Python objects and by redundant passes, not by DSP: STFT on 10 s of 16 kHz audio took ~49 ms through lists while the C++ core took ~6 ms. What changed, all covered by `tests/dsp/test_fast_paths.cpp` and `tests/python/test_numpy_api.py`:

- **Real-input FFT** (`src/dsp/real_fft.{h,cpp}`): frames are windowed and bit-reverse-loaded as N/2 complex samples in one pass, one N/2-point FFT runs on split re/im arrays (portable code that the compiler vectorizes for the baseline ISA; an AVX2 clone measured slower and was removed), then a split pass yields the N/2+1 unique bins. Checked against a double-precision naive DFT for every size from 16 to 8192. The previous implementation (AudioFlux's C STFT kernel) ran a full complex FFT and wrote all N bins per frame.
- **STFT**: flat frame-major `Spectrogram`; `onesided=True` returns 257 instead of 512 columns (half the work and memory, what librosa returns); the full spectrum is still available and is the conjugate-mirrored result of the same engine. Frame loops carry `#pragma omp` (threshold 128 frames) -- unmeasured here, one core.
- **MFCC**: block-streamed (64 frames), power spectra straight from the real FFT, sparse mel filters, and a DCT whose cosine basis is built once (the old one called `std::cos` ~340 times per frame).
- **Resample**: the polyphase bank is now designed directly in double precision from the closed-form Kaiser-windowed sinc (AudioFlux tabulated the same function and linearly interpolated it), cached process-wide (LRU, 32 MiB) so repeated `Resample(src, dst)` costs ~0.07 ms, and run by an AVX2+FMA / NEON kernel blocked for cache, split across cores for long inputs (OpenMP, when available). Same band edges as the old "Best" preset; in-band accuracy went from ~1e-3 (44.1 -> 16 kHz) to ~4e-8, i.e. at or below soxr_hq's own error (1e-7), and the output now agrees with librosa's `soxr_hq` to 1e-6 for every ratio tested. AudioFlux's resampler is created lazily and only backs continuous mode, `isScale`, arbitrary float ratios and non-Kaiser windows.
- **Load / Audio**: mono decodes straight into the channel buffer, `Audio.size()` no longer copies the channel, `Audio.to_numpy()` avoids Python floats.

Measured in one process, 30 repeats, min ms (libspeech NumPy path, STFT one-sided like librosa; resample against librosa's default `soxr_hq`):

| input | libspeech | librosa | audioflux | scipy |
|---|---|---|---|---|
| stft 10 s / 60 s | 2.0 / 12.7 | 4.0 / 21.3 | 8.2 / 47.6 | |
| mfcc 10 s / 60 s | 2.1 / 11.9 | 6.0 / 28.7 | 17.2 / 113 | |
| resample 44.1 -> 16 kHz, 10 s / 60 s | 2.8 / 15.9 | 3.0 / 16.4 | 73 / 424 | 8.8 / 45.9 |
| load 10 s / 44.1 kHz 60 s | 0.45 / 6.2 | 1.4 / 18.5 | | |

STFT, MFCC and load are 2x-3x ahead of librosa. Resample is level with soxr_hq (2.8 vs 3.0 ms and 15.9 vs 16.4 ms; run-to-run noise on this machine is 5-15%, so read it as parity, not a win) but with equal or better accuracy and a ~3x faster cached construction path; soxr is a multi-stage design and is hard to beat per core. Multi-core speed-ups (OpenMP in STFT/MFCC/resample) are untested here -- this sandbox has one core.

Also worth knowing: every call logs `Debug`/`Trace` lines to stderr through AixLog, and the Python API has no way to change the level.

## Not covered yet

- No scaling sweep (time vs input length) or CLI benchmarks like pygixml has.
- No comparison against `sherpa-onnx` for the ONNX models (still open in `checklist.md`).
