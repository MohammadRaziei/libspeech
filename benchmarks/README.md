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

## Reading the results

- **Two libspeech rows.** `libspeech` is the NumPy path: float32 arrays are read in place and results come back as NumPy arrays that own the C++ buffer (no list conversion, no copies). `libspeech_list` is the original `list[float]` in / nested-list out API, kept so its cost stays visible. A caller holding lists or float64/strided arrays pays one conversion (nanobind copies those once).
- **Outputs are not bit-identical** across libraries (window and filterbank conventions differ), and libspeech's STFT returns all `fftLength` bins per frame while librosa and audioflux return `fftLength/2 + 1`. Compare times as "the same call a user would make", not as identical arithmetic.
- **Ratios** in the report are relative to the fastest library in that row, at that input size only.
- **Single-core hosts** hide OpenMP gains in libspeech's STFT. Include the machine info footer when sharing numbers.

## Where time went, and what changed (single-core sandbox, treat as indicative)

Before optimization, the Python API was dominated by building Python objects, not DSP: STFT on 10 s of 16 kHz audio took ~49 ms through lists while the C++ core took ~6 ms. The fixes, all covered by tests in `tests/dsp/test_fast_paths.cpp` and `tests/python/test_numpy_api.py`:

- **STFT**: flat, frame-major `Spectrogram` / `stftInto()` (no per-frame vectors, no zero-fill); NumPy bindings hand the buffer over without copying.
- **MFCC**: one pass in 64-frame blocks (no full spectrogram), sparse mel filters, power only over the bins the filters touch, and a DCT whose cosine basis is built once. The old DCT called `std::cos` ~340 times per frame. Native C++ time on 10 s: 16.7 ms -> 5.2 ms, values equal to 1e-5.
- **Resample**: exact rational polyphase filter bank built from the same tables, contiguous weights, AVX2+FMA kernel with runtime dispatch (`LIBSPEECH_DISABLE_AVX2=1` forces the portable one). Native 10 s at 44.1->16 kHz: 56 ms -> ~4 ms. Output differs from the old kernel by interpolation rounding only (<= 2.2e-3 on a full-scale test tone) and is closer to the analytic signal.
- **Load**: mono decodes straight into the channel buffer; `Audio.size()` no longer copies the channel; `Audio.to_numpy()` replaces `data(0)` for numpy users.

Measured in one process, 40 repeats, 10 s inputs (min ms): load 0.32 (librosa 1.20), mfcc 6.0 (librosa 5.1, audioflux 15.7), stft 7.1 (librosa 3.8, audioflux 8.0), resample 6.1 (librosa 2.6, audioflux 61). So libspeech is clearly ahead on load and well ahead of audioflux on MFCC/resample, but still behind librosa on STFT and resample.

Why: STFT still computes a full-length complex FFT per frame (all 512 bins, real and imaginary planes) where librosa returns 257 bins. And libspeech's default resample quality is "Best" (64 zero crossings, a resampy-class filter) while librosa's default is soxr_hq; compare against `res_type="kaiser_best"` for like-for-like quality.

Also worth knowing: every call logs `Debug`/`Trace` lines to stderr through AixLog, and the Python API has no way to change the level.

## Not covered yet

- A real-input FFT (half the work for STFT/MFCC) is the next speed item.
- No scaling sweep (time vs input length) or CLI benchmarks like pygixml has.
- No comparison against `sherpa-onnx` for the ONNX models (still open in `checklist.md`).
