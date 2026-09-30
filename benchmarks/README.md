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

- **Two libspeech rows.** libspeech's Python API takes `list[float]` and returns nested lists. `libspeech` times exactly that. `libspeech_np` also counts `ndarray.tolist()` on input and `np.asarray()` on output, which is what a numpy user pays.
- **Outputs are not bit-identical** across libraries (window and filterbank conventions differ), and libspeech's STFT returns all `fftLength` bins per frame while librosa and audioflux return `fftLength/2 + 1`. Compare times as "the same call a user would make", not as identical arithmetic.
- **Ratios** in the report are relative to the fastest library in that row, at that input size only.
- **Single-core hosts** hide OpenMP gains in libspeech's STFT. Include the machine info footer when sharing numbers.

## Known finding (2026-09-30, single-core sandbox, 3 repeats, treat as indicative)

For STFT on 10 s of 16 kHz audio (n_fft 512, hop 128), native C++ takes about 6 ms, in line with librosa and audioflux, but the Python call takes about 49 ms. The gap is building `1247 x 512` nested Python lists twice (real and imaginary), not DSP. Returning numpy arrays from the bindings would remove most of it. Native C++ timing at the same parameters can be reproduced by editing `cpp/bench_stft.cpp` to use `STFT(9, Window_Hann, 128)` on 160000 samples.

Also worth knowing: every call currently logs `Debug`/`Trace` lines to stderr through AixLog, and the Python API has no way to change the level.

## Not covered yet

- No scaling sweep (time vs input length) or CLI benchmarks like pygixml has.
- No comparison against `sherpa-onnx` for the ONNX models (still open in `checklist.md`).
