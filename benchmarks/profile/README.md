# Profiling the parts outside the main benchmark suite

The main suite (`../python`) covers load, resample, STFT and MFCC. This directory profiles what it
does not: `FFT` (forward / inverse / dct / idct), the free `dctII` / `libspeech.dct`, `STFT::istft`, and the
ONNX model pipelines (Silero VAD, the two denoisers). Findings as of the first profile (single-core
sandbox, treat as indicative; min of several runs):

## Findings

| Area | Finding | Action |
|---|---|---|
| `dctII` / `libspeech.dct` | Direct O(N*K) sum that called `std::cos` per term: n=512 took 4.8 ms, n=4096 took **310 ms** (scipy: 0.1 / 0.15 ms). | **Fixed**: power-of-two sizes 16..8192 go through one real FFT (Makhoul), other sizes use a cos table built once. n=512: 0.029 ms, n=4096: 0.30 ms (Python, incl. list conversion); native 0.007 / 0.055 ms. Checked against the naive definition for every size/path and against scipy. |
| `SileroVadModel` default model | Default `model_path="silero_vad.onnx"` is a **404** on the `models` release (the file is `silero-vad.onnx`), so `SileroVad()` could never download its model. | **Fixed** (header and Python binding). |
| Silero VAD wrapper | 114 us per 512-sample chunk (60 s of audio: 213 ms, ~280x real time). A bare ONNX Runtime loop doing the same inferences takes the same time, and both make ~327 heap allocations per chunk -- the wrapper adds ~1. | None needed: the per-chunk vector copies in `processOnVector` are noise next to ONNX Runtime. |
| Denoisers | Facebook DNS64: 2.3x real time, SpeechBrain SepFormer: 3.1x real time on this 1-core machine; wrapper overhead is within run-to-run noise (<2%). Both force `SetIntraOpNumThreads(1)`, so more cores will not help. | **Not changed, unmeasurable here** (1 core): exposing the thread count (default 1 = today's behaviour) is the lever for multi-core machines. |
| `FFT` class (lists) | At N=512 it beats numpy (0.033 vs 0.048 ms). At N=65536 it is 4-13x slower (7.7 ms vs 1.95 ms; dct 5.3 vs 0.43): list conversion is ~2.3 ms of that and AudioFlux's complex FFT at large N is ~4.5 ms. Below N=4096 the Python time is mostly list conversion. | Open: NumPy in/out overloads (removes conversion), and the real-FFT engine for real input up to 8192. |
| `STFT::istft` | Python (nested lists) 28.6 ms vs librosa 8.3 ms, but converting the 1247x512 nested lists alone costs 34 ms: conversion dominates. Native: 14 ms = AudioFlux kernel 9.2 ms + 4.1 ms flattening `vector<vector>` + 5.6 MB of allocations. Round trip error 1.5e-7. | Open: flat/NumPy overload (drops the flatten and the conversion), and a real inverse FFT in place of AudioFlux's complex one (est. 2-3x on the kernel). |

## Running it

```bash
# Python-level (needs libspeech, numpy, scipy, librosa installed)
python benchmarks/profile/prof_dsp.py

# Native C++ (allocation counts and compute-vs-wrapper split); BUILD is a build dir with libspeech_dsp.a
# and libaudioflux.a (cmake -S . -B build -DBUILD_MODELS=ON -DBUILD_TESTS=ON && cmake --build build)
AF=src/third_party/audioflux
g++ -O3 -std=c++17 -fopenmp -Iinclude -I$AF/src -I$AF/include benchmarks/profile/prof_dsp.cpp \
    build/libspeech_dsp.a build/libaudioflux.a -o prof_dsp && ./prof_dsp
g++ -O3 -std=c++17 -fopenmp -Iinclude -I$AF/src -I$AF/include benchmarks/profile/prof_dct.cpp \
    build/libspeech_dsp.a build/libaudioflux.a -o prof_dct && ./prof_dct

# Models: put the three files from the `models` release into ~/.libspeech first
#   silero-vad.onnx  facebook-denoiser-dns64.onnx  speechbrain-sepformer-wham16k-enhancement.onnx
ORT=src/third_party/onnxruntime
g++ -O3 -std=c++17 -Iinclude -I$ORT/include -Isrc/third_party/aixlog/include benchmarks/profile/prof_models.cpp \
    build/libspeech_models.a -L$ORT/lib -lonnxruntime -Wl,-rpath,$ORT/lib \
    "$(python -c 'import httpp,os;print(os.path.join(httpp.get_lib_dir(),"libhttpp_core.so"))')" -o prof_models
./prof_models   # the denoisers take a few minutes on one core
```
