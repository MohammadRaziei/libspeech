<p align="center">
  <a href="https://github.com/mohammadraziei/libspeech">
    <img src="https://raw.githubusercontent.com/MohammadRaziei/libspeech/master/docs/logo/libspeech-logo-pods.svg" alt="LibSpeech Logo" width="300">
  </a>
  <h2 align="center">LibSpeech</h2>
  <h3 align="center">Lightweight Speech Processing Library</h3>
</p>

<p align="center">
  <em>C++ library with Python bindings for audio I/O, DSP, and ONNX-based speech models -- built to install with nothing but <code>pip</code>, no system dependencies (libcurl, OpenSSL, ...) required.</em>
</p>

<div align="center">

[![GitHub release](https://img.shields.io/github/release/mohammadraziei/libspeech?color=blue)](https://github.com/mohammadraziei/libspeech/releases)
[![License](https://img.shields.io/badge/license-MIT-purple)](LICENSE)
[![Python Versions](https://img.shields.io/pypi/pyversions/libspeech)](https://pypi.org/project/libspeech/)
[![C++ Standard](https://img.shields.io/badge/C++-17-blue)](https://en.cppreference.com/w/cpp/17)

[![Build Status](https://github.com/mohammadraziei/libspeech/actions/workflows/cmake.yml/badge.svg)](https://github.com/mohammadraziei/libspeech/actions)
[![Code Quality](https://sonarcloud.io/api/project_badges/measure?project=MohammadRaziei_libspeech&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=MohammadRaziei_libspeech)
[![CodeFactor](https://www.codefactor.io/repository/github/mohammadraziei/libspeech/badge/master)](https://www.codefactor.io/repository/github/mohammadraziei/libspeech/overview/master)
[![snyk.io](https://snyk.io/advisor/python/libspeech/badge.svg)](https://snyk.io/advisor/python/libspeech)
[![PyPI Downloads](https://img.shields.io/pypi/dm/libspeech)](https://pypi.org/project/libspeech/)

</div>

## Contents

<ul>
<li><a href="#-key-features">Key Features</a></li>
<li><a href="#-installation">Installation</a></li>
<li><a href="#-quick-start">Quick Start</a></li>
<li><a href="#-testing">Testing</a></li>
<li><a href="#-architecture">Architecture</a></li>
<li><a href="#-contributing">Contributing</a></li>
</ul>

## ✨ Key Features

- 🚀 **ONNX Runtime powered** models, without a heavy LibTorch dependency
- 🎙️ **Audio I/O** (`speech::io`): load/save WAV/MP3/FLAC, playback, resample, mono-mixing
- 🧮 **DSP** (`speech::dsp`): Resample, window functions, FFT/IFFT, DCT/IDCT, STFT/ISTFT, MFCC -- all with their own C++ unit tests and Python bindings
- 🔊 **Speech models** (`speech::models`): a `Denoiser` interface (Facebook, SpeechBrain, GTCRN and DPDFNet backends) and Silero VAD
- 🖥️ **Cross-platform**: Windows, Linux, macOS
- 🐍 **Python bindings** (nanobind) mirroring the C++ API 1:1
- 📦 **Zero system dependencies**: no `apt install libcurl-dev` / OpenSSL needed -- HTTPS model downloads go through [`httpp`](https://github.com/mohammadraziei/httpp) (`pip install httpp`), a separately-installed package with its own prebuilt HTTP(S) client and progress bar

## 📦 Installation

### Python package
```bash
pip install libspeech
```

### From source (C++/CMake)
```bash
git clone https://github.com/MohammadRaziei/libspeech.git
cd libspeech
git submodule update --init src/third_party/miniaudio

# speech::models (BUILD_MODELS=ON, the default) needs httpp on
# CMAKE_PREFIX_PATH for model downloads (HTTPS client + progress bar) --
# no more Mbed TLS/libcurl to build, just:
pip install httpp

mkdir build && cd build
cmake ..                      # downloads ONNX Runtime automatically on first configure
cmake --build . -j$(nproc)
ctest                         # runs the full test suite (see "Testing" below)
```

Only want the DSP layer (no ONNX Runtime/network access needed at all)?
```bash
cmake .. -DBUILD_MODELS=OFF
cmake --build . -j$(nproc)
```

## 🚀 Quick Start

### Python

```python
import libspeech

# --- Audio I/O ---
audio = libspeech.Audio()
audio.load("sample.wav")
mono = audio.to_mono()
resampled = mono.resample(16000)
resampled.save("sample_16k_mono.wav")

# --- DSP ---
mfcc = libspeech.MFCC(libspeech.MFCCParams())
coefficients = mfcc.compute(resampled.data(0))   # [num_frames][num_coefficients]

# --- NumPy: float32 in, NumPy out, no list conversion ---
x = resampled.to_numpy()                          # float32 1-D array (any float array works)

# The everyday functions follow librosa: keyword-only n_fft / hop_length / window / center,
# a one-sided complex64 result of shape (bins, frames), center=True padding.
z = libspeech.stft(x, n_fft=512, hop_length=128)  # (257, frames) -- same as librosa.stft(x, ...)
y = libspeech.istft(z, hop_length=128, length=len(x))

# The classes keep a transform around and give you control of the layout:
stft = libspeech.STFT(n_fft=512, hop_length=128)  # or STFT(9, ...): the exponent form
z = stft.stft_complex(x, onesided=True)           # (frames, 257) complex64, written in place
y = stft.istft(z, "weighted")                     # "weighted" (default) or "plain"
re, im = stft.stft(x)                             # the older (re, im) pair; onesided=False = all 512 bins

fft = libspeech.FFT(12)
spectrum = fft.fft(x[:4096])                      # complex64, like np.fft.fft (fft.ifft inverts it)
re, im = fft.forward(x[:4096])                    # the same as a pair; also inverse/dct/idct

# --- Speech models ---
# Denoiser("facebook") uses the backend's standard model (downloaded to ~/.libspeech on first use);
# model="path-or-url.onnx" picks another. `with` frees the model's memory on exit (or call close()).
# Calling it takes a list or a 1-D NumPy array (and returns the same kind) and lets other Python
# threads run meanwhile. num_threads: 1 (default), or 0 = one ONNX Runtime thread per hardware thread.
with libspeech.Denoiser("facebook") as denoiser:
    clean = denoiser(resampled.to_numpy())
    print(denoiser.readme())        # what the model is, who made it, its license (also: .model_path, .sha256)

# Every model is downloaded together with <model>.onnx.sha256 (fetched first, checked, then the model is
# hashed and compared) and <model>.onnx.readme. A cached model is verified on every load.
# ignore_cache=True downloads everything again. data/models.csv lists all models and their licenses.
denoiser = libspeech.Denoiser("gtcrn", ignore_cache=True)
path = libspeech.fetch_model("dpdfnet2.onnx")       # just download + verify, returns the file path
text = libspeech.model_readme("silero-vad.onnx")
digest = libspeech.sha256_file(path)                # also libspeech.sha256(data)

vad = libspeech.SileroVad()
vad.process(resampled.data(0))
for segment in vad.get_speech_timestamps():
    print(f"speech from {segment.start_s:.2f}s to {segment.end_s:.2f}s")
```

### C++

```cpp
#include "libspeech/audio.h"
#include "libspeech/dsp/mfcc.h"
#include "libspeech/models/denoiser.h"

speech::io::audio audio;
audio.load("sample.wav");
auto resampled = audio.to_mono().resample(16000);

speech::dsp::mfcc::params params;
params.sample_rate = 16000;
speech::dsp::mfcc mfcc(params);
auto coefficients = mfcc.compute(resampled.data(0));

auto denoiser = speech::models::denoiser::create("facebook", "facebook_denoiser.onnx");
auto clean = denoiser->process(resampled.data(0));   // also: process(const float*, size_t)
denoiser->close();                                    // optional: frees the model now, not at destruction
```

See [`examples/`](examples/) for complete, runnable programs.

### Views for C++ (no copies, no allocations)

`speech::span<T>` (a pointer and a length; a C++17 stand-in for `std::span`) and `speech::frame_view<T>`
(a buffer cut into frames) bundle what the raw-buffer calls otherwise take as loose arguments. A
`std::vector` converts to a `span` automatically; the pointer forms of every call stay available.

```cpp
#include "libspeech/dsp/stft.h"
speech::dsp::stft analysis(9, speech::dsp::window_type::hann, 128);

// Frames of a signal, as the stft cuts them (no copy), and a spectrogram frame by frame:
for (speech::span<const float> frame : analysis.frames(signal)) { /* N samples */ }
auto spec = analysis.spectrogram(signal.data(), signal.size(), /*onesided=*/true);
for (auto frame : spec.frames())              // one spectrum
    for (std::complex<float> bin : frame)     // one bin: two references, nothing allocated
        energy += std::norm(bin);

// Straight into interleaved std::complex<float> (= NumPy complex64), then back:
const int frames = analysis.cal_time_length(signal.size()), bins = analysis.num_bins(true);
std::vector<std::complex<float>> out(frames * bins);
analysis.stft_onesided_into(signal, speech::frame_view<std::complex<float>>(out.data(), frames, bins));
auto back = analysis.istft(speech::frame_view<const std::complex<float>>(out.data(), frames, bins));
```

Complex data comes in two layouts, and both iterate without allocating. Interleaved storage
(`complex64`, `std::complex<float>[]`) is a plain `reinterpret_cast` (`speech::as_complex()`), so its
iterator yields real `std::complex<float>&`. Split planes (all real parts, then all imaginary parts,
as in `complex_spectrogram`) cannot be reinterpreted, so `split_complex_span` hands out a
two-reference proxy that converts to and from `std::complex<float>`.

### Logging

libspeech is quiet by default: it only writes warnings and errors to stderr. Turn the detail up (or
everything off) from Python, C++, or the environment:

```python
import libspeech
libspeech.set_log_level("debug")   # trace | debug | info | warning (default) | error | off
libspeech.get_log_level()          # "debug"
```

```cpp
#include "libspeech/utils/log.h"
speech::utils::set_log_level(speech::utils::log_level::off);
```

```bash
LIBSPEECH_LOG=debug python my_script.py   # the starting level; set_log_level() overrides it
```

Lines look like `2026-10-09 12:00:00.123 [debug] (speech::dsp::FFT) Created FFT: ...`.

## 🧪 Testing

Tests are organized by language and module (`speech_test_<language>_<module>`),
all discoverable via `cmake --build build --target help`:

```
speech_test                       # everything
├── speech_test_cpp
│   ├── speech_test_cpp_dsp       # speech::dsp (C++), no network/ONNX needed
│   └── speech_test_cpp_models    # speech::models (C++)
└── speech_test_python
    ├── speech_test_python_audio  # libspeech.Audio
    ├── speech_test_python_dsp    # libspeech.Resample/FFT/STFT/MFCC/...
    ├── speech_test_python_numpy_api       # NumPy in/out fast paths
    ├── speech_test_python_simd            # SIMD backend selection and agreement
    ├── speech_test_python_librosa_compat  # load/save/resample/STFT/MFCC vs librosa
    └── speech_test_python_models # libspeech.Denoiser/SileroVad
```

The librosa cross-checks need the test extra (`pip install "libspeech[test]"` installs
librosa, soundfile, scipy and numpy); without them that module skips itself.

Run everything with `cmake --build build --target speech_test`, or just
`ctest` for the same suites without the extra build-tool chatter.

### Docs & coverage

```bash
cmake .. -DBUILD_DOCS=ON && cmake --build . --target speech_docs   # -> build/docs/out/index.html (C++ via Doxygen, Python via Sphinx)
```

Coverage follows the same layout as [ctoon](https://github.com/mohammadraziei/ctoon):
one target per component, a merged total, and a dashboard tying the reports
together.

```bash
cmake .. -DBUILD_PYTHON=ON -DBUILD_COVERAGE=ON   # + `pip install coverage`, lcov/genhtml on PATH
cmake --build . --target speech_coverage         # everything below, then the merge + dashboard
```

| Target | Covers | Output (under `build/coverage/`) |
|---|---|---|
| `speech_coverage_cpp_dsp` | `speech::dsp` (utest.h suite) | `cpp_dsp/` |
| `speech_coverage_cpp_models` | `speech::models` + download utils | `cpp_models/` |
| `speech_coverage_python` | the Python package's `.py` files (coverage.py) | `python/` |
| `speech_coverage_total` | lcov merge of all of the above | `total/` |

Open `build/coverage/index.html` for the dashboard. Each component (and the
total) also leaves a plain `coverage.lcov` next to its HTML, which is what CI
uploads to Codecov. `BUILD_COVERAGE` is off by default: it instruments
`speech_dsp`/`speech_models` with `--coverage`, which you don't want in a
normal build. Without `BUILD_PYTHON` you just get the C++ components.

## 🔧 Architecture

libspeech is split into three independent C++ libraries (each with its own
CMake target/namespace/Python module), aggregated by an umbrella `speech`
library:

| Namespace | CMake target | Python module | Depends on |
|---|---|---|---|
| `speech::dsp` | `speech_dsp` | `speech_dsp_py` | nothing but a vendored subset of [AudioFlux](https://github.com/libAudioFlux/audioflux)'s C sources |
| `speech::io` | `speech_io` | `speech_io_py` | `speech::dsp` (for resampling), miniaudio |
| `speech::models` | `speech_models` | `speech_models_py` | ONNX Runtime, [`httpp`](https://github.com/mohammadraziei/httpp) (for downloading model weights) |

See [`checklist.md`](checklist.md) for the detailed, up-to-date state of
the project (what's done, what's in progress, known issues), and
[`audioflux_issues.md`](audioflux_issues.md) for a couple of real bugs
found in AudioFlux while vendoring it (with repro steps and fixes, in case
they're useful upstream).

### SIMD and portability

Kernels that benefit from explicit SIMD (today the polyphase resampler, ~2.3x over its
portable twin on AVX2) have a portable C++ implementation plus vectorized twins that are
**selected at run time from the CPU**, never from build flags -- so one wheel or one default
CMake build is correct on every machine of its architecture and fast on the capable ones.
No `-march=native`, no per-ISA build variants. The real-input FFT behind STFT/MFCC is plain
portable C++ that the compiler vectorizes for the baseline ISA: a hand-targeted AVX2 clone
of it was measured (n_fft 64-8192) and was never faster, so it was dropped.

| Target | Fast path | Compilers | Notes |
|---|---|---|---|
| x86 / x86-64 | AVX2 + FMA (CPUID and OS YMM-state checked) | GCC, Clang, MSVC | MSVC path is written but not yet built in CI -- see below |
| ARM64 (Linux, macOS, Windows) | NEON | GCC, Clang, MSVC | |
| anything else (32-bit ARM, RISC-V, ...) | portable C++ | any C++17 compiler | the compiler may still auto-vectorize it |

```python
import libspeech
libspeech.simd_backend()        # 'avx2+fma' | 'neon' | 'generic'
libspeech.set_simd_enabled(False)   # run-time switch (objects/calls created afterwards)
```

- Environment: `LIBSPEECH_SIMD=off` forces the portable kernels (`LIBSPEECH_DISABLE_AVX2=1` still works).
- All backends are cross-checked against each other and against independent references in
  `tests/dsp/test_fast_paths.cpp`. The suite is run on x86-64 (GCC and Clang), 32-bit x86,
  AArch64 (under QEMU), and with SIMD disabled at run time.
- **MSVC**: the AVX2 resampler and the CPUID/XGETBV detection are written against MSVC's intrinsics but
  have not been compiled with `cl` yet (no Windows toolchain was available when this was written).
  If anything misbehaves there, `LIBSPEECH_SIMD=off` falls back to the portable code at run time.
- Layout for contributors: support is decided in the sources, not the build system.
  `src/dsp/simd_internal.h` holds every architecture/compiler macro (`LS_HAVE_X86_AVX2`,
  `LS_HAVE_ARM_NEON`, `LS_TARGET_AVX2`, ...): `#if LS_HAVE_X86_AVX2` the AVX2 kernel is compiled
  and used when the CPU has it, otherwise the portable one is. Kernels never test
  `__x86_64__`/`_MSC_VER` directly.

## 🤝 Contributing

We welcome contributions! Please see [`CONTRIBUTING.md`](CONTRIBUTING.md).

## 📄 License

This project is licensed under the Apache License 2.0 - see [LICENSE](LICENSE) and [NOTICE](NOTICE). The ONNX models it downloads have their own licenses (see `data/models.csv`; the Facebook denoiser models are non-commercial, CC BY-NC 4.0).

[![Stars](https://starchart.cc/mohammadraziei/libspeech.svg?variant=adaptive)](https://starchart.cc/mohammadraziei/libspeech)
