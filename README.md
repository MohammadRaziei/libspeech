<p align="center">
  <a href="https://github.com/MohammadRaziei/libspeech">
    <img src="https://raw.githubusercontent.com/MohammadRaziei/libspeech/master/docs/logo/libspeech-logo-pods.svg" alt="libspeech logo" width="280">
  </a>
</p>

<h1 align="center">libspeech</h1>

<p align="center">
  Audio I/O, signal processing and ONNX speech models for Python and C++.<br>
  <code>pip install libspeech</code>, no system dependencies.
</p>

<p align="center">
  <a href="https://pypi.org/project/libspeech/"><img alt="PyPI" src="https://img.shields.io/pypi/v/libspeech"></a>
  <a href="https://pypi.org/project/libspeech/"><img alt="Python versions" src="https://img.shields.io/pypi/pyversions/libspeech"></a>
  <a href="https://github.com/MohammadRaziei/libspeech/actions"><img alt="CI" src="https://github.com/MohammadRaziei/libspeech/actions/workflows/cmake.yml/badge.svg"></a>
  <a href="https://github.com/MohammadRaziei/libspeech/blob/master/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache%202.0-blue"></a>
  <a href="https://en.cppreference.com/w/cpp/17"><img alt="C++17" src="https://img.shields.io/badge/C%2B%2B-17-blue"></a>
</p>

libspeech is a C++17 library with Python bindings. It reads and writes audio, computes the usual
spectral features, and runs speech-enhancement and voice-activity models on ONNX Runtime, without
PyTorch. The Python API works on NumPy arrays and follows librosa where the two overlap.

- **Audio I/O**: load WAV, MP3 and FLAC; save WAV; play, resample, mix down to mono. `Audio` converts to and from NumPy and DLPack.
- **DSP**: FFT/IFFT, DCT/IDCT, STFT/ISTFT, MFCC, window functions and a polyphase resampler, with SIMD kernels chosen at run time from the CPU (AVX2/FMA, NEON, or portable C++).
- **Models**: speech denoising (Facebook Denoiser, SpeechBrain SepFormer, GTCRN, DPDFNet) and Silero voice-activity detection. Weights are downloaded on first use and verified with SHA-256.
- **Platforms**: Linux, macOS and Windows, on x86-64 and ARM64.

## Installation

```bash
pip install libspeech
```

Wheels cover CPython 3.9 and newer for Linux (x86-64), macOS (universal2) and Windows (x64),
including free-threaded builds (3.13t, 3.14t, 3.15t). Any other platform builds from the source
distribution (see [Building from source](#building-from-source)).

## Quick start

### Audio and NumPy

```python
import numpy as np
import libspeech

audio = libspeech.Audio()
audio.load("sample.wav")                     # WAV, MP3 or FLAC
mono = audio.to_mono().resample(16000)       # every transform returns a new Audio
mono.save("sample_16k_mono.wav")             # saves WAV
x = mono.to_numpy()                          # float32, shape (samples,): a copy of one channel

view = np.asarray(audio)                     # float32 (channels, samples), shares memory: no copy
same = libspeech.Audio(view, audio.sample_rate)   # array -> Audio (copied once); 1-D means mono
```

`np.from_dlpack(audio)` and `torch.from_dlpack(audio)` work too. Samples are stored as one planar
block, and every `load()` replaces it as a whole, so an array you took earlier stays valid.

### Spectral features

```python
z = libspeech.stft(x, n_fft=512, hop_length=128)       # complex64, (257, frames), like librosa.stft
y = libspeech.istft(z, hop_length=128, length=len(x))  # back to samples

mfcc = libspeech.MFCC(libspeech.MFCCParams())
coefficients = mfcc.compute(mono.data(0))              # [frames][coefficients]
```

`STFT`, `FFT`, `Resample` and the other classes keep their plans between calls and expose the
layout (split or interleaved complex output, one-sided or full).

### Speech models

```python
# Downloaded to ~/.libspeech on first use, checked against its SHA-256, then cached.
with libspeech.Denoiser("gtcrn") as denoiser:
    clean = denoiser(x)                      # a 1-D NumPy array (or a list) in, the same kind out
    print(denoiser.readme())                 # what the model is, who made it, its license

vad = libspeech.SileroVad()
vad.process(mono.data(0))
for segment in vad.get_speech_timestamps():
    print(f"speech from {segment.start_s:.2f}s to {segment.end_s:.2f}s")
```

`Denoiser` takes a backend name (`"gtcrn"`, `"dpdfnet"`, `"facebook"`, `"speechbrain"`) and uses
that backend's standard model; `model="path-or-url.onnx"` picks another.
`libspeech.fetch_model("dpdfnet2.onnx")` downloads and verifies a model without loading it.

### C++

```cpp
#include "libspeech/audio.h"
#include "libspeech/dsp/mfcc.h"
#include "libspeech/models/denoiser.h"

speech::io::audio audio;
audio.load("sample.wav");
auto mono = audio.to_mono().resample(16000);

speech::dsp::mfcc::params params;
params.sample_rate = 16000;
speech::dsp::mfcc mfcc(params);
auto coefficients = mfcc.compute(mono.data(0));

auto denoiser = speech::models::denoiser::create("gtcrn", "gtcrn_simple.onnx");
auto clean = denoiser->process(mono.data(0));
```

Complete programs are in [`examples/`](https://github.com/MohammadRaziei/libspeech/tree/master/examples).

## Models

| Backend | Model | Sample rate | License | Commercial use |
|---|---|---|---|---|
| `gtcrn` | `gtcrn_simple` | 16 kHz | MIT | yes |
| `dpdfnet` | `dpdfnet_baseline`, `dpdfnet2`, `dpdfnet4`, `dpdfnet8` | 16 kHz | Apache-2.0 | yes |
| `dpdfnet` | `dpdfnet2_48khz_hr` | 48 kHz | Apache-2.0 | yes |
| `speechbrain` | `speechbrain-sepformer-wham-enhancement`, `...-wham16k-enhancement` | 8 / 16 kHz | Apache-2.0 | yes |
| `facebook` | `facebook-denoiser-dns48`, `-dns64`, `-master64` | 16 kHz | CC BY-NC 4.0 | no |
| `silero_vad` | `silero-vad`, `silero-vad-half`, `silero-vad-16k-op15` | 8 / 16 kHz | MIT | yes |

Every model is published in the repository's [`models` release](https://github.com/MohammadRaziei/libspeech/releases/tag/models)
together with a `.sha256` and a `.readme` file (origin, training data, license). A cached model is
verified again on every load. [`data/models.csv`](https://github.com/MohammadRaziei/libspeech/blob/master/data/models.csv)
lists all of them with sizes and sources.

## Building from source

```bash
git clone https://github.com/MohammadRaziei/libspeech.git
cd libspeech
git submodule update --init src/third_party/miniaudio
pip install httpp                    # model downloads (BUILD_MODELS=ON, the default)

cmake -S . -B build                  # fetches ONNX Runtime on the first configure
cmake --build build -j
ctest --test-dir build
```

Only need the DSP and audio layers? `-DBUILD_MODELS=OFF` needs neither ONNX Runtime nor network access.
The Python extension is built with `-DBUILD_PYTHON=ON`. Options, test targets, coverage and the
SIMD layout are described in [`CONTRIBUTING.md`](https://github.com/MohammadRaziei/libspeech/blob/master/CONTRIBUTING.md).

## Project layout

| Namespace | CMake target | Python module | Depends on |
|---|---|---|---|
| `speech::dsp` | `speech_dsp` | `speech_dsp_py` | a vendored subset of [AudioFlux](https://github.com/libAudioFlux/audioflux) |
| `speech::io` | `speech_io` | `speech_io_py` | `speech::dsp`, miniaudio |
| `speech::models` | `speech_models` | `speech_models_py` | ONNX Runtime, [httpp](https://github.com/MohammadRaziei/httpp) |

Python users import everything from `libspeech`; the `speech_*_py` modules are an implementation detail.
Logging is quiet by default (warnings and errors); `libspeech.set_log_level("debug")` or the
`LIBSPEECH_LOG` environment variable turns it up.

## Contributing

Bug reports and pull requests are welcome. [`CONTRIBUTING.md`](https://github.com/MohammadRaziei/libspeech/blob/master/CONTRIBUTING.md)
covers the build, tests, style and the release process; changes are listed in the
[`CHANGELOG`](https://github.com/MohammadRaziei/libspeech/blob/master/CHANGELOG.md).

## License

Apache License 2.0, see [`LICENSE`](https://github.com/MohammadRaziei/libspeech/blob/master/LICENSE)
and [`NOTICE`](https://github.com/MohammadRaziei/libspeech/blob/master/NOTICE). Downloaded models
carry their own licenses (table above).
