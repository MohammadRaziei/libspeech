# libspeech C++ API {#mainpage}

**libspeech** is a C++17 library for audio I/O, DSP and ONNX-based speech models, with Python
bindings that mirror this API one to one. It installs with nothing but `pip` (Python) or CMake
(C++): no libcurl, OpenSSL or LibTorch.

| Namespace | Header | What it has |
|---|---|---|
| `speech::io` | `libspeech/audio.h` | load/save WAV, MP3, FLAC, playback, resample, mono mix |
| `speech::dsp` | `libspeech/dsp/*.h` | `resample`, `window`, `fft`, `dct_ii_plan`, `stft`, `mfcc`, SIMD switch |
| `speech::models` | `libspeech/models/*.h` | `denoiser` (Facebook, SpeechBrain), `silero_vad` |
| `speech::utils` | `libspeech/utils/*.h` | logging (`set_log_level`) |

## Installation

```cmake
include(FetchContent)
FetchContent_Declare(
  libspeech
  GIT_REPOSITORY https://github.com/MohammadRaziei/libspeech.git
  GIT_TAG        main
)
FetchContent_MakeAvailable(libspeech)

target_link_libraries(my_app PRIVATE speech::dsp speech::models speech::io)
```

Requires CMake 3.19+ and a C++17 compiler. `speech::models` needs ONNX Runtime (downloaded on
the first configure) and `pip install httpp` for model downloads; configure with
`-DBUILD_MODELS=OFF` for the DSP layer alone.

## Quick example

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
auto clean = denoiser->process(resampled.data(0));
denoiser->close();
```

## Naming

Everything is snake_case in the style of the STL; the old CamelCase names are listed in
`docs/api-renames.md`. Resampling quality is chosen with `speech::dsp::resample_quality`.
