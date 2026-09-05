from __future__ import annotations
from pathlib import Path

from ctypes import *

_here = Path(__file__).parent

# ONNXRuntime ships as a separate shared library we load explicitly before
# importing the compiled extensions (they dlopen symbols from it at import
# time). Its filename is version-suffixed (e.g. libonnxruntime.so.1.21.0);
# glob for whatever's actually here instead of hardcoding a version, so a
# future ONNXRuntime version bump (see cmake/ONNXRuntime.cmake) doesn't
# silently break this import.
for _onnxruntime_lib in _here.glob("libonnxruntime.so*"):
    cdll.LoadLibrary(_onnxruntime_lib.as_posix())
    break

# NOTE: AudioFlux and Mbed TLS are statically linked into libspeech.so
# itself (see CMakeLists.txt: speech_dsp/speech_models are STATIC
# libraries), so there is no separate libaudioflux.so/libmbedtls.so to
# preload here -- only libspeech.so itself needs an explicit load.
cdll.LoadLibrary(_here.joinpath("libspeech.so").as_posix())


from ._about import __version__

# The compiled extension modules (_about, speech_io_py, ...) are
# intentionally underscore/private-prefixed and not meant to be imported
# directly by users -- `import libspeech; libspeech.Audio(...)`, not
# `import speech_io_py`. Re-export the public names here, matching ctoon's
# `from .ctoon_py import *` pattern. Module names mirror the CMake target
# names (speech_dsp/speech_io/speech_models -> speech_dsp_py/speech_io_py/
# speech_models_py) so the two naming schemes stay in sync.
from .speech_io_py import Audio
from .speech_dsp_py import Resample, WindowType, window, dct, FFT, STFT, MFCC, MFCCParams
from .speech_models_py import Denoiser, SileroVad, SpeechTimestamp

__all__ = [
    "__version__",
    "Audio",
    "Resample",
    "WindowType",
    "window",
    "dct",
    "FFT",
    "STFT",
    "MFCC",
    "MFCCParams",
    "Denoiser",
    "SileroVad",
    "SpeechTimestamp",
]
