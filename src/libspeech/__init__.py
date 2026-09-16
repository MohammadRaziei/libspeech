from __future__ import annotations

from ctypes import *
from pathlib import Path

_here = Path(__file__).parent

# httpp and ONNXRuntime each ship as a separate shared library we load
# explicitly before importing the compiled extensions (they dlopen symbols
# from these at import time). Both get bundled right next to this
# __init__.py by CMakeLists.txt (see the SKBUILD install() rule / the
# POST_BUILD copy for a local dev build) rather than found via RPATH: an
# RPATH baked in at build time would only point back at wherever
# `pip install httpp` (or the ONNXRuntime download) happened to land on
# the *build* machine, which has no reason to exist -- or be the same
# path -- on whatever machine later installs this wheel. Loading a bundled
# copy by explicit path here works regardless of where/how this package
# itself ends up installed.
#
# httpp's filename isn't version-suffixed (unlike ONNXRuntime's, see
# below), but it does vary by platform (libhttpp_core.so / .dylib / .dll);
# glob for whichever one is actually here rather than hardcoding one.
for _httpp_lib in _here.glob("*httpp_core*"):
    cdll.LoadLibrary(_httpp_lib.as_posix())
    break

# ONNXRuntime ships as a separate shared library we load explicitly before
# importing the compiled extensions (they dlopen symbols from it at import
# time). Its filename is version-suffixed (e.g. libonnxruntime.so.1.21.0);
# glob for whatever's actually here instead of hardcoding a version, so a
# future ONNXRuntime version bump (see cmake/ONNXRuntime.cmake) doesn't
# silently break this import.
for _onnxruntime_lib in _here.glob("libonnxruntime.so*"):
    cdll.LoadLibrary(_onnxruntime_lib.as_posix())
    break

# NOTE: AudioFlux is statically linked into libspeech.so itself (see
# CMakeLists.txt: speech_dsp/speech_models are STATIC libraries), so there
# is no separate libaudioflux.so to preload here -- only libspeech.so
# itself needs an explicit load (httpp and ONNXRuntime, both dynamically
# linked, were already preloaded above).
cdll.LoadLibrary(_here.joinpath("libspeech.so").as_posix())


from ._about import __version__
from .speech_dsp_py import (
    FFT,
    MFCC,
    STFT,
    MFCCParams,
    Resample,
    WindowType,
    dct,
    window,
)

# The compiled extension modules (_about, speech_io_py, ...) are
# intentionally underscore/private-prefixed and not meant to be imported
# directly by users -- `import libspeech; libspeech.Audio(...)`, not
# `import speech_io_py`. Re-export the public names here, matching ctoon's
# `from .ctoon_py import *` pattern. Module names mirror the CMake target
# names (speech_dsp/speech_io/speech_models -> speech_dsp_py/speech_io_py/
# speech_models_py) so the two naming schemes stay in sync.
from .speech_io_py import Audio
from .speech_models_py import Denoiser, SileroVad, SpeechTimestamp

__all__ = [
    "FFT",
    "MFCC",
    "STFT",
    "Audio",
    "Denoiser",
    "MFCCParams",
    "Resample",
    "SileroVad",
    "SpeechTimestamp",
    "WindowType",
    "__version__",
    "dct",
    "window",
]
