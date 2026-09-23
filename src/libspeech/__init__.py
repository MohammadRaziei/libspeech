from __future__ import annotations

from ctypes import *
from pathlib import Path

_here = Path(__file__).parent

# ONNXRuntime and speech itself each ship as a separate shared library we
# load explicitly before importing the compiled extensions (they dlopen
# symbols from these at import time). Both get bundled right next to this
# __init__.py by CMakeLists.txt (see the SKBUILD install() rule / the
# local dev build's LIBRARY_OUTPUT_DIRECTORY) rather than found via
# RPATH: an RPATH baked in at build time would only point back at
# wherever the ONNXRuntime download happened to land on the *build*
# machine, which has no reason to exist -- or be the same path -- on
# whatever machine later installs this wheel. Loading a bundled copy by
# explicit path here works regardless of where/how this package itself
# ends up installed.
#
# Each filename varies by platform (libspeech.so / speech.dll /
# libspeech.dylib), and ONNXRuntime's is version-suffixed on top of that
# (e.g. libonnxruntime.so.1.21.0). Rather than guess any of that here
# (hardcoding one platform's name, or globbing for a pattern that only
# happens to match some platforms), CMake computes the exact filename it
# actually produced for each one and bakes it into _about -- see the
# target_compile_definitions() call on the _about target in
# CMakeLists.txt, and about.cpp. _about is itself a tiny, dependency-free
# compiled module (no link against speech/httpp/ONNXRuntime), so it's
# safe to import before any of those are loaded.
from ._about import HTTPP_LIB_PATH, LIB_PATH, ONNXRUNTIME_LIB_PATH, __version__

# httpp is a separate package (pip install httpp), not bundled inside
# libspeech -- loaded straight from wherever the installed httpp package
# actually lives, via httpp's own get_lib_dir() helper, so libspeech
# always runs against whatever httpp build is really installed rather
# than a copy shipped in here. This is a normal Python import, so it
# works the same way on every platform and in a local dev build too (no
# site-packages layout to assume, no symlink needed) -- httpp only has to
# be importable, same as any other Python dependency.
try:
    import httpp
except ImportError as e:
    raise ImportError(
        "libspeech requires the 'httpp' package. Install it with: pip install httpp"
    ) from e

# HTTPP_LIB_PATH is only the filename (computed by CMake, see above);
# httpp.get_lib_dir() is httpp's own answer for where that filename lives.
cdll.LoadLibrary(str(Path(httpp.get_lib_dir()) / HTTPP_LIB_PATH))

cdll.LoadLibrary(_here.joinpath(ONNXRUNTIME_LIB_PATH).as_posix())

# NOTE: AudioFlux is statically linked into libspeech.so itself (see
# CMakeLists.txt: speech_dsp/speech_models are STATIC libraries), so there
# is no separate libaudioflux.so to preload here -- only libspeech.so
# itself needs an explicit load (httpp and ONNXRuntime, both dynamically
# linked, were already preloaded above).
cdll.LoadLibrary(_here.joinpath(LIB_PATH).as_posix())
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
