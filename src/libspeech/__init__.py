from __future__ import annotations

import os
import sys
from ctypes import *
from pathlib import Path

_here = Path(__file__).parent


def _load_library(path: Path) -> None:
    """cdll.LoadLibrary(), with actionable diagnostics on Windows.

    Windows' LoadLibrary gives no way to ask *which* dependency of `path`
    is unresolvable -- "Could not find module ... (or one of its
    dependencies)" is the whole message, every time, whether it's `path`
    itself or something three levels down its import table. Rather than
    keep guessing blind from that one sentence, use pefile (pure Python,
    no compiler/Windows-SDK needed) to read `path`'s own direct import
    table and report by name exactly which of those aren't resolvable
    anywhere on the search path -- turning a CI failure into an actual
    answer instead of another round of speculation.
    """
    try:
        cdll.LoadLibrary(str(path))
    except FileNotFoundError as e:
        if sys.platform != "win32":
            raise
        try:
            import pefile  # noqa: PLC0415 -- optional, Windows-only diagnostic dependency; a top-level import would make it a hard dependency for every platform/success path instead of a lazy one only paid for when this exact failure happens

            pe = pefile.PE(str(path), fast_load=True)
            pe.parse_data_directories(
                directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]]
            )
            search_dirs = [path.parent, Path(os.environ.get("SYSTEMROOT", "C:/Windows")) / "System32"]
            search_dirs += [Path(p) for p in os.environ.get("PATH", "").split(os.pathsep) if p]
            missing = sorted(
                {
                    entry.dll.decode("ascii", "replace")
                    for entry in pe.DIRECTORY_ENTRY_IMPORT
                    if not any((d / entry.dll.decode("ascii", "replace")).is_file() for d in search_dirs)
                }
            )
        except Exception:
            msg = (
                f"Failed to load {path} (or one of its dependencies), and "
                "could not inspect it for more detail -- install 'pefile' "
                "(pip install pefile) and retry for a specific answer."
            )
            raise ImportError(msg) from e
        if not missing:
            # Every direct import resolves by our own (best-effort) search
            # -- Windows still refused to load it, so the real problem is
            # something our simple existence check can't see (architecture
            # mismatch, a transitive dependency two levels down, etc).
            msg = (
                f"Failed to load {path}, but all of its direct dependencies "
                "were found on disk -- the actual problem is likely an "
                "architecture mismatch or a transitive (indirect) "
                "dependency; re-run with Dependencies.exe or dumpbin "
                "/dependents for the full picture."
            )
            raise ImportError(msg) from e
        msg = f"Failed to load {path}: could not resolve dependencies: {', '.join(missing)}"
        raise ImportError(msg) from e


# On Windows, loading one DLL (e.g. speech.dll) that itself implicitly
# needs another (httpp_core.dll, onnxruntime.dll) only searches a fixed
# set of directories for that dependency -- the loading DLL's own
# directory, System32, PATH, etc -- which does NOT include wherever we
# explicitly cdll.LoadLibrary() a dependency FROM if that happens to be
# some other, unrelated directory (httpp's own install dir is nowhere
# near libspeech's). os.add_dll_directory() (the mechanism Python 3.8+
# itself introduced to replace the old implicit-PATH/CWD search it
# removed for security reasons) adds a directory to that search list for
# the rest of the process, so speech.dll's own implicit dependency
# resolution can actually find libhttpp_core.so/httpp_core.dll wherever
# httpp really lives, not just next to speech.dll itself.
if sys.platform == "win32":
    os.add_dll_directory(str(_here))

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
    msg = "libspeech requires the 'httpp' package. Install it with: pip install httpp"
    raise ImportError(msg) from e

_httpp_lib_dir = Path(httpp.get_lib_dir())
if sys.platform == "win32":
    os.add_dll_directory(str(_httpp_lib_dir))

# HTTPP_LIB_PATH is only the filename (computed by CMake, see above);
# httpp.get_lib_dir() is httpp's own answer for where that filename lives.
_load_library(_httpp_lib_dir / HTTPP_LIB_PATH)

_load_library(_here / ONNXRUNTIME_LIB_PATH)

# NOTE: AudioFlux is statically linked into libspeech.so itself (see
# CMakeLists.txt: speech_dsp/speech_models are STATIC libraries), so there
# is no separate libaudioflux.so to preload here -- only libspeech.so
# itself needs an explicit load (httpp and ONNXRuntime, both dynamically
# linked, were already preloaded above).
_load_library(_here / LIB_PATH)
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
