from __future__ import annotations

import os
import struct
import sys
from ctypes import *
from pathlib import Path

_here = Path(__file__).parent


def _pe_direct_imports(path: Path) -> list[str]:
    """The list of DLL names a Windows PE file (.dll/.exe) directly imports.

    A tiny, dependency-free reimplementation of the one thing we actually
    need from a full PE-parsing library: the direct import table.
    Cross-checked against `pefile`'s own output on real DLLs (httpp_core.dll,
    onnxruntime.dll) and it matches exactly -- not worth a third-party
    dependency (even a Windows/test-only one) for ~30 lines of struct
    unpacking against a format (PE/COFF) that's been stable since Win95.
    """
    data = path.read_bytes()
    if data[:2] != b"MZ":
        msg = "not a PE file (missing MZ header)"
        raise ValueError(msg)
    e_lfanew = struct.unpack_from("<I", data, 0x3C)[0]
    if data[e_lfanew : e_lfanew + 4] != b"PE\0\0":
        msg = "not a PE file (missing PE signature)"
        raise ValueError(msg)

    coff_off = e_lfanew + 4
    (num_sections,) = struct.unpack_from("<H", data, coff_off + 2)
    (size_opt_hdr,) = struct.unpack_from("<H", data, coff_off + 16)
    opt_off = coff_off + 20

    (magic,) = struct.unpack_from("<H", data, opt_off)
    # DataDirectory[] sits right after the standard+Windows-specific
    # optional header fields, whose total size differs between PE32 (32-bit
    # fields for image/stack/heap sizes) and PE32+ (64-bit) -- everything
    # else about walking the import table is identical either way.
    if magic == 0x10B:  # PE32
        data_dir_off = opt_off + 96
    elif magic == 0x20B:  # PE32+
        data_dir_off = opt_off + 112
    else:
        msg = f"unknown optional header magic {magic:#x}"
        raise ValueError(msg)

    # DataDirectory[1] is always the Import Table {RVA, Size}, PE32 or PE32+.
    import_rva, _import_size = struct.unpack_from("<II", data, data_dir_off + 8)
    if import_rva == 0:
        return []

    sections = []
    section_off = opt_off + size_opt_hdr
    for i in range(num_sections):
        off = section_off + i * 40
        vsize, vaddr = struct.unpack_from("<II", data, off + 8)
        raw_size, raw_ptr = struct.unpack_from("<II", data, off + 16)
        sections.append((vaddr, vsize, raw_ptr, raw_size))

    def rva_to_offset(rva: int) -> int:
        for vaddr, vsize, raw_ptr, raw_size in sections:
            if vaddr <= rva < vaddr + max(vsize, raw_size):
                return raw_ptr + (rva - vaddr)
        msg = f"RVA {rva:#x} not in any section"
        raise ValueError(msg)

    names = []
    descriptor_off = rva_to_offset(import_rva)
    while True:
        # IMAGE_IMPORT_DESCRIPTOR: OriginalFirstThunk, TimeDateStamp,
        # ForwarderChain, Name (RVA), FirstThunk -- 5 x uint32, and the
        # array ends with one all-zero entry.
        name_rva = struct.unpack_from("<5I", data, descriptor_off)[3]
        if name_rva == 0:
            break
        name_off = rva_to_offset(name_rva)
        end = data.index(b"\0", name_off)
        names.append(data[name_off:end].decode("ascii"))
        descriptor_off += 20
    return names


def _load_library(path: Path) -> None:
    """cdll.LoadLibrary(), with actionable diagnostics on Windows.

    Windows' LoadLibrary gives no way to ask *which* dependency of `path`
    is unresolvable -- "Could not find module ... (or one of its
    dependencies)" is the whole message, every time, whether it's `path`
    itself or something three levels down its import table. This isn't
    something CMake or our own C++ could tell us instead: it's not about
    what *we* linked `path` against (we already know and preload that,
    see below) -- it's about the compiler-injected C/C++ runtime imports
    (e.g. VCRUNTIME140_1.dll, MSVCP140_1.dll) that MSVC adds on its own
    based on which standard library features got used, whose actual
    presence depends entirely on the *installing* machine's VC++
    Redistributable version -- something no build-time tool can see,
    since it isn't known until this exact line runs, on that machine.
    Reading `path`'s own import table (see _pe_direct_imports above) is
    the only way to name the real culprit instead of guessing again.
    """
    try:
        cdll.LoadLibrary(str(path))
    except FileNotFoundError as e:
        if sys.platform != "win32":
            raise
        if not path.is_file():
            # Windows' error message is identical either way ("could not
            # find module X (or one of its dependencies)"), so check the
            # simpler explanation first: X itself was never actually
            # there, rather than assuming it's a real dependency and
            # going straight to PE-parsing to find out which one.
            msg = f"Failed to load {path}: that file doesn't exist (this is a packaging bug, not a missing dependency)"
            raise ImportError(msg) from e
        try:
            imports = _pe_direct_imports(path)
        except Exception as parse_exc:
            msg = (
                f"Failed to load {path} (or one of its dependencies); additionally failed to inspect "
                f"its import table: {parse_exc!r}"
            )
            raise ImportError(msg) from e
        search_dirs = [path.parent, Path(os.environ.get("SYSTEMROOT", "C:/Windows")) / "System32"]
        search_dirs += [Path(p) for p in os.environ.get("PATH", "").split(os.pathsep) if p]
        missing = sorted({name for name in imports if not any((d / name).is_file() for d in search_dirs)})
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
