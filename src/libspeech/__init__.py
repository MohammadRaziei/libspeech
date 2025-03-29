from __future__ import annotations
from pathlib import Path

from ctypes import *

cdll.LoadLibrary(Path(__file__).parent.joinpath("libonnxruntime.so.1.21.0").as_posix())
cdll.LoadLibrary(Path(__file__).parent.joinpath("libaudioflux.so").as_posix())
cdll.LoadLibrary(Path(__file__).parent.joinpath("libspeech.so").as_posix())


from ._about import __version__

from ._audio import Audio
