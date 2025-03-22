#!/bin/python3
from __future__ import annotations

import warnings
from pathlib import Path


# ctypes.CDLL(Path(__file__).parent/"libspeech.so")
from ._audio import Audio



def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return f'{filename}:{lineno}: {category.__name__}: {message}\n'



