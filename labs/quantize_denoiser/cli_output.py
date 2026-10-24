"""Console output for the command-line tools."""

from __future__ import annotations

import sys


def say(text: str = "") -> None:
    sys.stdout.write(f"{text}\n")
    sys.stdout.flush()
