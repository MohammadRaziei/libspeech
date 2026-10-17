"""Free-threaded CPython (3.13t and newer): importing libspeech must not turn the GIL back on.

An extension module that does not declare it can run without the GIL makes the interpreter
re-enable it on import (with a RuntimeWarning), which silently removes the point of a
free-threaded interpreter. The modules declare it through nanobind's FREE_THREADED flag
(src/bindings/python/CMakeLists.txt). On a regular interpreter there is nothing to check.
"""

from __future__ import annotations

import os
import subprocess
import sys
import sysconfig

import pytest

pytestmark = [
    pytest.mark.skipif(
        not sysconfig.get_config_var("Py_GIL_DISABLED"),
        reason="not a free-threaded CPython build",
    ),
    pytest.mark.skipif(
        os.environ.get("PYTHON_GIL") == "1",
        reason="PYTHON_GIL=1 forces the GIL on, so the check would say nothing",
    ),
]

# `import libspeech` loads every extension module the package uses (_about, speech_dsp_py,
# speech_io_py, speech_models_py).
IMPORT_ALL = "import sys, libspeech\nprint(sys._is_gil_enabled())"


def test_importing_libspeech_keeps_the_gil_off():
    out = subprocess.run(
        [sys.executable, "-W", "error::RuntimeWarning", "-c", IMPORT_ALL],
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == "False"
