"""Run one pytest file whose module skips itself when its optional dependencies are missing.

pytest then exits with 5 ("no tests ran"); for such a file that is a skip, not a failure.
"""

from __future__ import annotations

import sys

import pytest

code = pytest.main(sys.argv[1:])
sys.exit(0 if code == pytest.ExitCode.NO_TESTS_COLLECTED else int(code))
