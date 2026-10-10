"""Tests for libspeech's log-level control (set_log_level / get_log_level / LIBSPEECH_LOG).

What reaches stderr is process-wide state, so the output tests run in a subprocess with a controlled
environment. They only use the FFT (cheap, no model needed): FFT(9) logs one DEBUG line and the
invalid FFT(99) logs one ERROR line before raising.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

import libspeech

_SCRIPT = """
import sys
import libspeech
{setup}
libspeech.FFT(9)            # logs a DEBUG line ("Created FFT")
try:
    libspeech.FFT(99)       # logs an ERROR line, then raises
except ValueError:
    pass
print(libspeech.get_log_level())
"""


def _run(setup: str = "", env_log: str | None = None) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != "LIBSPEECH_LOG"}
    if env_log is not None:
        env["LIBSPEECH_LOG"] = env_log
    return subprocess.run(
        [sys.executable, "-c", _SCRIPT.format(setup=setup)],
        capture_output=True, text=True, timeout=120, check=False, env=env,
    )


@pytest.fixture
def restore_level():
    saved = libspeech.get_log_level()
    yield
    libspeech.set_log_level(saved)


def test_set_and_get_round_trip(restore_level):
    for name in ("trace", "debug", "info", "warning", "error", "off"):
        libspeech.set_log_level(name)
        assert libspeech.get_log_level() == name


def test_level_names_are_case_insensitive_and_warn_is_an_alias(restore_level):
    libspeech.set_log_level("DEBUG")
    assert libspeech.get_log_level() == "debug"
    libspeech.set_log_level("warn")
    assert libspeech.get_log_level() == "warning"


def test_unknown_level_raises_and_keeps_the_current_one(restore_level):
    libspeech.set_log_level("error")
    with pytest.raises(ValueError):
        libspeech.set_log_level("verbose")
    assert libspeech.get_log_level() == "error"


def test_default_is_quiet_for_debug_but_still_reports_errors():
    r = _run()
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "warning"
    assert "Created FFT" not in r.stderr  # fine-grained DEBUG output is off by default
    assert "[error]" in r.stderr          # errors still show


def test_off_writes_nothing_to_stderr():
    r = _run('libspeech.set_log_level("off")')
    assert r.returncode == 0, r.stderr
    assert r.stderr == ""


def test_debug_shows_debug_lines_in_a_readable_format():
    r = _run('libspeech.set_log_level("debug")')
    assert r.returncode == 0, r.stderr
    line = next(ln for ln in r.stderr.splitlines() if "Created FFT" in ln)
    assert "[debug]" in line  # severity, tag and message are space separated
    assert "(speech::dsp::FFT)" in line
    assert "[error]" in r.stderr


def test_environment_variable_sets_the_initial_level():
    assert _run(env_log="off").stderr == ""
    r = _run(env_log="debug")
    assert "Created FFT" in r.stderr
    assert r.stdout.strip() == "debug"


def test_set_log_level_overrides_the_environment_variable():
    r = _run('libspeech.set_log_level("off")', env_log="debug")
    assert r.stderr == ""


def test_invalid_environment_variable_is_reported_and_ignored():
    r = _run(env_log="loud")
    assert r.returncode == 0, r.stderr
    assert "ignoring invalid LIBSPEECH_LOG" in r.stderr
    assert r.stdout.strip() == "warning"
