"""The native (C++) log reaches Python's `logging`, and the Python logger controls the native level.

Everything here uses the FFT, which is cheap and needs no model: FFT(9) logs one DEBUG line ("Created FFT")
and inverse() with empty input logs one ERROR line before raising ValueError.
"""

from __future__ import annotations

import logging
import logging.config
import os
import subprocess
import sys
import threading

import pytest

import libspeech
from libspeech import _log

LOGGER = logging.getLogger("libspeech")


@pytest.fixture(autouse=True)
def fresh_bridge():
    _log.install()
    LOGGER.setLevel(logging.NOTSET)
    LOGGER.disabled = False
    yield
    LOGGER.setLevel(logging.NOTSET)
    LOGGER.disabled = False
    for handler in list(LOGGER.handlers):  # set_log_level() may have added the stderr handler
        LOGGER.removeHandler(handler)
    _log.install()  # another test may have replaced the callback


def messages(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == "libspeech"]


def make_debug_line() -> None:
    libspeech.FFT(9)  # DEBUG "Created FFT: ..."


def make_error_line() -> None:
    with pytest.raises(ValueError):
        libspeech.FFT(9).inverse([], [])  # ERROR "inverse() requires non-empty ..." then raises


def test_native_errors_arrive_with_no_setup(caplog):
    make_error_line()
    errors = [r for r in caplog.records if r.name == "libspeech" and r.levelno == logging.ERROR]
    assert any("inverse() requires non-empty" in r.getMessage() for r in errors)


def test_records_carry_the_native_origin(caplog):
    LOGGER.setLevel(logging.DEBUG)
    make_debug_line()
    record = next(r for r in caplog.records if "Created FFT" in r.getMessage())
    assert record.levelno == logging.DEBUG
    assert record.where == "speech::dsp::FFT"


def test_logger_level_turns_native_lines_on_and_off(caplog):
    LOGGER.setLevel(logging.DEBUG)
    make_debug_line()
    assert any("Created FFT" in m for m in messages(caplog))
    caplog.clear()
    LOGGER.setLevel(logging.WARNING)
    make_debug_line()
    assert not messages(caplog)


def test_logger_level_decides_the_native_level():
    LOGGER.setLevel(logging.DEBUG)
    assert libspeech.get_log_level() == "debug"
    LOGGER.setLevel(logging.WARNING)
    assert libspeech.get_log_level() == "warning"
    LOGGER.setLevel(logging.CRITICAL + 1)
    assert libspeech.get_log_level() == "off"


def test_root_logger_level_is_followed_too(caplog):
    root = logging.getLogger()
    saved = root.level
    try:
        root.setLevel(logging.DEBUG)  # what logging.basicConfig(level=DEBUG) does; the library logger only inherits it
        assert libspeech.get_log_level() == "debug"
        make_debug_line()
        assert any("Created FFT" in m for m in messages(caplog))
    finally:
        root.setLevel(saved)


def test_caplog_set_level_works(caplog):
    caplog.set_level(logging.DEBUG, logger="libspeech")
    make_debug_line()
    assert any("Created FFT" in m for m in messages(caplog))


def test_set_log_level_sets_the_logger(caplog):
    libspeech.set_log_level("debug")
    assert LOGGER.level == logging.DEBUG
    assert libspeech.get_log_level() == "debug"
    make_debug_line()
    assert any("Created FFT" in m for m in messages(caplog))
    libspeech.set_log_level("off")
    assert libspeech.get_log_level() == "off"
    caplog.clear()
    make_error_line()
    assert not messages(caplog)


def test_set_log_level_accepts_the_old_spellings():
    libspeech.set_log_level("WARN")
    assert libspeech.get_log_level() == "warning"
    libspeech.set_log_level(" None ")
    assert libspeech.get_log_level() == "off"
    libspeech.set_log_level("trace")
    assert libspeech.get_log_level() == "trace"
    assert LOGGER.level == 5


def test_unknown_level_raises_and_changes_nothing():
    LOGGER.setLevel(logging.ERROR)
    with pytest.raises(ValueError):
        libspeech.set_log_level("loud")
    with pytest.raises(TypeError):
        libspeech.set_log_level(10)  # type: ignore[arg-type]
    assert LOGGER.level == logging.ERROR
    assert libspeech.get_log_level() == "error"


def test_dictconfig_sets_the_native_level(caplog):
    logging.config.dictConfig(
        {"version": 1, "disable_existing_loggers": False, "loggers": {"libspeech": {"level": "DEBUG"}}}
    )
    assert libspeech.get_log_level() == "debug"
    make_debug_line()
    assert any("Created FFT" in m for m in messages(caplog))


def test_a_disabled_logger_silences_the_library(caplog):
    # The documented pitfall: dictConfig() sets `disabled` on loggers it does not mention unless it is told
    # `disable_existing_loggers: False`.
    LOGGER.setLevel(logging.DEBUG)
    LOGGER.disabled = True
    make_debug_line()
    assert not messages(caplog)


@pytest.fixture
def restore_disable():
    yield
    logging.disable(logging.NOTSET)


def test_logging_disable_turns_the_native_level_off(caplog, restore_disable):
    # logging.disable(...) drops every record below the given level, whatever the logger level is. The native
    # side must not build lines that Python would only drop (each one also costs a GIL acquisition).
    LOGGER.setLevel(logging.DEBUG)
    logging.disable(logging.CRITICAL)
    assert libspeech.get_log_level() == "off"
    make_error_line()
    assert not messages(caplog)
    logging.disable(logging.NOTSET)
    assert libspeech.get_log_level() == "debug"
    make_debug_line()
    assert any("Created FFT" in m for m in messages(caplog))


def test_logging_disable_keeps_what_it_lets_through(restore_disable):
    LOGGER.setLevel(logging.DEBUG)
    logging.disable(logging.WARNING)  # drops everything up to and including WARNING
    assert libspeech.get_log_level() == "error"
    logging.disable(logging.DEBUG)
    assert libspeech.get_log_level() == "info"


def test_a_level_set_while_logging_is_disabled_respects_the_disable(restore_disable):
    logging.disable(logging.WARNING)
    LOGGER.setLevel(logging.DEBUG)
    assert libspeech.get_log_level() == "error"


@pytest.mark.parametrize(
    ("number", "native"),
    [(0, "trace"), (5, "trace"), (10, "debug"), (15, "info"), (20, "info"), (25, "warning"), (30, "warning"),
     (40, "error"), (50, "off")],
)
def test_level_mapping(number, native):
    assert _log.native_level_name(number) == native


def test_explicit_callback_replaces_the_bridge(caplog):
    got = []
    libspeech.set_log_level("debug")
    libspeech.set_log_callback(lambda level, where, message: got.append((level, where, message)))
    make_debug_line()
    assert any(level == "debug" and where == "speech::dsp::FFT" and "Created FFT" in msg for level, where, msg in got)
    assert not messages(caplog)
    libspeech.set_log_callback(None)  # back to the native stderr output


def test_a_failing_callback_does_not_break_the_library():
    libspeech.set_log_level("debug")

    def boom(level, where, message):
        msg = "expected test output: callback failure"
        raise RuntimeError(msg)

    libspeech.set_log_callback(boom)
    assert libspeech.FFT(9) is not None  # the call goes through; the exception stays in the callback


def test_callback_must_be_callable():
    with pytest.raises(TypeError):
        libspeech.set_log_callback(42)  # type: ignore[arg-type]


def test_log_file_gets_the_lines(tmp_path):
    path = tmp_path / "native.log"
    libspeech.set_log_level("debug")
    assert libspeech.set_log_file(str(path)) is True
    make_debug_line()
    assert libspeech.set_log_file("") is True
    text = path.read_text()
    assert "Created FFT" in text
    assert "speech::dsp::FFT" in text


def test_log_file_that_cannot_be_opened_returns_false(tmp_path):
    assert libspeech.set_log_file(str(tmp_path / "no_such_dir" / "native.log")) is False


def test_lines_from_other_threads_arrive(caplog):
    LOGGER.setLevel(logging.DEBUG)
    threads = [threading.Thread(target=make_debug_line) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum("Created FFT" in m for m in messages(caplog)) == 4


_SCRIPT = """
import logging, sys
import libspeech
{setup}
libspeech.FFT(9)
"""


def _run(setup: str, env_log: str | None = None) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != "LIBSPEECH_LOG"}
    if env_log is not None:
        env["LIBSPEECH_LOG"] = env_log
    return subprocess.run(
        [sys.executable, "-c", _SCRIPT.format(setup=setup)],
        capture_output=True, text=True, timeout=120, check=False, env=env,
    )


def test_application_logging_configuration_is_respected():
    # The app configured logging itself, so set_log_level() must not add a second handler (no duplicate lines).
    r = _run(
        'logging.basicConfig(level=logging.WARNING, format="APP %(name)s %(message)s")\n'
        'libspeech.set_log_level("debug")'
    )
    assert r.returncode == 0, r.stderr
    lines = [ln for ln in r.stderr.splitlines() if "Created FFT" in ln]
    assert len(lines) == 1
    assert lines[0].startswith("APP libspeech ")


def test_basicconfig_after_set_log_level_does_not_print_twice():
    # set_log_level() found no logging configured and added its own stderr handler; the application configures
    # logging afterwards. From then on only the application's handler may print.
    r = _run(
        'libspeech.set_log_level("debug")\n'
        'logging.basicConfig(level=logging.DEBUG, format="APP %(name)s %(message)s")'
    )
    assert r.returncode == 0, r.stderr
    lines = [ln for ln in r.stderr.splitlines() if "Created FFT" in ln]
    assert len(lines) == 1
    assert lines[0].startswith("APP libspeech ")


def test_a_handler_added_to_the_library_logger_after_set_log_level_is_the_only_one_printing():
    r = _run(
        'libspeech.set_log_level("debug")\n'
        "handler = logging.StreamHandler(sys.stderr)\n"
        'handler.setFormatter(logging.Formatter("APP %(message)s"))\n'
        'logging.getLogger("libspeech").addHandler(handler)'
    )
    assert r.returncode == 0, r.stderr
    lines = [ln for ln in r.stderr.splitlines() if "Created FFT" in ln]
    assert len(lines) == 1
    assert lines[0].startswith("APP ")


def test_set_log_level_alone_still_prints_once_in_the_library_format():
    r = _run('libspeech.set_log_level("debug")')
    assert r.returncode == 0, r.stderr
    lines = [ln for ln in r.stderr.splitlines() if "Created FFT" in ln]
    assert len(lines) == 1
    assert "[DEBUG]" in lines[0]


def test_basicconfig_after_import_turns_native_lines_on():
    r = _run('logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")')
    assert r.returncode == 0, r.stderr
    assert any(ln.startswith("DEBUG libspeech: Created FFT") for ln in r.stderr.splitlines())


def test_environment_variable_turns_the_logger_on():
    r = _run("", env_log="debug")
    assert "Created FFT" in r.stderr
