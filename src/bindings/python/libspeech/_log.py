"""Routes libspeech's own (C++) log into Python's ``logging``, and keeps both levels in step.

The Python logger ``logging.getLogger("libspeech")`` is the single control:

* every line the native library writes arrives there, with no setup;
* ``setLevel`` on that logger (or on the root logger it inherits from), ``logging.basicConfig(level=...)``,
  ``logging.config.dictConfig``, pytest's ``caplog.set_level`` and :func:`set_log_level` all end up at the
  same place, and the native level follows.

The native side builds a log line only if its level lets it through, and every line that is sent costs a GIL
acquisition. So the native level is never left at ``trace`` "for Python to filter": it is set to the smallest
native level that lets through exactly what the logger would keep.
"""

from __future__ import annotations

import atexit
import logging
import os
import types

from . import speech_models_py as _native

LOGGER = logging.getLogger("libspeech")

_TRACE = 5
_PY_LEVEL = {
    "trace": _TRACE,
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}
_LIMITS = tuple(_PY_LEVEL.items())
_ALIASES = {"warn": "warning", "none": "off"}

if logging.getLevelName(_TRACE) == f"Level {_TRACE}":  # nobody named it yet
    logging.addLevelName(_TRACE, "TRACE")


def native_level_name(number: int) -> str:
    """The native level that lets exactly the lines through that a logger at ``number`` would keep."""
    for name, limit in _LIMITS:
        if number <= limit:
            return name
    return "off"  # CRITICAL and above: the native side has nothing that severe


def sync() -> None:
    """Make the native level follow the logger's effective level."""
    _native.set_log_level(native_level_name(LOGGER.getEffectiveLevel()))


def _emit(level: str, where: str, message: str) -> None:
    # Called by the native library, on whatever thread logs, with the GIL held.
    LOGGER.log(_PY_LEVEL.get(level, logging.ERROR), "%s", message, extra={"where": where})


def _set_level(self: logging.Logger, level: int | str) -> None:
    logging.Logger.setLevel(self, level)
    sync()


def _normalize(level: str) -> str:
    if not isinstance(level, str):
        message = f"level must be a string, not {type(level).__name__}"
        raise TypeError(message)
    name = level.strip().lower()
    name = _ALIASES.get(name, name)
    if name != "off" and name not in _PY_LEVEL:
        message = f"Unknown log level {level!r} (use trace, debug, info, warning, error or off)."
        raise ValueError(message)
    return name


class _Formatter(logging.Formatter):
    """``2026-10-09 12:00:00.123 [DEBUG] (where) message``; ``where`` is the native tag or function."""

    def __init__(self) -> None:
        super().__init__(
            "%(asctime)s.%(msecs)03d [%(levelname)s] (%(where)s) %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, "where"):
            record.where = record.funcName
        return super().format(record)


def ensure_handler() -> None:
    """If the application configured no logging at all, print the library's log to stderr.

    Used when the user asks for output explicitly (``set_log_level``, ``LIBSPEECH_LOG``). Without it a
    ``debug`` or ``info`` line would hit Python's last-resort handler, which only prints warnings and above.
    """
    if LOGGER.handlers or logging.getLogger().handlers:
        return
    handler = logging.StreamHandler()  # stderr
    handler.setFormatter(_Formatter())
    LOGGER.addHandler(handler)


def set_log_level(level: str) -> None:
    """Sets how much libspeech logs: ``"trace"``, ``"debug"``, ``"info"``, ``"warning"`` (the default),
    ``"error"`` or ``"off"`` (case-insensitive; ``"warn"`` and ``"none"`` also work).

    This sets the level of ``logging.getLogger("libspeech")``, so the Python logger and the native level
    stay one thing. If the application has configured no logging, a handler that prints to stderr is added so
    that the lines are visible. Raises ValueError for an unknown level.
    """
    name = _normalize(level)
    if name == "off":
        LOGGER.setLevel(logging.CRITICAL + 1)
        return
    LOGGER.setLevel(_PY_LEVEL[name])
    ensure_handler()


def get_log_level() -> str:
    """The current native log level, e.g. ``"warning"``."""
    return _native.get_log_level()


def install() -> None:
    """Route the native log into ``logging`` and keep the levels in step. Safe to call again."""
    # Wrap setLevel once, on the library's logger and on the root logger it inherits from: a level that is only
    # inherited is never announced by `logging`, and basicConfig(level=...), dictConfig and caplog all end up
    # in one of these two setLevel calls.
    for logger in (LOGGER, logging.getLogger()):
        if "setLevel" not in logger.__dict__:
            logger.setLevel = types.MethodType(_set_level, logger)
    _native.set_log_callback(_emit)
    requested = os.environ.get("LIBSPEECH_LOG")
    if requested and requested.strip():
        # The native side already parsed it (and complained about an invalid value); take its answer.
        set_log_level(_native.get_log_level())
    else:
        sync()


def _uninstall_at_exit() -> None:
    # Native threads may still log while the interpreter shuts down; from here on they go to stderr.
    _native.set_log_callback(None)


atexit.register(_uninstall_at_exit)
