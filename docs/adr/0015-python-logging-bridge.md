# ADR-0015: The native log goes through Python's `logging`, and the logger controls the native level

- **Status:** accepted
- **Date:** 2026-10-20
- **Source:** `src/bindings/python/libspeech/_log.py`, `tests/python/test_log_bridge.py`; pattern taken from httpp ADR-0018

## Context

A Python application configures logging with `logging`; the native log had its own level, file and callback, so users had to bridge it by hand.

## Decision

- Importing `libspeech` installs a callback that sends every native line to `logging.getLogger("libspeech")`; each record has `record.where` (the native tag or function).
- **The logger decides the native level**: the native level follows `logger.getEffectiveLevel()` (the smallest native level that lets through what the logger keeps; `trace` is 5; CRITICAL and above is `off`). It is never left at `trace` for Python to filter, because every sent line costs a GIL acquisition.
- `setLevel` is wrapped on the `"libspeech"` logger **and on the root logger**, so `basicConfig(level=...)`, `dictConfig`, `caplog.set_level` and `libspeech.set_log_level` all resync the native level. `logging.disable(...)` is not seen.
- `libspeech.set_log_level` sets the logger level and adds a stderr handler when the application configured no logging; `LIBSPEECH_LOG` does the same at import.
- An explicit `set_log_callback` or `set_log_file` replaces the bridge (one output at a time); `set_log_callback(None)` goes back to native stderr. At interpreter exit the bridge is removed so late native threads do not call into a finished interpreter.
- The callback takes the GIL, swallows every exception, and the callable is released under the GIL.

## Rejected alternatives

An opt-in `use_logging()` call; a native level fixed at `trace` with Python filtering; a handler that tries to detect level changes (`logging` has no such hook).

## Consequences

- With no logging configured, only warnings and errors appear, as message text only (Python's last-resort handler). If the application calls `basicConfig` after `set_log_level` added the stderr handler, lines print twice.
- `dictConfig` with the default `disable_existing_loggers=True` disables the logger and silences the library; pass `False`.
- The level is per process.
- 29 tests in `test_log_bridge.py`; six deliberate bugs in the bridge were each caught by a test (mutation check). Not run on Windows or macOS.
