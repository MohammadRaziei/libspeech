# ADR-0014: C++ log: own SPEECH_LOG on AixLog, level checked first, default warning, three outputs

- **Status:** accepted
- **Date:** 2026-10-14 and 2026-10-20
- **Source:** commits 65d08cf (public logging API) and ff444f1 (2026-10-20, unified with Python `logging`), `include/libspeech/detail/log.h`, `include/libspeech/utils/log.h`

## Context

Library code must be able to log without taking over the host application's logging, and a disabled message must cost almost nothing.

## Decision

- `SPEECH_LOG(SEV) << TAG(tag) << ...` (`include/libspeech/detail/log.h`). `TAG` and `COND` are AixLog helpers. The macro checks the level first with one atomic load, so a disabled message is never built.
- Lines are written through AixLog's own sinks (`SinkCerr`, `SinkFile`) via a private stream. The vendored AixLog carries one patch: it no longer points `std::clog` at itself for the whole process. The `LOG(...)` macros of AixLog therefore do not reach our sinks and are not used.
- Format: `time [Severity] (tag or function) message`.
- Default level is `warning` (it printed everything before 0.4.0). Control: `speech::utils::set_log_level`, `libspeech.set_log_level`, the `"libspeech"` Python logger (ADR-0015), or `LIBSPEECH_LOG` (read once, until the first `set_log_level`).
- Three outputs, one active at a time: stderr (default), a file (`set_log_file`), or a callback (`set_log_callback(level, where, message)`). Setting one replaces the previous.
- A callback runs on whatever thread logs, possibly several at once, and **never under a lock**: the Python callback needs the GIL, and a thread holding the GIL may be waiting to log. It must not throw or call back into the library.
- State is inline in the header because `speech_dsp` must work alone in a DSP-only build (ADR-0001); the public functions are in `utils.cpp`.

## Rejected alternatives

AixLog's `LOG` macros and its `std::clog` redirection; holding a lock while calling the callback.

## Consequences

- Severity names in the native stderr and file format are AixLog's (`Debug`, `Warn`, `Error`), not the lowercase names used before 2026-10-20.
- Tested on Linux only (C++: `tests/models/test_log.cpp`, including a regression test for a Windows-style `ERROR` macro). Windows/MSVC and macOS were not run for this change.
