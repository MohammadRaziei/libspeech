# aixlog (vendored, single header)

`include/aixlog.hpp` is copied directly from
[badaix/aixlog](https://github.com/badaix/aixlog) (MIT licensed, see
`LICENSE`). This used to be a full git submodule; since aixlog is a
single-header library and we only ever use that one header, it's vendored
here directly instead, to avoid an extra submodule clone/init step for
something this small.

Carries **one patch**, marked `[libspeech patch]` in the file: upstream's `Log` constructor points
`std::clog` at itself for the whole process, which a library must not do (it would swallow the host
application's own `std::clog` output), so it writes through a private `std::ostream` instead. Consequence: the
`LOG(...)` macros, which write to `std::clog`, no longer reach the sinks. libspeech does not use them: it writes
a line into AixLog's streambuf through its own ostream (`include/libspeech/detail/log.h`) and uses AixLog's
`SinkCerr` / `SinkFile` for the output and its `Tag` / `Conditional` helpers (`TAG(...)`, `COND(...)`).
The same patch is used by [httpp](https://github.com/MohammadRaziei/httpp).

To upgrade: take the new upstream file and redo that one change.
Any further change must be documented the same way as `../audioflux/ISSUES.md` does for AudioFlux.
