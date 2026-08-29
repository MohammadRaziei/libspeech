# aixlog (vendored, single header)

`include/aixlog.hpp` is copied directly from
[badaix/aixlog](https://github.com/badaix/aixlog) (MIT licensed, see
`LICENSE`). This used to be a full git submodule; since aixlog is a
single-header library and we only ever use that one header, it's vendored
here directly instead, to avoid an extra submodule clone/init step for
something this small.

Unmodified from upstream. If libspeech ever needs to patch this header,
document the change the same way as `/audioflux_issues.md` does for
AudioFlux.
