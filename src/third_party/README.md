# Third-party dependencies

Everything under `src/third_party/` is external code the project depends on.
It's organized into two kinds, side by side in this same directory:

## Git submodules

- **`miniaudio/`** — [mackron/miniaudio](https://github.com/mackron/miniaudio), used by `speech::io`.
- **`dr_libs/`** — [mackron/dr_libs](https://github.com/mackron/dr_libs), used by `speech::io`.

These stay as thin submodule pointers (see `.gitmodules`) since the full
upstream project is needed as-is. Initialize them with:

```bash
git submodule update --init src/third_party/miniaudio src/third_party/dr_libs
```

## Copied in place (not submodules)

- **`audioflux/`** — a subset of [libAudioFlux/audioflux](https://github.com/libAudioFlux/audioflux)'s
  C sources, powering `speech::dsp::Resample`, `speech::dsp::window`,
  `speech::dsp::FFT`, and `speech::dsp::STFT`. See `audioflux/README.md` and
  `/audioflux_issues.md` (project root) for what's vendored and why.
- **`aixlog/`** — [badaix/aixlog](https://github.com/badaix/aixlog)'s single
  header (`aixlog.hpp`), used project-wide for logging. See `aixlog/README.md`.

These two are copied directly into the repo instead of being submodules
because in each case only a small, specific slice of the upstream project is
actually needed. The vendoring workflow (copy what's needed, patch in place
if needed, document any patch) is documented per-folder.
