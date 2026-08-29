# Vendored third-party sources

Everything under `src/vendor/` is third-party code copied directly into
this repo (not a git submodule), each in its own subfolder with its own
README/LICENSE:

- **`audioflux/`** — a subset of [libAudioFlux/audioflux](https://github.com/libAudioFlux/audioflux)'s
  C sources, powering `speech::dsp::Resample`, `speech::dsp::window`,
  `speech::dsp::FFT`, and `speech::dsp::STFT`. See `audioflux/README.md` and
  `/audioflux_issues.md` (project root) for what's vendored and why.
- **`aixlog/`** — [badaix/aixlog](https://github.com/badaix/aixlog)'s single
  header (`aixlog.hpp`), used project-wide for logging. See `aixlog/README.md`.

Both used to be git submodules; they're vendored directly here instead
because in each case only a small, specific slice of the upstream project is
actually needed, and the vendoring workflow (copy what's needed, patch in
place if needed, document any patch) is documented per-folder.
