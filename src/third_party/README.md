# Third-party dependencies

Everything under `src/third_party/` is external code the project depends on.
It's organized into two kinds, side by side in this same directory:

## Git submodules

- **`miniaudio/`** — [mackron/miniaudio](https://github.com/mackron/miniaudio), used by `speech::io`
  for both audio file decode/encode (WAV/MP3/FLAC, via `ma_decoder`/
  `ma_encoder`) and device playback. There used to be a separate `dr_libs/`
  submodule for the decode/encode side too, but miniaudio.h already embeds
  its own renamed copy of dr_wav/dr_mp3/dr_flac internally (search for
  `dr_wav_h begin` in it) and exposes them through the same API -- so that
  second submodule was pure duplication, removed once `src/audio.cpp` was
  switched over to `ma_decoder`/`ma_encoder`.

This stays as a thin submodule pointer (see `.gitmodules`) since the full
upstream project is needed as-is. Initialize it with:

```bash
git submodule update --init src/third_party/miniaudio
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
