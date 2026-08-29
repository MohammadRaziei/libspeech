# Vendored AudioFlux (partial)

These files are copied (not submoduled) from
[libAudioFlux/audioflux](https://github.com/libAudioFlux/audioflux)
(MIT license, see `LICENSE.md`).

- Resample, FIR filter design, windowing, vector/complex ops, wave utils,
  FFT/IFFT/DCT/IDCT: from commit `824f76d5f19d0358779e513d708a987e4fb9224e`.
- STFT/ISTFT: from a later shallow clone at commit
  `0c3f55b409b07381bfe770711e3642e19f333bee` (the full submodule had since
  been removed from this repo -- see checklist.md's note on the vendoring
  workflow going forward).

**Why vendored instead of a submodule:** libspeech is moving away from the
full AudioFlux submodule (`src/third_party/audioflux`), which pulls in a lot
of Python-binding-focused code we don't need. Only the specific C files
required by `speech::dsp::*` wrappers are copied here, one DSP operator at a
time, as each is ported and test-covered.

**What's here right now:** resampling, FIR filter design, windowing,
vector/complex ops, wave utils (for `speech::dsp::Resample` and
`speech::dsp::window`), the FFT/IFFT/DCT/IDCT engine (for
`speech::dsp::FFT`), and the STFT/ISTFT engine (for `speech::dsp::STFT`,
built on top of the FFT engine and windowing).

**Modifications:** any change made to this code vs. the original upstream
source is recorded in `/audioflux_issues.md` (project root), so it can be
turned into a PR against AudioFlux later. Do not silently edit these files
without adding an entry there.

**Once this folder covers everything the project needs, the
`src/third_party/audioflux` submodule will be dropped entirely.**
