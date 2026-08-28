# Vendored AudioFlux (partial)

These files are copied (not submoduled) from
[libAudioFlux/audioflux](https://github.com/libAudioFlux/audioflux)
(MIT license, see `LICENSE.md`), commit `824f76d5f19d0358779e513d708a987e4fb9224e`.

**Why vendored instead of a submodule:** libspeech is moving away from the
full AudioFlux submodule (`src/third_party/audioflux`), which pulls in a lot
of Python-binding-focused code we don't need. Only the specific C files
required by `speech::dsp::*` wrappers are copied here, one DSP operator at a
time, as each is ported and test-covered.

**What's here right now:** resampling, FIR filter design, windowing,
vector/complex ops, wave utils (for `speech::dsp::Resample` and
`speech::dsp::window`), plus the FFT/IFFT/DCT/IDCT engine (for
`speech::dsp::FFT`).

**Modifications:** any change made to this code vs. the original upstream
source is recorded in `/audioflux_issues.md` (project root), so it can be
turned into a PR against AudioFlux later. Do not silently edit these files
without adding an entry there.

**Once this folder covers everything the project needs, the
`src/third_party/audioflux` submodule will be dropped entirely.**
