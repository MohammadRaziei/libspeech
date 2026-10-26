# Architecture Decision Records

Each file records one architectural decision with its **context, decision, rejected alternatives and consequences**.
They were reconstructed from `git log`, `CHANGELOG.md`, `checklist.md`, `CONTRIBUTING.md` and the code, and the
logging and SpeechBrain records (ADR-0009, 0014, 0015) were written while the work was done. Wherever something was not run
or verified, the ADR says so. "none recorded in the history" means the commit messages mention no alternative; it does
not mean none was considered.

| ADR | Decision | Status |
|---|---|---|
| [0001](0001-three-layers-one-library.md) | Three layers (dsp, io, models) behind one `speech` library | accepted |
| [0002](0002-dsp-flat-nonvirtual-own-public-types.md) | DSP is flat, non-virtual C++ with its own public types | accepted |
| [0003](0003-vendoring-policy.md) | Vendoring: small slices copied in, patches documented; large or security-relevant code is a dependency | accepted |
| [0004](0004-httpp-is-a-runtime-dependency.md) | httpp is an external runtime dependency (downloads, SHA-256, progress), not vendored | accepted |
| [0005](0005-verified-model-distribution.md) | Models are downloaded on first use, with checksum and readme files, verified before they are trusted | accepted |
| [0006](0006-onnx-runtime-cpu.md) | ONNX Runtime is downloaded at configure time and runs on the CPU | accepted |
| [0007](0007-denoiser-interface.md) | Denoiser: one interface, pimpl backends, non-virtual process() and close(), thread-safe | accepted |
| [0008](0008-streaming.md) | Streaming: only models that are frame-based stream; chunked input must equal one-shot | accepted |
| [0009](0009-denoiser-output-level.md) | A denoiser returns audio at the level of its input (SpeechBrain is level-matched) | accepted |
| [0010](0010-python-bindings-nanobind.md) | Python bindings: nanobind, one module per C++ module, one public package `libspeech` | accepted |
| [0011](0011-packaging-wheels.md) | Wheels: one cp312-abi3 wheel, separate free-threaded wheels, no musllinux, no delvewheel on Windows | accepted |
| [0012](0012-runtime-simd-dispatch.md) | SIMD is chosen at run time from the CPU; no -march=native | accepted |
| [0013](0013-naming-snake-case.md) | C++ names are STL-style snake_case; Python classes keep PascalCase | accepted |
| [0014](0014-cpp-logging.md) | C++ log: own SPEECH_LOG on AixLog, level checked first, default warning, three outputs | accepted |
| [0015](0015-python-logging-bridge.md) | The native log goes through Python's `logging`, and the logger controls the native level | accepted |
| [0016](0016-testing-and-ci.md) | Testing: real behavior first, real references, a separate gate for heavy model tests | accepted |
| [0017](0017-versioning-and-license.md) | One version source (`version.h`) and Apache-2.0 | accepted |
| [0018](0018-audio-planar-storage.md) | `Audio` stores samples in one planar shared block; miniaudio is the only decoder | accepted |
| [0019](0019-dpdfnet-streaming.md) | DPDFNet streams; the stream compensates the network's four-hop delay like `process()` does | accepted |
| [0020](0020-rms-normalizer.md) | RMS normalizer: dsp functions, `Audio.normalize_rms`, Python `rms_normalize` / `rms_dbfs` | accepted |
| [0021](0021-denoiser-speed-and-quantization.md) | Denoiser speed: libspeech adds no measurable overhead; quantized models are opt-in and user-made (nothing shipped) | accepted |
| [0022](0022-speech-weighted-rms-normalizer.md) | Speech-weighted RMS normalization: hard and soft thresholds on the VAD probabilities | accepted |

## Not decided yet

Tracked in `checklist.md`; they need the project owner:
streaming for `facebook` and `speechbrain`; selecting an ONNX Runtime provider (GPU); a streaming (running RMS) normalizer,
peak and LUFS normalization; shipping int8 models and an API to select them; the default `num_threads`; OpenMP and multi-core measurements; Windows and macOS runs of the logging change.

## Writing a new ADR

File name `NNNN-short-english-title.md`. Sections: status, date, source, context, decision, rejected alternatives,
consequences (including what is **not** verified). Do not edit an accepted ADR; when a decision changes, write a new
one and mark the old one "superseded by ADR-NNNN".
