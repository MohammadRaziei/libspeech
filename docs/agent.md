# libspeech: Agent Guide

> This file is the guide for an agent (or any developer) working on libspeech.
> Language of reports to the user: **Persian**. Everything written inside the repository (code, comments, docs, ADRs,
> commit messages, CHANGELOG) is **English**. Do not use em dashes in repository text.
> Project conventions (naming, vendoring, SIMD, logging, versioning, packaging) are in [`CONTRIBUTING.md`](../CONTRIBUTING.md)
> and are not repeated here. Decisions already made are in [`docs/adr`](adr/README.md). The current state is in
> [`checklist.md`](../checklist.md).

---

## 1. Consult before critical decisions

No critical decision is made without asking the user. Small, clear, non-critical next steps are done and reported afterwards,
with every choice and its reason.

Critical, so ask first:

- a change to the layer structure (dsp / io / models, ADR-0001) or to a public API (C++ or Python);
- choosing or replacing a library, or adding a dependency (ADR-0003, ADR-0004);
- a change of behavior of an existing model or function that users can observe (for example the level of a denoiser output, ADR-0009);
- streaming for a model that is not frame-based (ADR-0008);
- anything that changes the wheels or the CI matrix (ADR-0011).

Format of a consultation:

```
Critical decision: [title]
Problem: [what and why]
Options: 1. [option]: advantage / cost   2. [option]: advantage / cost
My suggestion: [option and why]
Question: which option should we choose?
```

Decisions already made are not reopened without a new reason: see the ADR index. When a decision changes, write a new ADR
and mark the old one superseded; never edit an accepted ADR.

---

## 2. Build and test: through CMake only

Build and run everything through CMake targets and `ctest`. Do not call test binaries, `pytest` or `g++` directly for the
result you report. (Quick ad-hoc measurement scripts that build and test nothing are fine; say so when you use them.)

```bash
cmake -S . -B build-dsp -DBUILD_MODELS=OFF -DBUILD_TESTS=ON   # DSP only: no ONNX Runtime, no network
cmake --build build-dsp --target speech_test_cpp_dsp

cmake -S . -B build-py -DBUILD_PYTHON=ON -DBUILD_TESTS=ON      # everything (downloads ONNX Runtime on first configure)
cmake --build build-py
ctest --test-dir build-py --output-on-failure
cmake --build build-py --target speech_test_python_models_e2e # real models: network, about 1 GB RAM, not part of ctest
cmake -S . -B build-py -DBUILD_DOCS=ON && cmake --build build-py --target speech_docs
```

- After adding a test file or a module, **re-run `cmake -S . -B <build>`**: a build tree can keep a stale file list and silently
  not compile the new test. Check the new test really appears in the test list.
- Build trees are git-ignored (`build-*`, `build/`).
- Long builds on one core take minutes: start them detached and poll, do not block.

### Fresh sandbox set-up (nothing is installed)

```bash
pip install cmake ninja pytest nanobind ruff librosa numpy scipy --break-system-packages
pip install httpp --break-system-packages        # from PyPI (0.7.0). `pip install git+https://github.com/.../httpp.git` FAILS here (Cython error)
cmake -S . -B build-py -DBUILD_PYTHON=ON -DBUILD_TESTS=ON
cd build-py && setsid nohup cmake --build . -j2 > ../build.log 2>&1 < /dev/null &   # then poll the log
```

- Do not `pip install onnxruntime` into the process that imports `libspeech` (libspeech bundles its own, 1.21.0); use a separate
  process for raw ONNX Runtime comparisons.
- Reference timings on one core: full build about 10 minutes; `ctest` about 35 s; `speech_test_python_models_e2e` about 60 s.

---

## 3. Working rules

1. **Test first, with real behavior** (CONTRIBUTING: testing philosophy, ADR-0016). A bug found while adding a feature gets a
   regression test before the fix.
2. **Mutation-check every new test.** Introduce a deliberate bug in the code under test, confirm the test fails, revert the bug,
   confirm the file equals the original. A test that passes with its check deleted is rewritten.
3. **Ruff** on the Python code (`ruff check .` excluding build trees) must be clean. Doxygen must stay warning-free when a public
   header changes.
4. **Write the change into `CHANGELOG.md`** (Unreleased) and update `checklist.md` and the ADRs when a decision is made.
5. **Report only what was run.** Every number carries its measurement conditions (cores, input, number of runs). One core says
   nothing about many cores. Say plainly what was **not** run (Windows, macOS, OpenMP, microphone input) and do not call it done.
6. **Never leave stray files.** Tests that write files must remove them even when they fail. Never commit build output or test
   leftovers (an empty `libspeech_test_log_switch.txt` was committed once).
7. Never hold a lock while calling user code (a log callback, a Python callback): it may need the GIL (ADR-0014).

---

## 4. Delivering work

- After every step, send **the whole project as a zip without `.git`** (and without build trees, caches and the downloaded
  ONNX Runtime). Do not send diff or patch files.
- Send other deliverables (a demo, scripts) as separate zips.
- The summary to the user says what changed, what was verified and how, and what is open.

---

## 5. Repository map

| Path | What |
|---|---|
| `include/libspeech/` | public headers (`detail/` is internal, shared by the libraries) |
| `src/dsp`, `src/io` (`src/audio.cpp`), `src/models`, `src/utils` | the three layers and helpers |
| `src/bindings/python/` | nanobind modules and the `libspeech` package (`_log.py`, `_functional.py`, `__init__.py`) |
| `src/third_party/` | copied slices (AudioFlux, AixLog; patches documented), miniaudio submodule |
| `tests/` | `dsp/`, `models/` (utest.h), `python/` (pytest), `utest/` |
| `benchmarks/` | benchmarks and profiling scripts |
| `data/models.csv` | every downloadable model with its checksum and license |
| `docs/` | Sphinx and Doxygen sources, `adr/`, this file |
| `checklist.md`, `CHANGELOG.md`, `CONTRIBUTING.md` | state, release notes, conventions |

---

## 6. Known pitfalls (each one happened)

- A cached or stale CMake file list hides a new test: re-run configure.
- `ERROR` can be a macro on Windows: keep the regression test in `tests/models/test_log.cpp`.
- MSVC drops unreferenced static-library objects: `/WHOLEARCHIVE` for the three archives (ADR-0001).
- A raw SI-SNR model output (SpeechBrain) is about 15x louder than its input and clips when saved: never return it raw (ADR-0009).
- `logging.disable(...)` and `dictConfig` with the default `disable_existing_loggers` silence the library (ADR-0015).
- Reading a config does not prove a wheel is right: build one and open it (ADR-0011).
- **Editing a source while a build is running** can leave an object file newer than the edited source that was compiled from the
  old text: the next build thinks it is current and the link fails with undefined symbols. Wait for the build to end, or
  `touch` every file edited meanwhile before rebuilding.
- Detached jobs: use `setsid nohup ... < /dev/null &`. A server started in one tool call may be gone in the next; start it in the
  same command as the client. `pgrep -f "<text>"` also matches the shell that contains the text.
- `ruff check .` must exclude the build trees (`--exclude 'build-*'`), or it reports errors from copied files.
- `Denoiser(backend, "file.onnx")`: a bare or relative name means "a file in the models release"; ADR-0021 says an absolute path, a
  `file://` path or a URL works for your own model, but in the current code a local path gives a 404 (see the checklist, "Regression?").
- Dynamic int8 quantization of convolutions does not load in the bundled ONNX Runtime 1.21; `uint8` damages the output (ADR-0021).
- Labs (`labs/<name>/`) keep their tests in `labs/<name>/tests/`, outside `tests/python/`, and are run by hand
  (`pytest labs/<name>/tests`). They are not registered in CMake or run by CI: they need packages (soundfile, scipy, torch, ...)
  that the CI test environments do not all have, and a test file in `tests/python/` that fails to import breaks every CI job.
- Page sizes shown on the GitHub `resources` release are wrong; the files are small (the real sizes come from the HTTP headers).

---

## 7. Definition of Done

- [ ] The new behavior has tests that were mutation-checked, and `ctest` passes through CMake.
- [ ] `ruff check` is clean; docs build without new warnings if a public header changed.
- [ ] `CHANGELOG.md`, `checklist.md` and, for a decision, an ADR are updated.
- [ ] What was not run is written down.
- [ ] The whole project zip (no `.git`, no build trees, no stray files) was sent.
