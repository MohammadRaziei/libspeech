# libspeech: hand-off to the next chat (2026-10-27)

Read first: `docs/agent.md` (rules, set-up, pitfalls), `checklist.md` (state), `docs/adr/README.md` (decisions).
Reports to the owner are in **Persian** (code and commands in English); everything written in the repository is **English**
with no em dashes and no unnecessary `noqa` comments. The default branch is `master`.
Owner's standing preferences: no diff or patch files (send the whole project as a zip without `.git`, build trees and caches
after every step, automatically); build and test through CMake targets and `ctest`; run `ruff check . --exclude 'build-*'`
(no lint errors); numbers must be measured and reported, and what was not run must be said plainly; every new test is
mutation-checked (break the code on purpose, see the test fail, restore, see it pass); do not reopen decisions listed in the ADRs.

## Next task (start here): a new normalization idea, the gain chosen by the model's own confidence

**Status: an idea only. Nothing is implemented, nothing is measured.** The owner asked for an idea "that does not exist at all and
has not been published". This idea is **not proven new**: it rests on a handful of web searches (listed below), not on a systematic
literature review. Do not claim novelty in code, docs or ADRs until the prior-art search in task 1 is done. Do not write code for the
library API before the experiment in task 4 gives a result. A negative result is an acceptable outcome and must be written down.

### What was done before this idea (all of it known techniques, none of it new)

Everything implemented in the earlier part of this session is a known method, combined with the VAD probabilities: speech-weighted RMS
(ADR-0022), `clip_percentile` / soft limiter / weighted CMVN (ADR-0023), LUFS (ADR-0024), streaming speech AGC (ADR-0025). The owner
noticed this and asked for something that is not in the literature.

### The idea: choose the input gain by the confidence of the downstream model itself

Neural speech models (Silero VAD, the denoisers) are sensitive to the input level, and RMS or LUFS normalization picks the level
from a signal statistic that the model knows nothing about. Instead: search over one scalar, the input gain g (for example from
-30 dB to +30 dB), and take the gain at which the model's own output is the most decisive.

- **VAD objective:** the mean binary entropy of the per-window speech probabilities (`SileroVad.speech_probabilities()`) over the
  signal; lower is more decisive. An alternative is the fraction of probabilities that sit in the undecided band, for example 0.2 to 0.8.
- **Denoiser objective:** how bimodal the predicted mask is (speech bins near 1, noise bins near 0). **Open question:** the
  previous chat did not check whether `Denoiser` exposes a mask at all. Look in `include/libspeech/models/` first. If it does not,
  use a proxy on the waveform output (for example how much the enhanced output, scaled back by 1/g, changes between neighbouring gains) or
  restrict the first experiment to the VAD.
- **Search:** one scalar, a golden-section search on log g with about 10 model passes, on a representative excerpt (about 10 s) so it
  is cheap. No labels, no change to the model weights.
- **Why it might be useful:** the operating point is chosen for the model, not for a loudness standard; it needs no clean reference.
- **Why it might not work (test these, do not assume):** entropy can be minimized by a gain that makes the model confidently wrong
  (very high gain amplifies noise into confident "speech"); the optimum may simply coincide with RMS normalization (then the idea adds
  nothing); a flat objective near the optimum gives an unstable gain; the objective of a denoiser mask may not track quality.

### Closest prior work found (so the claim of novelty stays modest)

- Test-time adaptation for ASR (SUTA) minimizes the entropy of the model's output per utterance, but it adapts **model parameters**, not the input gain.
- Test-time adaptation for speech enhancement uses mask bimodality ("mask polarization") as the adaptation signal, again adapting
  **model weights**.
- Patents on AGC use VAD or confidence values to set **time constants** or to gate the gain, not to select the gain by the model's output.
- I did not find a method that selects only the input gain from the model's own output confidence. That is a statement about a few
  searches, not a proof.

### Tasks, in order

1. **Prior-art search first (about one hour).** Queries to try, in different phrasings and on Google Scholar, arXiv, ISCA and IEEE Xplore:
   "input gain selection neural network speech", "entropy minimization input scaling speech", "level calibration VAD neural",
   "gain-invariant voice activity detection", "automatic input level selection for ASR front end", "input-level sensitivity of speech
   enhancement models", "Silero VAD input level". Record every close hit (title, link, what it does, how it differs) in the lab README. If an
   identical method exists, stop and tell the owner in Persian; then the work becomes a plain implementation of a published method with a citation.
2. **Read the model API.** Check what `SileroVad` and `Denoiser` expose (probabilities yes; masks unknown). Ask the owner before adding anything to a public API.
3. **Data.** The sandbox network only reaches GitHub, PyPI and a few package hosts, so speech corpora cannot be downloaded. Ask the
   owner for a small set of clean speech clips (and, if possible, noise clips) so that mixtures with known SNR and a known speech mask
   can be made. The synthetic generator in `tests/python/test_speech_normalize.py` (`_speech_like`) works for smoke tests, but it is a weak
   stand-in for a neural VAD; the owner's file `alex-noisy.mp3` is real but has no reference.
4. **Experiment as a lab, not as an API:** a new folder `labs/gain_by_confidence/` (look at `labs/quantize_denoiser/` for the layout)
   with a README that states the objective, the search and the evaluation **before** the numbers are produced.
   Compare, on the same clips and SNRs (for example -5, 0, 5, 10, 20 dB) and with gain offsets applied to the input
   (for example -30 to +30 dB): no normalization, RMS to -20 dBFS, speech-weighted RMS, LUFS -23, and the confidence-selected gain.
   Metrics: VAD F1 against the known speech mask; for the denoiser SI-SDR improvement against the clean reference. Report means with a
   confidence interval, the selected gain next to the RMS gain, how often the search lands on a boundary of the range, and the cost in
   model passes. Decide the pass/fail thresholds before running and write them in the README.
5. **Decision.** Only if the experiment shows a clear, repeatable gain over RMS and LUFS normalization: propose the API to the owner
   (consultation format in `docs/agent.md` section 1), then implement with tests and mutation checks, an ADR marked experimental, and a CHANGELOG entry.
   Otherwise: write the negative result into the lab README and an ADR with status "rejected", and tell the owner.
6. Always: build and test through CMake (`ctest`, the e2e target), `ruff check . --exclude 'build-*'`, mutation-check new tests, report
   in Persian, zip the whole project without `.git`, build trees and caches after every step.

### Other open ideas (all known methods, no novelty claim)

- Noise-floor aware gain / SNR report: take the noise floor from the windows weighted by `1 - speech weight`. Note: noise-compensated
  active speech level estimation already exists (Dionelis and Brookes, below), so this would be an implementation of known work.
- Level measured only in the 300 to 3400 Hz band.
- LUFS extras: multi-channel, loudness range (LRA), true peak (4x oversampling), a variant that counts only the VAD speech blocks.
- AGC extras: look-ahead limiting, a combination with the streaming denoiser, an AGC that follows the weights per sample.
- Items already listed in `checklist.md` and under "Open decisions" below are unchanged.

### References (only links that were actually retrieved are given; check them before citing)

- N. Dionelis and M. Brookes, "Active speech level estimation in noisy signals with quadrature noise suppression", EUSIPCO 2016.
  https://spiral.imperial.ac.uk/handle/10044/1/41396 (PDF: https://spiral.imperial.ac.uk/bitstream/10044/1/41396/2/Dionelis2016.pdf).
  It reports that ITU-T P.56 is accurate only above about 15 dB SNR. Used to rule out the "noise-compensated speech level" idea as new.
- P. Kabal, "Measurement of the speech active level (ITU-T Recommendation P.56)", McGill report, 1999.
  https://www.mmsp.ece.mcgill.ca/Documents/Reports/1999/KabalR1999.pdf
- T. Raichle, E. Amini, B. Yang, "Test-Time Adaptation for Speech Enhancement via Mask Polarization". https://arxiv.org/pdf/2601.14770
- S. Kammoun, S. Leglaive, X. Alameda-Pineda, T. Gerkmann, "Test-time adaptation for speech enhancement with an autoregressive speech prior".
  https://arxiv.org/pdf/2609.03622
- A. Behera, R. A. Easow, V. Parvathala, K. S. R. Murty, "Test-Time Training for Speech Enhancement". https://arxiv.org/pdf/2508.01847
- Lin et al., SUTA (single-utterance test-time adaptation by entropy minimization for ASR): mentioned in the paper above; **no link was
  retrieved**, search for it.
- US patent 10,867,617 (AGC whose time constant depends on confidence levels):
  https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10867617
- US patent 9,842,608 ("Automatic selective gain control of audio data for speech recognition"):
  https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/9842608
- Standards behind the code of this session, no links were checked: ITU-R BS.1770-4 (loudness, used in ADR-0024), ITU-T P.56 (active
  speech level), EBU Tech 3341 (loudness test signals; **not** used to test the code yet), libebur128 (source of the K-weighting formulas
  that were verified numerically against the standard's 48 kHz coefficients).

### What was verified and what was not (this idea)

Verified: nothing about the idea itself. Verified only that the cited papers and patents exist at the links above (from search results;
they were not read in full). Not verified: novelty, usefulness, whether the denoiser exposes a mask, any number.

## This session (2026-10-26), in order

1. **DPDFNet streaming for all variants.** `dpdfnet_baseline`, `dpdfnet4`, `dpdfnet8` stream like `dpdfnet2` (latency 959 samples
   each). 54 new tests in `tests/python/test_dpdfnet_streaming.py` (`models` marker). Mutation: shifting the stream's delay by one
   hop made 48 of them fail.
2. **Python logging bridge** (`src/bindings/python/libspeech/_log.py`). `logging.disable(level)` now reaches the native level
   (the native level follows `max(effective level, disable + 1)`, `logging.disable` is wrapped once). The stderr handler that
   `set_log_level()` adds stays silent once the application configures logging, so lines are no longer printed twice. 6 tests in
   `tests/python/test_log_bridge.py`, three mutations caught.
3. **Threading tests** `tests/python/test_denoiser_threading.py` (`models` marker, gtcrn and dpdfnet): concurrent `process()` and
   concurrent streams on one `Denoiser` equal the serial results; `process()` and `feed()` release the GIL (a sleeping thread keeps
   ticking). Registered in the `speech_test_python_models_e2e` target. Mutations (GIL release removed, shared state) caught.
4. **Benchmarks**: `log_mel` and `delta` rows in `benchmarks/python/` (ops, memory driver, report, CMake). Measured on one core with
   the CMake pipeline (10 s / 60 s of audio): `log_mel` 2.0 / 11.7 ms (librosa 6.5 / 31.3, audioflux 20.1 / 115.9), `delta` of 13 MFCC
   0.13 / 0.33 ms (librosa 1.1 / 1.6). `delta` was **not optimized** (about 3 % of the MFCC pipeline); an OpenMP threshold needs a
   multi-core machine. Peak memory of the two rows was not run.
5. **Quantization lab** `labs/quantize_denoiser/` (new `labs/` folder, one subfolder per story, see `labs/README.md`):
   `onnx_ptq.py` (static int8 for convolutions, dynamic int8 for the LSTM, `--lstm-only`, per-node `sensitivity`, `evaluate`),
   `distill_qat.py` (teacher-student weight-quantization-aware fine-tuning in PyTorch, exports a length-independent ONNX),
   `denoiser_model.py`, `weight_qat.py`, `mixtures.py`, `audio_metrics.py`, `lab_helpers.py`, the Colab notebook
   `quantize_denoiser.ipynb`, and tests in `labs/quantize_denoiser/tests/` (run by hand, **not** in CMake or CI).
   Measured on `dns48` with real noisy recordings (`alex-noisy.mp3`, `audio-fa-noisy.mp3` from the `resources` release), one thread,
   pip `onnxruntime` 1.24.4, agreement = dB against the fp32 model's own output:

   | variant | agreement | speed (RTF fp32 -> int8) | size |
   |---|---|---|---|
   | full static int8 (convs) + dynamic LSTM | 25.8 dB mean, 24.5 min | 0.171 -> 0.137 (1.25x) | 75.5 -> 19.2 MB |
   | **LSTM only** (convs stay fp32) | 43.4 dB mean, 39.6 min | 0.128 -> 0.083 (1.54x) | 75.5 -> 47.2 MB |

   Through libspeech's bundled ONNX Runtime (one 6 s clip) the full static int8 model was **slower** than fp32 (RTF 0.262 vs 0.171,
   25.1 dB). RTF of the same fp32 model differs between runs on this machine (0.128 to 0.171): compare only numbers from one run.
   Facts about `dns48`: 18.8 M weights, LSTM 50 %, 1x1 convs 17 %, k=8 convs 17 %, ConvTranspose 17 %, four fixed k=112 resampling
   convs, no BatchNorm. ORT profile (fp32, 1 thread, 6 s): Conv about 29 %, LSTM 25 %, ConvTranspose 17 %, Split (GLU) 11 %.
   Tried and dropped: kernel-1 Conv to MatMul and removing the GLU Split (bit-identical output, 1.01x to 1.02x). Block benchmark with
   random weights (375 frames, 768 channels): LSTM 2x768 256 ms, int8 68.7 ms, GRU 200 ms, 768-to-256 bottleneck LSTM 15.8 ms,
   two causal convs 25 ms; so replacing the LSTM buys little beyond LSTM-int8, and further speed needs a smaller student.
   The weights' int8 grid used by `distill_qat.py` was verified to equal onnxruntime's per-channel symmetric weight quantization
   (scale ratio 1.0000000, qmax 127 on all 20 conv weights; biases become int32 on their own scale).
6. **VAD (`SileroVad`)**: `speech_probabilities()` (float32, one value per complete 512-sample window since `reset()`),
   `pop_speech_probabilities()` (new windows only, for streaming), `window_samples`, `threshold()`, `set_threshold(value)`
   (`0 < value <= 1`, else `ValueError` / `std::invalid_argument`; applies from the next window, survives `reset()`).
   The silence level is now `max(threshold - 0.15, 0.01)` as in the reference Silero code: below threshold 0.16 it used to be
   negative and a segment never ended. For thresholds from 0.16 up the behaviour is **identical** (verified: the original source from
   git and the new one give the same segments for 0.3 to 0.9 on `alex-noisy.mp3`; only 0.1 differs: 1 segment of 16.8 s before,
   2 segments of 16.4 s now). Tests in `tests/python/test_vad_streaming.py` (`models` marker), eight mutations caught.
7. **CI incidents, both caused by the lab tests I first put into `tests/python/`**: (a) `import soundfile` at module level failed
   collection in the wheel job on Python 3.14t (the wheel test extras have soundfile/scipy/librosa only for Python < 3.13, no PyTorch);
   (b) a helper script named `run_optional_test.py` matches pytest's `*_test.py` pattern, so the coverage target (pytest on the whole
   folder) imported it. Fix: the lab tests live in `labs/<lab>/tests/` and are run by hand; nothing from `labs/` is in CMake or CI.
8. Small: CHANGELOG stray `\n` fixed, `.gitignore` entries for lab outputs, `labs/**` per-file ruff ignores in `pyproject.toml`.

### Next session: speech-weighted RMS normalization (ADR-0022)

- Owner's request: use the VAD probabilities for normalization, with a hard and a soft threshold. Done: `speech::dsp::speech_weights()`
  (hard: `p >= t`; soft: `(p - t) / (1 - t)` clamped), `weighted_rms()`, `rms_normalize_weighted()`; Python `speech_weights`,
  `rms_normalize_weighted`, `speech_rms_normalize(x, mode, threshold=None, vad=None)` (`_normalize.py`). One gain, peak ceiling wins,
  no speech means unchanged, 1-D only. Tests: `tests/dsp/test_normalize.cpp`, `tests/python/test_normalize.py` (no model),
  `tests/python/test_speech_normalize.py` (real Silero, in `speech_test_python_models_e2e`). All mutation-checked.
- Added later (ADR-0023, all opt-in): `clip_percentile` (winsorized speech level), `limit_peaks` / `limiter_lookahead` (soft look-ahead
  limiter instead of lowering the whole gain), `weighted_cmvn` + `frame_weights` (feature normalization from speech frames). Not
  done: noise-floor aware gain, band-limited level (listed in `checklist.md`).
- Streaming speech AGC added (ADR-0025): `speech_agc` (C++, `speech_dsp`), `SpeechAgc` and `libspeech.speech_agc()` (Python). Feed chunks that
  are a multiple of 512 with the weights from `pop_speech_probabilities()` -> `speech_weights()`; the e2e test shows streaming equals offline.
- LUFS added (ADR-0024): `integrated_loudness`, `loudness_normalize` (mono). The K-weighting formulas must be the bilinear ones from the
  analogue prototype (they reproduce the standard's 48 kHz coefficients); the audio-EQ-cookbook shelf is 0.2 dB off, do not use it.
- Class getters lost `get_` (`SileroVad.speech_timestamps()`, `threshold()`); free functions such as `get_log_level` kept their names.
- Python files are copied to the build tree at **configure** time: edit one, re-run `cmake -S . -B build-py` before testing.
- A zip drops symlinks: after unpacking, recreate `src/third_party/onnxruntime/lib/libonnxruntime.so` and `.so.1` (both pointing to
  `libonnxruntime.so.1.21.0`) or the link and `cpp_models` fail.

## Earlier sessions (kept from the previous hand-off, 2026-10-21)

- DPDFNet streaming (`Denoiser("dpdfnet").stream()`, ADR-0019; latency `n_fft - 1 + 4 hops`: 959 samples at 16 kHz, 2879 at 48 kHz).
- RMS normalizer (ADR-0020): `include/libspeech/dsp/normalize.h`, `Audio::normalize_rms`, Python `rms_normalize`, `rms_dbfs`.
  One gain for the whole signal, peak ceiling wins over the target, silent or non-finite input unchanged.
- Demo re-run (generator in the owner's `libspeech-demo-generator.zip`, not in the repo): VAD and GTCRN streaming equal one-shot;
  RTF on one core: gtcrn 0.05 to 0.07, dpdfnet about 0.17, facebook about 0.2, speechbrain about 3.0 to 3.4.
- Speed and quantization study (ADR-0021): libspeech is at raw ONNX Runtime speed on all four backends; dynamic int8 with a real gain
  in libspeech's bundled ORT 1.21: facebook LSTM-only 1.87x at 47.3 dB, speechbrain MatMul+Gemm 1.72x at 33.5 dB. Nothing shipped.
- Before that (older sessions): Denoiser fixes, log API (own `SPEECH_LOG`), istft, fft class, span views, snake_case renames,
  `window_type`, high-level Python API, stubs, model files with `.sha256` and `.readme`, download lock moved into `httpp`
  (needs httpp newer than PyPI 0.6.1), log-mel and delta features, SpeechBrain level matching, GTCRN and DPDFNet denoisers.

## Open decisions for the owner (ask first, format in `docs/agent.md` section 1)

1. **Local model paths.** ADR-0021 item 7 and the `Denoiser` docstring say `model` may be an absolute path, `file://` or a URL,
   but in the current code `fetch_model` treats anything that is not http(s) as a file name of the models release (404 for a local
   path). Probably a regression from the model-files rework, not bisected. Fix it or fix the docs. The lab works around it by copying
   a model into `~/.libspeech` with `.sha256` and `.readme` (`lab_helpers.register_in_libspeech`).
2. **VAD short segments.** A segment shorter than `min_speech_duration_ms` is neither stored nor discarded; it stays open and merges
   with the next speech (the reference drops it and resets). Existing behaviour: on `alex-noisy.mp3` total speech is 9.6 s at
   threshold 0.5 and 14.1 s at 0.7 with 250 ms, but 8.6 s then 6.8 s with 0 ms. Keep or match the reference?
3. **Ship int8 models** (LSTM-only looks like the best trade-off so far) and an API to select them: needs a listening test, a
   measurement in libspeech's own ORT, release files with checksums (ADR-0005). The full static int8 is slower there.
4. **Faster than LSTM-int8** needs a smaller student (fewer channels or `resample=2`) distilled from the teacher: not implemented.
5. Default `num_threads` (1 now), streaming normalizer, streaming for facebook and speechbrain, GPU provider selection: unchanged.

## What the owner does next (not done, needs him)

Push `labs/` to `master`, open
`https://colab.research.google.com/github/MohammadRaziei/libspeech/blob/master/labs/quantize_denoiser/quantize_denoiser.ipynb`,
set a GPU runtime, run with `STEPS = 200` first, then 3000, and send back the tables of Part 1 and Part 2 and `quantized_models.zip`
(`results.json`, `qat/report.json`, the int8 models). Install of libspeech in the notebook is optional (`RUN_LIBSPEECH`).

## NOT run or not verified

`distill_qat.py` on a real checkpoint and on a GPU (the original PyTorch weights cannot be downloaded in the sandbox; defaults
3000 steps, lr 2e-5, batch 8 are guesses); the notebook on real Colab (only `SMOKE = True` in the sandbox); the per-node
`sensitivity` run on `dns48` (did not finish); the int8 models in libspeech's ORT other than one clip; any listening test;
Windows, macOS, OpenMP, multi-core (`num_threads=0`), microphone or socket streaming; the Doxygen and Sphinx build; a newer ONNX
Runtime than the bundled 1.21.0 inside libspeech; the whole CI (only simulated environments and the coverage target).

## Practical notes

- A fresh sandbox has no tools: follow "Fresh sandbox set-up" in `docs/agent.md`. This sandbox: 1 core, little disk (an install of
  PyTorch from PyPI almost filled it: install its dependencies one by one), `httpp` downloads fail with a connection error but `curl`
  works, so models were cached by hand in `~/.libspeech` (`.onnx`, `.onnx.sha256`, `.onnx.readme` from the `models` release).
- Edited `.py` files of the package are copied to `build-py` at **configure** time: re-run `cmake -S . -B build-py` after editing them.
- Run one model test file through CMake: `PYTEST_ADDOPTS='-k test_vad_streaming' cmake --build build-py --target speech_test_python_models_e2e`.
- `python3 /tmp/script.py` puts `/tmp` on `sys.path`, not the current directory: use `PYTHONPATH=.` from `build-py`.
- Do not edit a source while a build runs (stale object files), see `docs/agent.md` section 6.
- A test module that needs optional packages must skip itself at module level; the wheel CI test environment has only numpy, pytest,
  pytest-xdist (and soundfile, scipy, librosa on Python < 3.13). Lab tests stay out of CI.
- `match_level` (SpeechBrain level matching) still has its own copy of the gain logic because `speech_models` does not link `speech_dsp`.
