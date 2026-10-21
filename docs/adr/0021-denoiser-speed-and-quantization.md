# ADR-0021: Denoiser speed: libspeech adds no measurable overhead; quantized models are opt-in and user-made

- **Status:** accepted (findings only; nothing was shipped and no public API changed)
- **Date:** 2026-10-21
- **Source:** ad-hoc profiling and quantization scripts (not part of the repo), `checklist.md` ("Performance and quantization")

## Context

After the demo run the owner found the denoisers slower than expected (RTF in the demo: gtcrn about 0.05, dpdfnet about 0.17,
facebook about 0.2, speechbrain about 3.3) and asked whether libspeech or the models can be made faster.

## Findings

All numbers: **one CPU core**, `num_threads=1`, ONNX Runtime 1.21.0 (the one libspeech bundles) unless a row says otherwise.

1. **libspeech is at the speed of raw ONNX Runtime.** Same model, same input, ONNX Runtime called directly from Python
   (pip `onnxruntime` 1.24.4) against libspeech, 12 s of audio:

   | backend | libspeech | raw ONNX Runtime |
   |---|---|---|
   | gtcrn | 0.65 s | 0.62 s (0.82 ms per frame, 750 frames) |
   | dpdfnet (dpdfnet2) | 2.01 s | about 2.0 s (1.66 ms per frame, 1200 frames) |
   | facebook | 2.3 to 2.4 s | 2.35 s |
   | speechbrain | 39 to 40 s | 39.07 s |

   The overhead of our own code (STFT, tensor set-up) is under about 5 percent, visible only for gtcrn. Nothing was optimized
   in libspeech for this reason.
2. **ONNX Runtime options do not help the frame models.** `ORT_ENABLE_EXTENDED` and `allow_spinning=0` gave the same
   0.82 ms per frame for gtcrn as the default. `onnxsim` graph simplification: gtcrn 415 to 414 nodes, 0.829 to 0.866 ms per frame;
   dpdfnet2 893 to 588 nodes, 1.663 to 1.653 ms per frame; outputs equal (max difference 6.4e-6 and 0).
3. **Where the time goes.** gtcrn: about 60 percent of kernel time is data movement (ScatterND, Slice, Transpose, Reshape,
   Gather, Concat); this comes from the exported graph. speechbrain (SepFormer): 26.3 M parameters, 4043 nodes, MatMul 65.6 percent,
   Add 8.0 percent, Softmax 7.1 percent of kernel time (2 s clip, profiler on, so absolute times are inflated). It is a
   dual-path transformer made for quality, not for real time on a CPU.
4. **Dynamic int8 quantization from the existing `.onnx` files** (`onnxruntime.quantization.quantize_dynamic`, weights int8,
   run through libspeech, 4 s of real speech, SNR is the distance of the quantized output from the fp32 output of the same
   model, not a quality score):

   | model | quantized operators | speed | SNR vs fp32 |
   |---|---|---|---|
   | facebook | LSTM only | 1.87x (0.92 s to 0.49 s) | 47.3 dB |
   | speechbrain | MatMul and Gemm | 1.72x (13.5 s to 7.9 s) | 33.5 dB |
   | speechbrain | MatMul only | 1.61x | 33.5 dB |
   | facebook | static QDQ for the convolutions (3 calibration clips) | 1.12x | 21.9 dB |
   | facebook | MatMul only | no change (the model has no MatMul) | identical |
   | both | `uint8` | facebook output wrong (-92.7 dB), speechbrain output destroyed (0.4 dB) | |
   | both | `int8` with convolutions quantized | **does not load** in ONNX Runtime 1.21 (no int8 `ConvInteger` kernel) | |

   With pip ONNX Runtime 1.24.4 the `int8` speechbrain variant did run (11.7 s to 6.9 s on 4 s, SNR 23.3 dB), so loading depends
   on the ONNX Runtime version bundled with libspeech.
5. **Threads.** `num_threads=0` means `std::thread::hardware_concurrency()` (at least 1) in all four backends. It can differ
   from `nproc` when the CPU is restricted by affinity or cgroups. Its effect on the large models was **not measured**
   (one core available).
6. **GIL.** `Denoiser.process()`, `__call__` and `DenoiserStream.feed()/finish()` release the GIL while the model runs
   (read in `bind_models.cpp`); a multi-threaded Python test was not run.
7. **Model argument.** `Denoiser(backend, model)`: an absolute path or `file://` path or an `http(s)` URL works (tested); a
   bare or relative file name is looked up in libspeech's `models` release. A model loaded from a path has no `.sha256` or
   `.readme`, so it is not verified (404 lines are logged for the two sidecar files).

## Decision

- No change to libspeech code, behavior or public API for speed.
- Quantized models are **not** shipped and there is **no `precision` or quantization argument**. A user who wants them makes
  them with `quantize_dynamic` (the recipe above) and passes the absolute path.
- The recipes with a real gain, if the owner decides to ship them later: facebook LSTM-only, speechbrain MatMul and Gemm.

## Rejected for now

- Quantizing convolutions: no gain for facebook (static QDQ) and not loadable in the bundled ONNX Runtime (dynamic).
- Graph simplification of gtcrn or dpdfnet (`onnxsim`): no measurable speedup.
- Changing the default `num_threads` (1) to 0: not measured, and it changes how many cores a call uses.

## Consequences and what is not verified

- A decision is still needed before shipping int8 models: hosting them in the `models` release with `.sha256` and `.readme`
  (ADR-0005), an API to select them, and a **listening test** (the 33.5 dB figure is one 4 s clip; nobody has listened yet).
- Not run: any multi-core measurement, `num_threads=0`, quantization of gtcrn and dpdfnet, a newer ONNX Runtime than 1.21,
  more than one input clip, Windows and macOS.
