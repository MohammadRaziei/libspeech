# ADR-0019: DPDFNet streams; the stream compensates the network's four-hop delay like `process()` does

- **Status:** accepted
- **Date:** 2026-10-21
- **Source:** `src/models/dpdfnet_denoiser.cpp` (`dpdfnet_stream`), `tests/python/test_dpdfnet_streaming.py`, CHANGELOG (Unreleased)

## Context

ADR-0008 left DPDFNet streaming open: the model is frame-based like GTCRN, and `detail::stft_streamer` already supported a
`shift` (the network answers four hops late and `process()` flushes the tail with silence to get those samples).

## Decision

- `dpdfnet_denoiser` overrides `supports_streaming()` and `create_stream_impl()`. A `dpdfnet_stream` owns its recurrent
  state (a copy of the model's initial state, with the normalisation statistics from the model metadata) and a
  `stft_streamer` built with `shift = 4 hops`. `step()` (one frame through the model) is shared by `process()` and the
  streams, so the two cannot drift apart. The model is held by a `shared_ptr`, so a running stream survives `close()`
  (same as GTCRN).
- Latency is `n_fft - 1 + 4 hops`: 959 samples (60 ms) for the 16 kHz models (n_fft 320, hop 160) and 2879 samples (60 ms)
  for `dpdfnet2_48khz_hr`. The bound is reached (checked feeding one sample at a time).
- The streamed audio equals `process()` on the whole signal, tail included, for any chunking (`assert_array_equal`, chunks
  from 1 sample to 10^6, input lengths from 1 to 1500 samples, `dpdfnet2` and the 48 kHz model).
- No public API changes: `Denoiser.stream()` / `create_stream()` already existed; `supports_streaming` is now true for
  `dpdfnet`.

## Rejected alternatives

Streaming without the delay compensation (output lags the input by 640 samples, total output length differs from the
input): breaks the rule of ADR-0008 that a stream returns exactly as many samples as it got, aligned with the input.

## Consequences

- Not verified: microphone or socket input, latency in a live setting, speed of streaming (not measured), the other
  three 16 kHz variants (`dpdfnet_baseline`, `dpdfnet4`, `dpdfnet8`; they share the code but only `dpdfnet2` and the
  48 kHz model were run).
- `step()` is shared with `process()`, so an error inside it would not show up in the equality tests; the one-shot
  output was compared with sherpa-onnx in ADR-0008.
