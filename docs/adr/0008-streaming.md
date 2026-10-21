# ADR-0008: Streaming: only models that are frame-based stream; chunked input must equal one-shot

- **Status:** accepted
- **Date:** 2026-10-19
- **Source:** commits 91b3757, 9a59b71, CHANGELOG (Unreleased), checklist.md (Open items: streaming)

## Context

Real-time use needs chunked input. Only some models are frame-based.

## Decision

- `gtcrn` streams: a streamer holds the model state and an STFT streamer (`stft_streamer`) and accepts chunks of any size. `dpdfnet` is frame-based in the same way and `stft_streamer` supports its hop, but its streamer is **not wired yet** (open item in `checklist.md`).
- `SileroVad` streams through `feed`, `pop_speech_timestamps` and `finish`, with variable-sized chunks.
- Streaming output must equal the one-shot `process()` output, and the output always has the input's length. On real speech clips (2026-10-20), GTCRN streamed in 1000-sample chunks differed from one-shot by 0.0, and streaming VAD (1600-sample chunks) gave the same segments as one-shot.
- Separately, the one-shot outputs of GTCRN and DPDFNet were checked against sherpa-onnx 1.13.8 on one 9.8 s clip: max |difference| 1.5e-7 (GTCRN and the four 16 kHz DPDFNet models), 1.4e-5 for the 48 kHz one. DPDFNet's 4-hop latency is compensated in `process()` by flushing with silence, so its last 40 ms are real output.
- `facebook` and `speechbrain` are not streamed: chunked inference with overlap would change their output relative to `process()`, which needs an explicit decision.

## Rejected alternatives

Streaming `facebook` and `speechbrain` by overlapped chunks (open; not decided). SpeechBrain runs at about 3x real time on one CPU core, so streaming it would be of little use there.

## Consequences

- Not verified: microphone or socket input, and latency in a live setting. The tests feed arrays.
- DPDFNet streaming is open work.
