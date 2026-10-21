# ADR-0005: Models are downloaded on first use, with checksum and readme files, verified before they are trusted

- **Status:** accepted
- **Date:** 2026-10-17
- **Source:** CHANGELOG 0.4.0 (Added), commits f0ee8a3, 05bea82, be00157, `include/libspeech/models/model_files.h`, `data/models.csv`

## Context

Models are large and not shipped in the wheel. An interrupted or corrupted download used to leave a broken file that later calls trusted.

## Decision

- A model name resolves to the project's `models` release (`https://github.com/MohammadRaziei/libspeech/releases/download/models/`). Next to each model are `<model>.onnx.sha256` (one line, `sha256sum` format) and `<model>.onnx.readme` (what it is, who made it, license, provenance).
- Order: checksum first (an HTML error page is rejected), then the readme (a missing one fails before the big download), then the model, which httpp verifies and deletes on mismatch. The sidecar files are installed next to the model only after it matched.
- A cached model is hashed on every load and downloaded again on mismatch. `ignore_cache=True` downloads everything again and leaves the old cache untouched if that fails.
- Concurrency: httpp's lock directory `<file>.lock/` with a heartbeat file; a stale lock from a dead process is taken over; a killed download resumes from its `.part` file; a cache hit never takes the lock.
- An explicit URL may lack the sidecar files and is then used unverified, with a warning.
- `data/models.csv` lists every model (backend, sample rate, size, sha256, license, `commercial_use`, source); a test checks it against the release's `.sha256` files.
- Downloaded models keep their own licenses (`NOTICE`).

## Rejected alternatives

Shipping models in the wheel; trusting any existing file at the cache path (the old `download_file` behavior).

## Consequences

- First use needs the network; the offline tests use a local HTTP server.
- Heavy real-model tests (about 1 GB RAM) are not part of `speech_test`; see ADR-0016.
