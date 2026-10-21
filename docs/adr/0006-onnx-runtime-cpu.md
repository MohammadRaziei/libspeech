# ADR-0006: ONNX Runtime is downloaded at configure time and runs on the CPU

- **Status:** accepted
- **Date:** 2026-09 to 2026-10
- **Source:** `cmake/ONNXRuntime.cmake`, commit 66b46b1, CHANGELOG 0.4.0 (providers)

## Context

All neural backends are ONNX exports. ONNX Runtime must exist on every platform the wheels target.

## Decision

- `cmake/ONNXRuntime.cmake` downloads the official release (default `onnx_version` 1.21.0) at first configure; on macOS a universal2 build uses Microsoft's universal2 archive.
- Models run on the CPU. `libspeech.available_providers()` and `is_cuda_available()` only report what the loaded build offers; selecting a provider for a session is not implemented. The `USE_GPU` option exists in CMake.
- musllinux wheels are skipped: Microsoft publishes only glibc builds of ONNX Runtime, and linking one into a musl container fails (commit 66b46b1).

## Rejected alternatives

Building ONNX Runtime from source; musllinux support.

## Consequences

- The first configure needs the network (not so with `-DBUILD_MODELS=OFF`).
- GPU execution is not available yet and is not verified.
