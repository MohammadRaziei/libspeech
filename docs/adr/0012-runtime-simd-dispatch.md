# ADR-0012: SIMD is chosen at run time from the CPU; no -march=native

- **Status:** accepted
- **Date:** 2026-10-02
- **Source:** CONTRIBUTING.md (SIMD and portability), commit c9e793c, checklist.md (OpenMP)

## Context

One wheel or one default build must be correct on every machine and fast on capable ones.

## Decision

- Kernels that benefit (today the polyphase resampler, about 2.3x over its portable twin on AVX2) have a portable C++ version plus vectorized twins selected at run time: AVX2+FMA on x86 (CPUID and OS YMM state checked), NEON on ARM64, portable everywhere else.
- No `-march=native`, no per-ISA build variants. All architecture and compiler macros live in `src/dsp/simd_internal.h` (`LS_HAVE_X86_AVX2`, `LS_HAVE_ARM_NEON`, ...); kernels never test `__x86_64__` or `_MSC_VER` themselves.
- `libspeech.simd_backend()`, `set_simd_enabled(False)`, `LIBSPEECH_SIMD=off` (`LIBSPEECH_DISABLE_AVX2=1` still works).
- A hand-targeted AVX2 clone of the real FFT was measured (n_fft 64 to 8192) and was never faster, so it was dropped.
- DSP loops link `OpenMP_CXX`. The speedup of the OpenMP path was **not verified**, and no multi-core numbers exist (the development sandbox has one core).

## Rejected alternatives

`-march=native`; per-ISA build variants; the AVX2 FFT clone.

## Consequences

- All backends are cross-checked in `tests/dsp/test_fast_paths.cpp`; they were run on x86-64 (GCC, Clang), 32-bit x86, AArch64 under QEMU and with SIMD off.
