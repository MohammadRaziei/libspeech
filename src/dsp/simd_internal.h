// Private: compile-time architecture / compiler detection and the macros the
// vectorized kernels are written against. Public query API: libspeech/dsp/simd.h.
//
// Strategy -- plain "if the target supports it, use it; otherwise fall back", decided
// in the sources by compiler/architecture macros, with no build-system switches:
//   * every kernel has a portable implementation that always compiles;
//   * a vectorized twin is compiled only where LS_HAVE_<ISA> says the compiler and
//     target can build it (#if supported ... #else the portable one is all there is);
//   * the twin is selected at run time from what the CPU reports.
// So a plain `pip wheel` / default CMake build gets the fast path on capable CPUs
// and stays correct on older ones -- no -march=native, no per-ISA build variants.
#ifndef LIBSPEECH_SRC_DSP_SIMD_INTERNAL_H
#define LIBSPEECH_SRC_DSP_SIMD_INTERNAL_H

#include "libspeech/dsp/simd.h"

// ---- Architecture ------------------------------------------------------------
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define LS_ARCH_X86 1
#else
#define LS_ARCH_X86 0
#endif

// ARM64 only: the explicit NEON kernels use AArch64-only intrinsics (vfmaq_f32,
// vaddvq_f32). 32-bit ARM takes the portable path.
#if defined(__aarch64__) || defined(_M_ARM64)
#define LS_ARCH_ARM64 1
#else
#define LS_ARCH_ARM64 0
#endif

// ---- Compiler family (clang-cl defines both _MSC_VER and __clang__: GNU-like wins)
#if defined(__GNUC__) || defined(__clang__)
#define LS_COMPILER_GNU_LIKE 1
#else
#define LS_COMPILER_GNU_LIKE 0
#endif

#if defined(_MSC_VER) && !LS_COMPILER_GNU_LIKE
#define LS_COMPILER_MSVC 1
#else
#define LS_COMPILER_MSVC 0
#endif

#if LS_COMPILER_GNU_LIKE
#define LS_ALWAYS_INLINE inline __attribute__((always_inline))
#define LS_RESTRICT __restrict__
#elif LS_COMPILER_MSVC
#define LS_ALWAYS_INLINE __forceinline
#define LS_RESTRICT __restrict
#else
#define LS_ALWAYS_INLINE inline
#define LS_RESTRICT
#endif

// ---- x86: AVX2 + FMA ----------------------------------------------------------
// GNU-like compilers: per-function `target` attribute, so the rest of the TU keeps
// the baseline ISA. MSVC needs no attribute (intrinsics are always available).
#if LS_ARCH_X86 && LS_COMPILER_GNU_LIKE
#define LS_HAVE_X86_AVX2 1
#define LS_TARGET_AVX2 __attribute__((target("avx2,fma")))
#elif LS_ARCH_X86 && LS_COMPILER_MSVC && _MSC_VER >= 1900
#define LS_HAVE_X86_AVX2 1
#define LS_TARGET_AVX2
#else
#define LS_HAVE_X86_AVX2 0
#define LS_TARGET_AVX2
#endif

#if LS_HAVE_X86_AVX2
#include <immintrin.h>
#if LS_COMPILER_MSVC
#include <intrin.h>
#endif
#endif

// ---- ARM64: NEON ----------------------------------------------------------------
#if LS_ARCH_ARM64 && (LS_COMPILER_GNU_LIKE || LS_COMPILER_MSVC)
#define LS_HAVE_ARM_NEON 1
#if LS_COMPILER_MSVC
#include <arm64_neon.h>
#else
#include <arm_neon.h>
#endif
#else
#define LS_HAVE_ARM_NEON 0
#endif

namespace speech::dsp::simd {

// Cheap per-call checks (one atomic load + a cached static).
inline bool useAvx2Fma() { return activeBackend() == Backend::X86Avx2Fma; }
inline bool useNeon() { return activeBackend() == Backend::Arm64Neon; }

}  // namespace speech::dsp::simd

#endif  // LIBSPEECH_SRC_DSP_SIMD_INTERNAL_H
