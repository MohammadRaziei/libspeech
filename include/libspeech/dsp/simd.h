//
// Runtime SIMD backend selection for speech::dsp.
//
// Kernels that benefit from explicit SIMD (today: the polyphase resampler; the
// real-input FFT is portable code the compiler vectorizes for the baseline ISA --
// a hand-targeted AVX2 clone of it measured slower) have a portable C++
// implementation that runs everywhere, plus vectorized implementations that
// are compiled in only where the toolchain supports them and are picked at RUN
// time from what the CPU actually offers -- so a single binary (or wheel) is
// correct on every machine of its architecture, with no -march flags needed:
//
//   x86 / x86-64   AVX2 + FMA   (GCC, Clang, MSVC; CPUID and OS YMM-state checked)
//   ARM64          NEON         (always present on AArch64: Linux, macOS, Windows)
//   anything else  generic C++
//
// Controls (all optional):
//   - environment LIBSPEECH_SIMD=off   force the generic kernels
//     (legacy spelling LIBSPEECH_DISABLE_AVX2=1 still works)
//   - set_enabled(false) at run time (affects objects/calls created afterwards)
//
#ifndef LIBSPEECH_DSP_SIMD_H
#define LIBSPEECH_DSP_SIMD_H

#include "libspeech/export.h"

namespace speech::dsp::simd {

enum class backend {
    generic,     // portable C++ (the compiler may still auto-vectorize it)
    x86_avx2_fma,  // hand-written AVX2 + FMA kernels
    arm64_neon,   // hand-written AArch64 NEON kernels
};

// The backend kernels will use right now: the best one this binary was compiled
// with AND this CPU supports, unless disabled by environment or set_enabled().
SPEECH_API backend active_backend();

// Short stable name of active_backend(): "generic", "avx2+fma" or "neon".
SPEECH_API const char* backend_name();

// Run-time kill switch (default: enabled). Disabling selects the generic
// kernels for everything constructed or called afterwards: objects that
// already chose a kernel at construction (STFT, MFCC) keep it. Thread-safe.
SPEECH_API void set_enabled(bool enabled);
SPEECH_API bool is_enabled();

}  // namespace speech::dsp::simd

#endif  // LIBSPEECH_DSP_SIMD_H
