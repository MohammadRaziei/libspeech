#include "libspeech/dsp/simd.h"

#include <atomic>
#include <cstdlib>
#include <cstring>

#include "simd_internal.h"

namespace speech::dsp::simd {

namespace {

std::atomic<bool> g_enabled{true};

bool envForcesGeneric() {
    if (const char* v = std::getenv("LIBSPEECH_SIMD")) {
        if (std::strcmp(v, "off") == 0 || std::strcmp(v, "0") == 0 ||
            std::strcmp(v, "generic") == 0 || std::strcmp(v, "none") == 0) {
            return true;
        }
    }
    return std::getenv("LIBSPEECH_DISABLE_AVX2") != nullptr;  // legacy spelling
}

#if LS_HAVE_X86_AVX2
bool cpuHasAvx2Fma() {
#if LS_COMPILER_GNU_LIKE
    // libgcc / compiler-rt also verify that the OS saves YMM state (XGETBV).
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else  // MSVC: CPUID + XGETBV by hand
    int r[4];
    __cpuid(r, 0);
    if (r[0] < 7) return false;
    __cpuid(r, 1);
    const bool osxsave = (r[2] & (1 << 27)) != 0;
    const bool avx = (r[2] & (1 << 28)) != 0;
    const bool fma = (r[2] & (1 << 12)) != 0;
    if (!(osxsave && avx && fma)) return false;
    if ((_xgetbv(0) & 0x6) != 0x6) return false;  // XMM and YMM state enabled by the OS
    __cpuidex(r, 7, 0);
    return (r[1] & (1 << 5)) != 0;  // AVX2
#endif
}
#endif

// Best backend this binary + CPU + environment allow; computed once.
Backend detect() {
    if (envForcesGeneric()) return Backend::Generic;
#if LS_HAVE_X86_AVX2
    if (cpuHasAvx2Fma()) return Backend::X86Avx2Fma;
#endif
#if LS_HAVE_ARM_NEON
    return Backend::Arm64Neon;  // Advanced SIMD is mandatory on AArch64
#endif
    return Backend::Generic;
}

Backend detected() {
    static const Backend b = detect();
    return b;
}

}  // namespace

Backend activeBackend() {
    return g_enabled.load(std::memory_order_relaxed) ? detected() : Backend::Generic;
}

const char* backendName() {
    switch (activeBackend()) {
        case Backend::X86Avx2Fma: return "avx2+fma";
        case Backend::Arm64Neon: return "neon";
        default: return "generic";
    }
}

void setEnabled(bool enabled) { g_enabled.store(enabled, std::memory_order_relaxed); }
bool isEnabled() { return g_enabled.load(std::memory_order_relaxed); }

}  // namespace speech::dsp::simd
