#include "libspeech/dsp/simd.h"

#include <atomic>
#include <cstdlib>
#include <cstring>

#include "simd_internal.h"

namespace speech::dsp::simd {

namespace {

std::atomic<bool> g_enabled{true};

bool env_forces_generic() {
    if (const char* v = std::getenv("LIBSPEECH_SIMD")) {
        if (std::strcmp(v, "off") == 0 || std::strcmp(v, "0") == 0 ||
            std::strcmp(v, "generic") == 0 || std::strcmp(v, "none") == 0) {
            return true;
        }
    }
    return std::getenv("LIBSPEECH_DISABLE_AVX2") != nullptr;  // legacy spelling
}

#if LS_HAVE_X86_AVX2
bool cpu_has_avx2_fma() {
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
backend detect() {
    if (env_forces_generic()) return backend::generic;
#if LS_HAVE_X86_AVX2
    if (cpu_has_avx2_fma()) return backend::x86_avx2_fma;
#endif
#if LS_HAVE_ARM_NEON
    return backend::arm64_neon;  // Advanced SIMD is mandatory on AArch64
#endif
    return backend::generic;
}

backend detected() {
    static const backend b = detect();
    return b;
}

}  // namespace

backend active_backend() {
    return g_enabled.load(std::memory_order_relaxed) ? detected() : backend::generic;
}

const char* backend_name() {
    switch (active_backend()) {
        case backend::x86_avx2_fma: return "avx2+fma";
        case backend::arm64_neon: return "neon";
        default: return "generic";
    }
}

void set_enabled(bool enabled) { g_enabled.store(enabled, std::memory_order_relaxed); }
bool is_enabled() { return g_enabled.load(std::memory_order_relaxed); }

}  // namespace speech::dsp::simd
