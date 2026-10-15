#include "utest.h"

#include <algorithm>
#include <string>

#include "libspeech/models/providers.h"

UTEST(providers, cpu_provider_is_always_available) {
    const auto providers = speech::models::available_providers();
    ASSERT_FALSE(providers.empty());
    ASSERT_TRUE(std::find(providers.begin(), providers.end(), "CPUExecutionProvider") != providers.end());
}

UTEST(providers, cuda_available_implies_cuda_provider_listed) {
    const auto providers = speech::models::available_providers();
    const bool listed = std::find(providers.begin(), providers.end(), "CUDAExecutionProvider") != providers.end();
    // is_cuda_available() may only be true when the provider is listed (a CPU-only build is false).
    if (speech::models::is_cuda_available()) {
        ASSERT_TRUE(listed);
    }
    if (!listed) {
        ASSERT_FALSE(speech::models::is_cuda_available());
    }
}
