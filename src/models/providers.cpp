#include "libspeech/models/providers.h"

#include <algorithm>
#include <exception>

#include "onnxruntime_cxx_api.h"

namespace speech::models {

std::vector<std::string> available_providers() {
    return Ort::GetAvailableProviders();
}

bool is_cuda_available() {
    const auto providers = available_providers();
    if (std::find(providers.begin(), providers.end(), "CUDAExecutionProvider") == providers.end()) {
        return false;  // CPU-only ONNX Runtime build
    }
    // Listed != usable: the provider library, CUDA/cuDNN and a driver must all load, and ONNX
    // Runtime reports that by throwing when the provider is appended.
    try {
        Ort::SessionOptions options;
        OrtCUDAProviderOptions cuda_options{};
        options.AppendExecutionProvider_CUDA(cuda_options);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

}  // namespace speech::models
