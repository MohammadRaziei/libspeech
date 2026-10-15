#ifndef LIBSPEECH_MODELS_PROVIDERS_H
#define LIBSPEECH_MODELS_PROVIDERS_H

#include <string>
#include <vector>

#include "libspeech/export.h"

namespace speech::models {

/**
 * Names of the ONNX Runtime execution providers the loaded ONNX Runtime build offers, in ONNX
 * Runtime's priority order, e.g. {"CUDAExecutionProvider", "CPUExecutionProvider"}.
 * "CPUExecutionProvider" is always present.
 */
SPEECH_API std::vector<std::string> available_providers();

/**
 * True if models can use CUDA: the loaded ONNX Runtime was built with the CUDA provider AND that
 * provider could actually be created on this machine (its libraries and the driver load).
 * Never throws; returns false for a CPU-only ONNX Runtime.
 */
SPEECH_API bool is_cuda_available();

}  // namespace speech::models

#endif  // LIBSPEECH_MODELS_PROVIDERS_H
