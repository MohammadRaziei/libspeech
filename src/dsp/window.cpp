#include "libspeech/dsp/window.h"

#include <stdexcept>
#include <cstdlib>

#include "aixlog.hpp"
#include "libspeech/detail/log.h"
#include "dsp/flux_window.h"  // Vendored AudioFlux C header (src/third_party/audioflux)
#include "window_type_internal.h"

namespace speech::dsp::window {

namespace {
constexpr const char* log_tag = "speech::dsp::window";
}

std::vector<float> generate(window_type type, int length) {
    if (length <= 0) {
        SPEECH_LOG(ERROR) << TAG(log_tag) << "generate() called with non-positive length=" << length
                   << std::endl;
        throw std::invalid_argument("Window length must be positive.");
    }

    float* raw = window_calFFTWindow(to_audioflux(type), length);
    if (raw == nullptr) {
        // Not expected to happen (AudioFlux's dispatcher has a rect fallback
        // for unrecognized types), but guard anyway since we're crossing a
        // C boundary with a malloc'd pointer.
        SPEECH_LOG(ERROR) << TAG(log_tag) << "window_calFFTWindow returned null (type="
                   << static_cast<int>(type) << ", length=" << length << ")." << std::endl;
        throw std::runtime_error("Window generation failed.");
    }

    std::vector<float> result(raw, raw + length);
    free(raw);

    SPEECH_LOG(DEBUG) << TAG(log_tag) << "Generated window type=" << static_cast<int>(type) << " length="
               << length << std::endl;
    return result;
}

}  // namespace speech::dsp::window
