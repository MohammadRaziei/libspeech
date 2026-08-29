#include "libspeech/dsp/window.h"

#include <stdexcept>
#include <cstdlib>

#include "aixlog.hpp"
#include "dsp/flux_window.h"  // Vendored AudioFlux C header (src/vendor/audioflux)

namespace speech::dsp::window {

namespace {
constexpr const char* kTag = "speech::dsp::window";
}

std::vector<float> generate(WindowType type, int length) {
    if (length <= 0) {
        LOG(ERROR) << TAG(kTag) << "generate() called with non-positive length=" << length
                   << std::endl;
        throw std::invalid_argument("Window length must be positive.");
    }

    float* raw = window_calFFTWindow(type, length);
    if (raw == nullptr) {
        // Not expected to happen (AudioFlux's dispatcher has a rect fallback
        // for unrecognized types), but guard anyway since we're crossing a
        // C boundary with a malloc'd pointer.
        LOG(ERROR) << TAG(kTag) << "window_calFFTWindow returned null (type="
                   << static_cast<int>(type) << ", length=" << length << ")." << std::endl;
        throw std::runtime_error("Window generation failed.");
    }

    std::vector<float> result(raw, raw + length);
    free(raw);

    LOG(DEBUG) << TAG(kTag) << "Generated window type=" << static_cast<int>(type) << " length="
               << length << std::endl;
    return result;
}

}  // namespace speech::dsp::window
