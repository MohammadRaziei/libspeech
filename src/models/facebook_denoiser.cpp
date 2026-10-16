#include "libspeech/models/facebook_denoiser.h"

#include "onnx_denoiser_impl.h"

namespace speech::models {

// The exported dns64 graph pads its input to a fixed window internally (about 32 s at 16 kHz:
// every activation has the same size for a 0.1 s and a 5 s input, ~7 s per call on one core) and
// silently returns only that many samples for longer input, so longer input is run in windows.
constexpr std::size_t max_window_samples = 512000;

struct facebook_denoiser::impl : detail::onnx_denoiser_impl {
    impl(const std::string& url, int sample_rate, int num_threads, bool ignore_cache)
        : onnx_denoiser_impl(url, sample_rate, num_threads, /*with_channel_dim=*/true, max_window_samples, ignore_cache) {}
};

facebook_denoiser::facebook_denoiser(const std::string& url, int sample_rate, int num_threads, bool ignore_cache)
    : denoiser(sample_rate), impl_(std::make_unique<impl>(url, sample_rate, num_threads, ignore_cache)) {
    set_model_files(impl_->files());
}

facebook_denoiser::~facebook_denoiser() = default;

std::vector<float> facebook_denoiser::process_impl(const float* data, std::size_t size) {
    return impl_->run(data, size);
}

void facebook_denoiser::close_impl() noexcept {
    impl_.reset();  // frees the session and the ONNX Runtime environment
}

}  // namespace speech::models
