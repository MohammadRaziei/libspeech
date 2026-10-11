#include "libspeech/models/facebook_denoiser.h"

#include "onnx_denoiser_impl.h"

namespace speech::models {

struct facebook_denoiser::impl : detail::onnx_denoiser_impl {
    impl(const std::string& url, int sample_rate, int num_threads)
        : onnx_denoiser_impl(url, sample_rate, num_threads, /*with_channel_dim=*/true) {}
};

facebook_denoiser::facebook_denoiser(const std::string& url, int sample_rate, int num_threads)
    : denoiser(sample_rate), impl_(std::make_unique<impl>(url, sample_rate, num_threads)) {}

facebook_denoiser::~facebook_denoiser() = default;

std::vector<float> facebook_denoiser::process_impl(const float* data, std::size_t size) {
    return impl_->run(data, size);
}

void facebook_denoiser::close_impl() noexcept {
    impl_.reset();  // frees the session and the ONNX Runtime environment
}

}  // namespace speech::models
