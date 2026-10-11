#include "libspeech/models/speechbrain_denoiser.h"

#include "onnx_denoiser_impl.h"

namespace speech::models {

struct speechbrain_denoiser::impl : detail::onnx_denoiser_impl {
    impl(const std::string& url, int sample_rate, int num_threads)
        : onnx_denoiser_impl(url, sample_rate, num_threads, /*with_channel_dim=*/false) {}
};

speechbrain_denoiser::speechbrain_denoiser(const std::string& url, int sample_rate, int num_threads)
    : denoiser(sample_rate), impl_(std::make_unique<impl>(url, sample_rate, num_threads)) {}

speechbrain_denoiser::~speechbrain_denoiser() = default;

std::vector<float> speechbrain_denoiser::process_impl(const float* data, std::size_t size) {
    return impl_->run(data, size);
}

void speechbrain_denoiser::close_impl() noexcept {
    impl_.reset();  // frees the session and the ONNX Runtime environment
}

}  // namespace speech::models
