#include "libspeech/models/facebook_denoiser.h"

#include "onnx_denoiser_impl.h"

namespace speech::models {

struct FacebookDenoiser::Impl : detail::OnnxDenoiserImpl {
    Impl(const std::string& url, int sample_rate, int num_threads)
        : OnnxDenoiserImpl(url, sample_rate, num_threads, /*with_channel_dim=*/true) {}
};

FacebookDenoiser::FacebookDenoiser(const std::string& url, int sample_rate, int num_threads)
    : Denoiser(sample_rate), impl_(std::make_unique<Impl>(url, sample_rate, num_threads)) {}

FacebookDenoiser::~FacebookDenoiser() = default;

std::vector<float> FacebookDenoiser::processImpl(const float* data, std::size_t size) {
    return impl_->run(data, size);
}

void FacebookDenoiser::closeImpl() noexcept {
    impl_.reset();  // frees the session and the ONNX Runtime environment
}

}  // namespace speech::models
