#include "libspeech/models/speechbrain_denoiser.h"

#include "onnx_denoiser_impl.h"

namespace speech::models {

struct SpeechBrainDenoiser::Impl : detail::OnnxDenoiserImpl {
    Impl(const std::string& url, int sample_rate, int num_threads)
        : OnnxDenoiserImpl(url, sample_rate, num_threads, /*with_channel_dim=*/false) {}
};

SpeechBrainDenoiser::SpeechBrainDenoiser(const std::string& url, int sample_rate, int num_threads)
    : Denoiser(sample_rate), impl_(std::make_unique<Impl>(url, sample_rate, num_threads)) {}

SpeechBrainDenoiser::~SpeechBrainDenoiser() = default;

std::vector<float> SpeechBrainDenoiser::processImpl(const float* data, std::size_t size) {
    return impl_->run(data, size);
}

void SpeechBrainDenoiser::closeImpl() noexcept {
    impl_.reset();  // frees the session and the ONNX Runtime environment
}

}  // namespace speech::models
