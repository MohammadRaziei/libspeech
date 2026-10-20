#include "libspeech/models/speechbrain_denoiser.h"

#include "level_match.h"
#include "onnx_denoiser_impl.h"

namespace speech::models {

struct speechbrain_denoiser::impl : detail::onnx_denoiser_impl {
    impl(const std::string& url, int sample_rate, int num_threads, bool ignore_cache)
        : onnx_denoiser_impl(url, sample_rate, num_threads, /*with_channel_dim=*/false, /*max_samples=*/0, ignore_cache) {}
};

speechbrain_denoiser::speechbrain_denoiser(const std::string& url, int sample_rate, int num_threads, bool ignore_cache)
    : denoiser(sample_rate), impl_(std::make_unique<impl>(url, sample_rate, num_threads, ignore_cache)) {
    set_model_files(impl_->files());
}

speechbrain_denoiser::~speechbrain_denoiser() = default;

std::vector<float> speechbrain_denoiser::process_impl(const float* data, std::size_t size) {
    // The model's output level is arbitrary (SI-SNR training): bring it back to the input's, so that it does not
    // come out ~15x louder and clip when saved. See detail::match_level.
    std::vector<float> out = impl_->run(data, size);
    detail::match_level(data, size, out);
    return out;
}

void speechbrain_denoiser::close_impl() noexcept {
    impl_.reset();  // frees the session and the ONNX Runtime environment
}

}  // namespace speech::models
