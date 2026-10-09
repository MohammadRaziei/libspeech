#include "libspeech/models/denoiser.h"

#include <mutex>
#include <stdexcept>

#include "libspeech/models/facebook_denoiser.h"
#include "libspeech/models/speechbrain_denoiser.h"

namespace speech::models {

Denoiser::Denoiser(int sample_rate) : sample_rate(sample_rate) {}

Denoiser::~Denoiser() = default;

std::vector<float> Denoiser::process(const std::vector<float>& input_audio) {
    return process(input_audio.data(), input_audio.size());
}

std::vector<float> Denoiser::process(const float* data, std::size_t size) {
    if (data == nullptr || size == 0) {
        throw std::invalid_argument("Input audio data is empty.");
    }
    // Shared lock: concurrent process() calls are fine, close() waits for them.
    std::shared_lock<std::shared_mutex> lock(mutex_);
    if (closed_) {
        throw std::runtime_error("Denoiser is closed.");
    }
    return processImpl(data, size);
}

void Denoiser::close() noexcept {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    if (!closed_) {
        closed_ = true;
        closeImpl();
    }
}

bool Denoiser::closed() const noexcept {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return closed_;
}

std::unique_ptr<Denoiser> Denoiser::Create(const std::string& backend,
                                            const std::string& url,
                                            int sample_rate,
                                            int num_threads) {
    if (backend != "facebook" && backend != "speechbrain") {
        throw std::invalid_argument("Unknown denoiser backend: " + backend);
    }
    if (num_threads < 0) {
        throw std::invalid_argument("num_threads must be >= 0 (0 = one per hardware thread)");
    }
    if (backend == "facebook") {
        return std::make_unique<FacebookDenoiser>(url, sample_rate, num_threads);
    }
    return std::make_unique<SpeechBrainDenoiser>(url, sample_rate, num_threads);
}

}  // namespace speech::models
