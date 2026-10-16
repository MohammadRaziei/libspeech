#include "libspeech/models/denoiser.h"

#include <fstream>
#include <mutex>
#include <sstream>
#include <stdexcept>

#include "libspeech/models/dpdfnet_denoiser.h"
#include "libspeech/models/facebook_denoiser.h"
#include "libspeech/models/gtcrn_denoiser.h"
#include "libspeech/models/speechbrain_denoiser.h"

namespace speech::models {

denoiser::denoiser(int sample_rate) : sample_rate(sample_rate) {}

denoiser::~denoiser() = default;

std::vector<float> denoiser::process(const std::vector<float>& input_audio) {
    return process(input_audio.data(), input_audio.size());
}

std::vector<float> denoiser::process(const float* data, std::size_t size) {
    if (data == nullptr || size == 0) {
        throw std::invalid_argument("Input audio data is empty.");
    }
    // Shared lock: concurrent process() calls are fine, close() waits for them.
    std::shared_lock<std::shared_mutex> lock(mutex_);
    if (closed_) {
        throw std::runtime_error("Denoiser is closed.");
    }
    return process_impl(data, size);
}

void denoiser::close() noexcept {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    if (!closed_) {
        closed_ = true;
        close_impl();
    }
}

bool denoiser::closed() const noexcept {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return closed_;
}

std::string denoiser::readme() const {
    if (files_.readme.empty()) {
        throw std::runtime_error("No readme is available for model " + files_.model.string());
    }
    std::ifstream in(files_.readme, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Cannot read " + files_.readme.string());
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::unique_ptr<denoiser> denoiser::create(const std::string& backend,
                                            const std::string& url,
                                            int sample_rate,
                                            int num_threads,
                                            bool ignore_cache) {
    if (backend != "facebook" && backend != "speechbrain" && backend != "gtcrn" &&
        backend != "dpdfnet") {
        throw std::invalid_argument("Unknown denoiser backend: " + backend);
    }
    if (num_threads < 0) {
        throw std::invalid_argument("num_threads must be >= 0 (0 = one per hardware thread)");
    }
    if (backend == "facebook") {
        return std::make_unique<facebook_denoiser>(url, sample_rate, num_threads, ignore_cache);
    }
    if (backend == "gtcrn") {
        return std::make_unique<gtcrn_denoiser>(url, sample_rate, num_threads, ignore_cache);
    }
    if (backend == "dpdfnet") {
        return std::make_unique<dpdfnet_denoiser>(url, sample_rate, num_threads, ignore_cache);
    }
    return std::make_unique<speechbrain_denoiser>(url, sample_rate, num_threads, ignore_cache);
}

}  // namespace speech::models
