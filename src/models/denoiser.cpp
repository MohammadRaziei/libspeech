#include "libspeech/models/denoiser.h"
#include "libspeech/models/facebook_denoiser.h"
#include "libspeech/models/speechbrain_denoiser.h"

#include <stdexcept>

namespace speech::models {

std::unique_ptr<Denoiser> Denoiser::Create(const std::string& backend,
                                            const std::string& url,
                                            int sample_rate) {
    if (backend == "facebook") {
        return std::make_unique<FacebookDenoiser>(url, sample_rate);
    }
    if (backend == "speechbrain") {
        return std::make_unique<SpeechBrainDenoiser>(url, sample_rate);
    }
    throw std::invalid_argument("Unknown denoiser backend: " + backend);
}


}  // namespace speech::models