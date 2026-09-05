//
// Common interface for all denoiser backends (Facebook, SpeechBrain, ...).
//

#ifndef LIBSPEECH_DENOISER_H
#define LIBSPEECH_DENOISER_H

#include <memory>
#include <string>
#include <vector>

namespace speech::models {

/**
 * Denoiser: shared interface every denoising backend implements.
 * Lets callers swap backends without caring which one is loaded.
 */
class Denoiser {
   public:
    virtual ~Denoiser() = default;

    /**
     * @param input_audio Mono audio samples normalized to [-1, 1].
     * @return Denoised audio, same length as input.
     */
    virtual std::vector<float> process(const std::vector<float>& input_audio) = 0;

    // Factory: picks a backend by name ("facebook" | "speechbrain").
    static std::unique_ptr<Denoiser> Create(const std::string& backend,
                                             const std::string& url,
                                             int sample_rate = 16000);
};

}  // namespace speech::models

#endif  // LIBSPEECH_DENOISER_H
