//
// Created by mohammad on 3/19/25.
//

#ifndef LIBSPEECH_FACEBOOK_DENOISER_H
#define LIBSPEECH_FACEBOOK_DENOISER_H


#include "libspeech/models/onnx_model.h"
#include "libspeech/models/denoiser.h"
#include <vector>

namespace speech::models {

/**
 * FacebookDenoiser class: Inherits from ONNXModel and provides functionality for denoising audio.
 */
class FacebookDenoiser : public ONNXModel, public Denoiser {
   public:
    /**
     * Constructor for FacebookDenoiser.
     * @param url The URL of the ONNX model file to download.
     * @param base_dir The base directory where the model will be stored.
     */
    FacebookDenoiser(const std::string& url, const int sample_rate=16000);

    /**
     * Destructor for FacebookDenoiser.
     */
    virtual ~FacebookDenoiser();

    /**
     * Processes an input audio tensor using the denoiser model.
     * @param input_audio A vector of floats representing the input audio (normalized between -1 and 1).
     * @param sample_rate The sample rate of the input audio (e.g., 16000 Hz).
     * @return A vector of floats representing the denoised audio.
     */
    std::vector<float> process(const std::vector<float>& input_audio) override;

};

}  // namespace speech::models

#endif  // LIBSPEECH_FACEBOOK_DENOISER_H
