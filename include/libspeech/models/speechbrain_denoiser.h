//
// Created by mohammad on 3/26/25.
//

#ifndef LIBSPEECH_SPEECHBRAIN_DENOISER_H
#define LIBSPEECH_SPEECHBRAIN_DENOISER_H


#include "libspeech/models/onnx_model.h"
#include "libspeech/models/denoiser.h"
#include "libspeech/export.h"
#include <vector>

namespace speech::models {

/**
 * SpeechBrainDenoiser class: Inherits from ONNXModel and provides functionality for denoising audio.
 */
class SPEECH_API SpeechBrainDenoiser : public ONNXModel, public Denoiser {
public:
    /**
     * Constructor for SpeechBrainDenoiser.
     * @param url The URL of the ONNX model file to download.
     * @param sample_rate The sample rate (Hz) the model expects its input at. Defaults to 16000.
     */
    SpeechBrainDenoiser(const std::string& url, const int sample_rate=16000);

    /**
     * Destructor for SpeechBrainDenoiser.
     */
    virtual ~SpeechBrainDenoiser();

    /**
     * Processes an input audio tensor using the denoiser model.
     * @param input_audio A vector of floats representing the input audio (normalized between -1 and 1).
     * @return A vector of floats representing the denoised audio.
     */
    std::vector<float> process(const std::vector<float>& input_audio) override;

};

}  // namespace speech::models

#endif //LIBSPEECH_SPEECHBRAIN_DENOISER_H
