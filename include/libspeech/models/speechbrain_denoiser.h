//
// SpeechBrain (SepFormer) denoiser backend.
//

#ifndef LIBSPEECH_SPEECHBRAIN_DENOISER_H
#define LIBSPEECH_SPEECHBRAIN_DENOISER_H

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "libspeech/export.h"
#include "libspeech/models/denoiser.h"

namespace speech::models {

/**
 * SpeechBrain (SepFormer) denoiser. Normally created through Denoiser::Create("speechbrain", ...), but can be
 * constructed directly. Not copyable; see Denoiser for the process()/close() contract.
 */
class SPEECH_API SpeechBrainDenoiser final : public Denoiser {
   public:
    /**
     * @param url Model file name or URL of the ONNX model (downloaded to the cache dir on first use).
     * @param sample_rate The sample rate (Hz) the model expects its input at. Defaults to 16000.
     * @param num_threads ONNX Runtime intra-op threads: 1 (default), or 0 for one per hardware thread.
     */
    explicit SpeechBrainDenoiser(const std::string& url, int sample_rate = 16000, int num_threads = 1);

    ~SpeechBrainDenoiser() override;

   protected:
    std::vector<float> processImpl(const float* data, std::size_t size) override;
    void closeImpl() noexcept override;

   private:
    struct Impl;  // ONNX Runtime state, defined in the .cpp
    std::unique_ptr<Impl> impl_;
};

}  // namespace speech::models

#endif  // LIBSPEECH_SPEECHBRAIN_DENOISER_H
