//
// Facebook (demucs DNS64) denoiser backend.
//

#ifndef LIBSPEECH_FACEBOOK_DENOISER_H
#define LIBSPEECH_FACEBOOK_DENOISER_H

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "libspeech/export.h"
#include "libspeech/models/denoiser.h"

namespace speech::models {

/**
 * Facebook (demucs DNS64) denoiser. Normally created through Denoiser::Create("facebook", ...), but can be
 * constructed directly. Not copyable; see Denoiser for the process()/close() contract.
 */
class SPEECH_API facebook_denoiser final : public denoiser {
   public:
    /**
     * @param url Model file name or URL of the ONNX model (downloaded to the cache dir on first use).
     * @param sample_rate The sample rate (Hz) the model expects its input at. Defaults to 16000.
     * @param num_threads ONNX Runtime intra-op threads: 1 (default), or 0 for one per hardware thread.
     */
    explicit facebook_denoiser(const std::string& url, int sample_rate = 16000, int num_threads = 1,
                       bool ignore_cache = false);

    ~facebook_denoiser() override;

   protected:
    std::vector<float> process_impl(const float* data, std::size_t size) override;
    void close_impl() noexcept override;

   private:
    struct impl;  // ONNX Runtime state, defined in the .cpp
    std::unique_ptr<impl> impl_;
};

}  // namespace speech::models

#endif  // LIBSPEECH_FACEBOOK_DENOISER_H
