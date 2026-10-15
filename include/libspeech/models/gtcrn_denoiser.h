//
// GTCRN speech enhancement backend (https://github.com/Xiaobin-Rong/gtcrn).
//

#ifndef LIBSPEECH_GTCRN_DENOISER_H
#define LIBSPEECH_GTCRN_DENOISER_H

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "libspeech/export.h"
#include "libspeech/models/denoiser.h"

namespace speech::models {

/**
 * GTCRN: a very small (~0.5 MB) streaming speech-enhancement model that works on the STFT of 16 kHz
 * audio (512-point sqrt-Hann window, hop 256). Normally created through denoiser::create("gtcrn", ...),
 * but can be constructed directly. Not copyable; see denoiser for the process()/close() contract.
 *
 * The model consumes one spectrogram frame at a time and carries its own state between frames, so
 * the input can be any length: it is processed frame by frame, no windowing of the waveform needed.
 */
class SPEECH_API gtcrn_denoiser final : public denoiser {
   public:
    /**
     * The standard model: a file name, so it is looked up in ~/.libspeech and otherwise downloaded from
     * libspeech's `models` release (see base_model). Provenance, license and checksum are in
     * gtcrn_simple.onnx.readme / .sha256 next to it.
     */
    static constexpr const char* default_model = "gtcrn_simple.onnx";

    /**
     * @param url Model file name or URL of the streaming GTCRN ONNX model.
     * @param sample_rate Must be 16000: the model was trained for 16 kHz audio.
     * @param num_threads ONNX Runtime intra-op threads: 1 (default), or 0 for one per hardware thread.
     * @throws std::invalid_argument if sample_rate is not 16000 or num_threads is negative.
     * @throws std::runtime_error if the file is not a GTCRN model with the expected inputs.
     */
    explicit gtcrn_denoiser(const std::string& url = default_model, int sample_rate = 16000, int num_threads = 1);

    ~gtcrn_denoiser() override;

   protected:
    std::vector<float> process_impl(const float* data, std::size_t size) override;
    void close_impl() noexcept override;

   private:
    struct impl;  // ONNX Runtime state, defined in the .cpp
    std::unique_ptr<impl> impl_;
};

}  // namespace speech::models

#endif  // LIBSPEECH_GTCRN_DENOISER_H
