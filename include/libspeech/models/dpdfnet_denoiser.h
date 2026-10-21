//
// DPDFNet speech enhancement backend (https://github.com/ceva-ip/DPDFNet).
//

#ifndef LIBSPEECH_DPDFNET_DENOISER_H
#define LIBSPEECH_DPDFNET_DENOISER_H

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "libspeech/export.h"
#include "libspeech/models/denoiser.h"

namespace speech::models {

/**
 * DPDFNet: a family of causal, single-channel speech-enhancement models (DeepFilterNet2 with
 * dual-path RNNs). Like GTCRN they work on the STFT, one frame at a time, carrying their own state
 * between frames, so the input can be any length. Normally created through
 * denoiser::create("dpdfnet", ...), but can be constructed directly. Not copyable; see denoiser for
 * the process()/close() contract.
 *
 * Models (hosted in libspeech's `models` release; each is downloaded to the cache dir on first use, and
 * comes with a .readme (provenance, license) and a .sha256). `url` may be one of these names or any
 * file name / URL of a DPDFNet ONNX export:
 *   - "dpdfnet_baseline", "dpdfnet2", "dpdfnet4", "dpdfnet8": 16 kHz, growing in size and quality
 *   - "dpdfnet2_48khz_hr": 48 kHz in, 48 kHz out
 *
 * The network answers four hops (two window lengths) late, so the output is advanced by that much
 * to line up with the input, and the end of the input is flushed through the model so that nothing
 * is lost at the tail.
 */
class SPEECH_API dpdfnet_denoiser final : public denoiser {
   public:
    /** The model used when none is given: a good size/quality middle ground at 16 kHz. */
    static constexpr const char* default_model = "dpdfnet2";

    /**
     * @param url Variant name (see above), model file name or URL.
     * @param sample_rate The sample rate the model works at: 16000 for every variant but
     *        "dpdfnet2_48khz_hr" (48000). Checked against the model once it is loaded.
     * @param num_threads ONNX Runtime intra-op threads: 1 (default), or 0 for one per hardware thread.
     * @throws std::invalid_argument if sample_rate does not match the model or num_threads is negative.
     * @throws std::runtime_error if the file is not a supported DPDFNet export.
     */
    explicit dpdfnet_denoiser(const std::string& url = default_model, int sample_rate = 16000, int num_threads = 1,
                              bool ignore_cache = false);

    ~dpdfnet_denoiser() override;

    /**
     * DPDFNet streams (denoiser::create_stream()): the same frames and the same arithmetic as process(), so the
     * streamed audio equals process() on the whole signal for any chunking. Latency: n_fft - 1 samples of STFT
     * look-ahead plus the network's four-hop delay (e.g. 319 + 640 = 959 samples, 60 ms at 16 kHz for the 16 kHz
     * models).
     */
    bool supports_streaming() const noexcept override { return true; }

   protected:
    std::vector<float> process_impl(const float* data, std::size_t size) override;
    std::unique_ptr<denoiser_stream> create_stream_impl() override;
    void close_impl() noexcept override;

   private:
    struct impl;  // ONNX Runtime state, defined in the .cpp
    friend class dpdfnet_stream;
    // shared with the streams created from this denoiser, so close() cannot pull the model out from under one
    std::shared_ptr<impl> impl_;
};

}  // namespace speech::models

#endif  // LIBSPEECH_DPDFNET_DENOISER_H
