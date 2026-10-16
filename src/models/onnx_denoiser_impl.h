// Internal (not installed): the ONNX Runtime half shared by the denoiser backends.
// Kept out of the public headers so they don't pull in onnxruntime and so the public classes
// stay single-inheritance (see Denoiser).
#ifndef LIBSPEECH_ONNX_DENOISER_IMPL_H
#define LIBSPEECH_ONNX_DENOISER_IMPL_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "libspeech/models/onnx_model.h"

namespace speech::models::detail {

class onnx_denoiser_impl : public onnx_model {
   public:
    /**
     * @param with_channel_dim true: input shape [1, 1, L] (Facebook); false: [1, L] (SpeechBrain).
     * @param num_threads intra-op threads; 0 = one per hardware thread.
     * @param max_samples longest input the exported graph handles in one call; longer input is
     *        run in overlapping windows and cross-faded (0 = no limit).
     * @param ignore_cache download the model again even if it is cached.
     */
    onnx_denoiser_impl(const std::string& url, int sample_rate, int num_threads, bool with_channel_dim,
                       std::size_t max_samples = 0, bool ignore_cache = false)
        : onnx_model(url, sample_rate, speech::utils::get_default_model_cache_dir(), ignore_cache), with_channel_dim_(with_channel_dim), max_samples_(max_samples) {
        if (num_threads < 0) {
            throw std::invalid_argument("num_threads must be >= 0");
        }
        if (num_threads == 0) {
            num_threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
        }
        init_engine_threads(1, num_threads);
        init_onnx_model();
    }

    // Safe to call from several threads at once: ONNX Runtime's Run() is thread-safe and this
    // only reads members that are immutable after construction.
    std::vector<float> run(const float* data, std::size_t size) {
        constexpr std::size_t overlap = 16000;  // 1 s at 16 kHz, cross-faded between windows
        if (max_samples_ == 0 || size <= max_samples_) {
            return run_once(data, size);
        }
        const std::size_t hop = max_samples_ - overlap;
        std::vector<float> out(size, 0.0f);
        for (std::size_t start = 0;; start += hop) {
            const std::size_t len = std::min(max_samples_, size - start);
            const bool first = start == 0;
            const bool last = start + len >= size;
            const std::vector<float> y = run_once(data + start, len);
            if (y.size() != len) {
                throw std::runtime_error("denoiser output length differs from its input length");
            }
            for (std::size_t i = 0; i < len; ++i) {
                float w = 1.0f;
                if (!first && i < overlap) {
                    w = (static_cast<float>(i) + 0.5f) / static_cast<float>(overlap);
                }
                if (!last && i >= len - overlap) {
                    w = 1.0f - (static_cast<float>(i - (len - overlap)) + 0.5f) / static_cast<float>(overlap);
                }
                out[start + i] += w * y[i];
            }
            if (last) break;
        }
        return out;
    }

   private:
    std::vector<float> run_once(const float* data, std::size_t size) {
        const int64_t length = static_cast<int64_t>(size);
        const std::vector<int64_t> shape = with_channel_dim_ ? std::vector<int64_t>{1, 1, length}
                                                              : std::vector<int64_t>{1, length};

        // ONNX Runtime wants a non-const pointer but never writes to an input tensor.
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, const_cast<float*>(data), size, shape.data(), shape.size());

        const char* input_names[] = {input_name.c_str()};
        const char* output_names[] = {output_name.c_str()};
        auto output_tensors = session->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1,
                                           output_names, 1);

        const float* output_data = output_tensors[0].GetTensorMutableData<float>();
        const size_t output_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
        return std::vector<float>(output_data, output_data + output_size);
    }

    const bool with_channel_dim_;
    const std::size_t max_samples_;
};

}  // namespace speech::models::detail

#endif  // LIBSPEECH_ONNX_DENOISER_IMPL_H
