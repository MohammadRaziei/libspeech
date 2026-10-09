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

class OnnxDenoiserImpl : public ONNXModel {
   public:
    /**
     * @param with_channel_dim true: input shape [1, 1, L] (Facebook); false: [1, L] (SpeechBrain).
     * @param num_threads intra-op threads; 0 = one per hardware thread.
     */
    OnnxDenoiserImpl(const std::string& url, int sample_rate, int num_threads, bool with_channel_dim)
        : ONNXModel(url, sample_rate), with_channel_dim_(with_channel_dim) {
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

   private:
    const bool with_channel_dim_;
};

}  // namespace speech::models::detail

#endif  // LIBSPEECH_ONNX_DENOISER_IMPL_H
