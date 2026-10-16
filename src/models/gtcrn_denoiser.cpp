#include "libspeech/models/gtcrn_denoiser.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "libspeech/models/onnx_model.h"
#include "spectral.h"

namespace speech::models {

namespace {
constexpr std::size_t n_fft = 512;
constexpr std::size_t hop = 256;
constexpr std::size_t bins = n_fft / 2 + 1;  // 257

// "gtcrn_simple" (the extension is optional) is the published export, hosted in libspeech's models release;
// anything else is passed through as a file name or URL.
std::string resolve_model(const std::string& name) {
    return name == "gtcrn_simple" ? std::string(gtcrn_denoiser::default_model) : name;
}

std::size_t element_count(const std::vector<int64_t>& shape) {
    std::size_t count = 1;
    for (const int64_t d : shape) {
        if (d <= 0) throw std::runtime_error("GTCRN model has a dynamic input shape; expected the streaming export");
        count *= static_cast<std::size_t>(d);
    }
    return count;
}
}  // namespace

struct gtcrn_denoiser::impl : onnx_model {
    impl(const std::string& url, int sample_rate, int num_threads, bool ignore_cache)
        : onnx_model(resolve_model(url), sample_rate, speech::utils::get_default_model_cache_dir(), ignore_cache), fft_(n_fft), window_(detail::sqrt_hann_window(n_fft)) {
        if (num_threads < 0) throw std::invalid_argument("num_threads must be >= 0");
        if (num_threads == 0) num_threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
        init_engine_threads(1, num_threads);
        init_onnx_model();

        // Streaming export: mix [1, 257, 1, 2] plus three recurrent caches, mirrored by the outputs.
        const std::size_t count = session->GetInputCount();
        if (count != 4 || session->GetOutputCount() != 4) {
            throw std::runtime_error("not a streaming GTCRN model: expected 4 inputs (mix + 3 caches) and 4 outputs");
        }
        for (std::size_t i = 0; i < count; ++i) {
            in_names_.emplace_back(session->GetInputNameAllocated(i, allocator).get());
            out_names_.emplace_back(session->GetOutputNameAllocated(i, allocator).get());
            shapes_.push_back(session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
        }
        if (shapes_[0] != std::vector<int64_t>{1, static_cast<int64_t>(bins), 1, 2}) {
            throw std::runtime_error("not a streaming GTCRN model: the first input must have shape [1, 257, 1, 2]");
        }
        for (std::size_t i = 0; i < count; ++i) sizes_.push_back(element_count(shapes_[i]));
        for (const auto& s : in_names_) in_ptrs_.push_back(s.c_str());
        for (const auto& s : out_names_) out_ptrs_.push_back(s.c_str());
    }

    // Safe to call from several threads at once: the caches are local to the call, and the members
    // are read-only after construction (ONNX Runtime's Run() is thread-safe).
    std::vector<float> run(const float* x, std::size_t n) const {
        std::vector<std::vector<float>> state(3);
        for (std::size_t i = 0; i < 3; ++i) state[i].assign(sizes_[i + 1], 0.0f);
        std::vector<float> mix(2 * bins);

        return detail::stft_apply_istft(x, n, fft_, window_, hop, [&](float* re, float* im) {
            for (std::size_t k = 0; k < bins; ++k) {
                mix[2 * k] = re[k];
                mix[2 * k + 1] = im[k];
            }
            std::vector<Ort::Value> inputs;
            inputs.push_back(Ort::Value::CreateTensor<float>(memory_info, mix.data(), mix.size(), shapes_[0].data(),
                                                             shapes_[0].size()));
            for (std::size_t i = 0; i < 3; ++i) {
                inputs.push_back(Ort::Value::CreateTensor<float>(memory_info, state[i].data(), state[i].size(),
                                                                 shapes_[i + 1].data(), shapes_[i + 1].size()));
            }
            std::vector<Ort::Value> outputs = session->Run(Ort::RunOptions{nullptr}, in_ptrs_.data(), inputs.data(),
                                                           inputs.size(), out_ptrs_.data(), out_ptrs_.size());
            const float* enhanced = outputs[0].GetTensorData<float>();
            for (std::size_t k = 0; k < bins; ++k) {
                re[k] = enhanced[2 * k];
                im[k] = enhanced[2 * k + 1];
            }
            for (std::size_t i = 0; i < 3; ++i) {
                const float* next = outputs[i + 1].GetTensorData<float>();
                std::copy(next, next + state[i].size(), state[i].begin());
            }
        });
    }

   private:
    detail::real_fft fft_;
    std::vector<float> window_;
    std::vector<std::string> in_names_, out_names_;
    std::vector<const char*> in_ptrs_, out_ptrs_;
    std::vector<std::vector<int64_t>> shapes_;
    std::vector<std::size_t> sizes_;
};

gtcrn_denoiser::gtcrn_denoiser(const std::string& url, int sample_rate, int num_threads, bool ignore_cache)
    : denoiser(sample_rate), impl_(nullptr) {
    if (sample_rate != 16000) {
        throw std::invalid_argument("GTCRN works on 16 kHz audio (got sample_rate=" + std::to_string(sample_rate) +
                                    "); resample first");
    }
    impl_ = std::make_unique<impl>(url, sample_rate, num_threads, ignore_cache);
    set_model_files(impl_->files());
}

gtcrn_denoiser::~gtcrn_denoiser() = default;

std::vector<float> gtcrn_denoiser::process_impl(const float* data, std::size_t size) {
    return impl_->run(data, size);
}

void gtcrn_denoiser::close_impl() noexcept {
    impl_.reset();  // frees the session and the ONNX Runtime environment
}

}  // namespace speech::models
