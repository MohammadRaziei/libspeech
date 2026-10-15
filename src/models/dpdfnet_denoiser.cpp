#include "libspeech/models/dpdfnet_denoiser.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "libspeech/models/onnx_model.h"
#include "spectral.h"

namespace speech::models {

namespace {

// The published DPDFNet exports are hosted in libspeech's models release (base_model resolves a bare file
// name there): "dpdfnet2" and "dpdfnet2.onnx" both mean the file dpdfnet2.onnx. Anything else is passed
// through as a file name or URL.
std::string resolve_model(const std::string& name) {
    static const char* const known[] = {"dpdfnet_baseline", "dpdfnet2", "dpdfnet4", "dpdfnet8", "dpdfnet2_48khz_hr"};
    std::string stem = name;
    const std::string suffix = ".onnx";
    if (stem.size() > suffix.size() && stem.compare(stem.size() - suffix.size(), suffix.size(), suffix) == 0) {
        stem.resize(stem.size() - suffix.size());
    }
    for (const char* k : known) {
        if (stem == k) return stem + suffix;
    }
    return name;
}

std::vector<float> parse_floats(const std::string& text, const char* key) {
    std::vector<float> values;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        try {
            values.push_back(std::stof(item));
        } catch (const std::exception&) {
            throw std::runtime_error(std::string("DPDFNet metadata '") + key + "' is not a list of numbers");
        }
    }
    return values;
}

}  // namespace

struct dpdfnet_denoiser::impl : onnx_model {
    impl(const std::string& url, int sample_rate, int num_threads) : onnx_model(resolve_model(url), sample_rate) {
        if (num_threads < 0) throw std::invalid_argument("num_threads must be >= 0");
        if (num_threads == 0) num_threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
        init_engine_threads(1, num_threads);
        init_onnx_model();

        if (session->GetInputCount() != 2 || session->GetOutputCount() != 2) {
            throw std::runtime_error("not a DPDFNet model: expected 2 inputs (spec, state) and 2 outputs");
        }
        for (std::size_t i = 0; i < 2; ++i) {
            in_names_.emplace_back(session->GetInputNameAllocated(i, allocator).get());
            out_names_.emplace_back(session->GetOutputNameAllocated(i, allocator).get());
        }
        for (const auto& s : in_names_) in_ptrs_.push_back(s.c_str());
        for (const auto& s : out_names_) out_ptrs_.push_back(s.c_str());

        if (text("model_type") != "dpdfnet") throw std::runtime_error("not a DPDFNet model (model_type metadata)");
        const int model_rate = integer("sample_rate");
        if (model_rate != sample_rate) {
            throw std::invalid_argument("this DPDFNet model works on " + std::to_string(model_rate) +
                                        " Hz audio (got sample_rate=" + std::to_string(sample_rate) +
                                        "); pass sample_rate=" + std::to_string(model_rate) +
                                        " (and resample your audio to it)");
        }
        // Only the configuration the published models use: reflect-padded center frames, a Vorbis window
        // as long as the FFT, no scaling. Anything else would be processed wrongly, so refuse it.
        n_fft_ = static_cast<std::size_t>(integer("n_fft"));
        hop_ = static_cast<std::size_t>(integer("hop_length"));
        const auto freq_bins = static_cast<std::size_t>(integer("freq_bins"));
        if (text("window_type") != "vorbis" || text("pad_mode") != "reflect" || integer("center") != 1 ||
            integer("normalized") != 0 || static_cast<std::size_t>(integer("window_length")) != n_fft_ ||
            n_fft_ < 2 || hop_ == 0 || freq_bins != n_fft_ / 2 + 1) {
            throw std::runtime_error("unsupported DPDFNet STFT configuration in the model metadata");
        }

        spec_shape_ = session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        const std::vector<int64_t> state_shape = session->GetInputTypeInfo(1).GetTensorTypeAndShapeInfo().GetShape();
        if (spec_shape_ != std::vector<int64_t>{1, 1, static_cast<int64_t>(freq_bins), 2} || state_shape.size() != 1 ||
            state_shape[0] <= 0) {
            throw std::runtime_error("unexpected DPDFNet input shapes (want spec [1,1,F,2] and state [S])");
        }
        state_shape_ = state_shape;
        const auto state_size = static_cast<std::size_t>(state_shape[0]);
        if (static_cast<std::size_t>(integer("state_size")) != state_size) {
            throw std::runtime_error("DPDFNet state_size metadata does not match the graph");
        }

        // The recurrent state starts as zeros, except that its first slots hold the initial running
        // normalisation statistics the export ships in its metadata: erb_norm first, then spec_norm.
        const std::vector<float> erb = parse_floats(text("erb_norm_init"), "erb_norm_init");
        const std::vector<float> spec = parse_floats(text("spec_norm_init"), "spec_norm_init");
        const auto erb_size = static_cast<std::size_t>(integer("erb_norm_state_size"));
        const auto spec_size = static_cast<std::size_t>(integer("spec_norm_state_size"));
        if (erb.size() != erb_size || spec.size() != spec_size || erb_size + spec_size > state_size) {
            throw std::runtime_error("DPDFNet normalisation-state metadata is inconsistent");
        }
        initial_state_.assign(state_size, 0.0f);
        std::copy(erb.begin(), erb.end(), initial_state_.begin());
        std::copy(spec.begin(), spec.end(), initial_state_.begin() + static_cast<std::ptrdiff_t>(erb_size));

        fft_ = std::make_unique<detail::real_fft>(n_fft_);
        window_ = detail::vorbis_window(n_fft_);
        latency_ = 2 * n_fft_;  // four hops: the network answers this many samples late
    }

    // Safe to call from several threads at once: the state is local to the call, and the members
    // are read-only after construction (ONNX Runtime's Run() is thread-safe).
    std::vector<float> run(const float* x, std::size_t n) const {
        std::vector<float> state = initial_state_;
        const std::size_t bins = n_fft_ / 2 + 1;
        std::vector<float> spec(2 * bins);

        return detail::stft_apply_istft(
            x, n, *fft_, window_, hop_,
            [&](float* re, float* im) {
                for (std::size_t k = 0; k < bins; ++k) {
                    spec[2 * k] = re[k];
                    spec[2 * k + 1] = im[k];
                }
                std::vector<Ort::Value> inputs;
                inputs.push_back(Ort::Value::CreateTensor<float>(memory_info, spec.data(), spec.size(),
                                                                 spec_shape_.data(), spec_shape_.size()));
                inputs.push_back(Ort::Value::CreateTensor<float>(memory_info, state.data(), state.size(),
                                                                 state_shape_.data(), state_shape_.size()));
                std::vector<Ort::Value> outputs = session->Run(Ort::RunOptions{nullptr}, in_ptrs_.data(),
                                                               inputs.data(), inputs.size(), out_ptrs_.data(),
                                                               out_ptrs_.size());
                const float* enhanced = outputs[0].GetTensorData<float>();
                for (std::size_t k = 0; k < bins; ++k) {
                    re[k] = enhanced[2 * k];
                    im[k] = enhanced[2 * k + 1];
                }
                const float* next = outputs[1].GetTensorData<float>();
                std::copy(next, next + state.size(), state.begin());
            },
            latency_);
    }

   private:
    std::string text(const char* key) const {
        Ort::ModelMetadata metadata = session->GetModelMetadata();
        Ort::AllocatedStringPtr value = metadata.LookupCustomMetadataMapAllocated(key, allocator);
        if (!value) throw std::runtime_error(std::string("DPDFNet model has no '") + key + "' metadata");
        return value.get();
    }
    int integer(const char* key) const {
        try {
            return std::stoi(text(key));
        } catch (const std::logic_error&) {
            throw std::runtime_error(std::string("DPDFNet metadata '") + key + "' is not an integer");
        }
    }

    std::size_t n_fft_ = 0, hop_ = 0, latency_ = 0;
    std::unique_ptr<detail::real_fft> fft_;
    std::vector<float> window_;
    std::vector<std::string> in_names_, out_names_;
    std::vector<const char*> in_ptrs_, out_ptrs_;
    std::vector<int64_t> spec_shape_, state_shape_;
    std::vector<float> initial_state_;
};

dpdfnet_denoiser::dpdfnet_denoiser(const std::string& url, int sample_rate, int num_threads)
    : denoiser(sample_rate), impl_(std::make_unique<impl>(url, sample_rate, num_threads)) {}

dpdfnet_denoiser::~dpdfnet_denoiser() = default;

std::vector<float> dpdfnet_denoiser::process_impl(const float* data, std::size_t size) {
    return impl_->run(data, size);
}

void dpdfnet_denoiser::close_impl() noexcept {
    impl_.reset();  // frees the session and the ONNX Runtime environment
}

}  // namespace speech::models
