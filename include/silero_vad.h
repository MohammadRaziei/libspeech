//
// Created by mohammad on 3/16/25.
//

#ifndef LIBSPEECH_SILERO_VAD_H
#define LIBSPEECH_SILERO_VAD_H

#include "onnx_model.h"

#include <vector>
#include <cmath>
#include <stdarg.h>

// Forward declaration of timestamp_t class
class timestamp_t {
   public:
    int start;
    int end;

    timestamp_t(int start = -1, int end = -1)
        : start(start), end(end) {}

    timestamp_t& operator=(const timestamp_t& a) {
        start = a.start;
        end = a.end;
        return *this;
    }

    bool operator==(const timestamp_t& a) const {
        return (start == a.start && end == a.end);
    }

    std::string c_str() const {
        return format("{start:%08d, end:%08d}", start, end);
    }

   private:
    std::string format(const char* fmt, ...) const {
        char buf[256];
        va_list args;
        va_start(args, fmt);
        const auto r = std::vsnprintf(buf, sizeof(buf), fmt, args);
        va_end(args);
        if (r < 0)
            return {};
        const size_t len = r;
        if (len < sizeof(buf))
            return std::string(buf, len);
#if __cplusplus >= 201703L
        std::string s(len, '\0');
        va_start(args, fmt);
        std::vsnprintf(s.data(), len + 1, fmt, args);
        va_end(args);
        return s;
#else
        auto vbuf = std::unique_ptr<char[]>(new char[len + 1]);
        va_start(args, fmt);
        std::vsnprintf(vbuf.get(), len + 1, fmt, args);
        va_end(args);
        return std::string(vbuf.get(), len);
#endif
    }
};




class SileroVadModel : public ONNXModel {
   private:
    // Context-related additions
    const int context_samples = 64;  // For 16kHz, 64 samples are added as context.
    std::vector<float> _context;     // Holds the last 64 samples from the previous chunk (initialized to zero).

    // Original window size (e.g., 32ms corresponds to 512 samples)
    int window_size_samples;

    // Effective window size = window_size_samples + context_samples
    int effective_window_size;

    // Additional declaration: samples per millisecond
    int sr_per_ms;

    // ONNX Runtime input/output buffers
    std::vector<Ort::Value> ort_inputs;
    std::vector<const char*> input_node_names = { "input", "state", "sr" };
    std::vector<float> input;
    unsigned int size_state = 2 * 1 * 128;
    std::vector<float> _state;
    std::vector<int64_t> sr;
    int64_t input_node_dims[2] = {};
    const int64_t state_node_dims[3] = { 2, 1, 128 };
    const int64_t sr_node_dims[1] = { 1 };
    std::vector<Ort::Value> ort_outputs;
    std::vector<const char*> output_node_names = { "output", "stateN" };

    // Memory management for ONNX Runtime
    Ort::MemoryInfo memory_info;  // Add this line

    // Model configuration parameters
    int sample_rate;
    float threshold;
    int min_silence_samples;
    int min_silence_samples_at_max_speech;
    int min_speech_samples;
    float max_speech_samples;
    int speech_pad_samples;
    int audio_length_samples;

    // State management
    bool triggered = false;
    unsigned int temp_end = 0;
    unsigned int current_sample = 0;
    int prev_end;
    int next_start = 0;
    std::vector<timestamp_t> speeches;
    timestamp_t current_speech;

    // Resets internal state (_state, _context, etc.)
    void reset_states();

    // Inference: runs inference on one chunk of input data.
    void predict(const std::vector<float>& data_chunk);

   public:
    // Constructor: sets model path, sample rate, window size (ms), and other parameters.
    SileroVadModel(const std::string& model_path = "silero_vad.onnx",
                   int sample_rate = 16000, int window_frame_size = 32,
                   float threshold = 0.5, int min_silence_duration_ms = 100,
                   int speech_pad_ms = 30, int min_speech_duration_ms = 250,
                   float max_speech_duration_s = std::numeric_limits<float>::infinity());

    // Process the entire audio input.
    void process(const std::vector<float>& input_wav);

    // Returns the detected speech timestamps.
    const std::vector<timestamp_t> get_speech_timestamps() const;

    // Public method to reset the internal state.
    void reset();
};


#endif  // LIBSPEECH_SILERO_VAD_H
