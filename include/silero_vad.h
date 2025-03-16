//
// Created by mohammad on 3/16/25.
//

#ifndef LIBSPEECH_SILERO_VAD_H
#define LIBSPEECH_SILERO_VAD_H

#include "onnx_model.h"
#include <vector>
#include <cmath>
#include <algorithm>

class timestamp_t {
   public:
    int start;
    int end;
    timestamp_t(int start = -1, int end = -1) : start(start), end(end) {}
};

class SileroVadModel : public ONNXModel {
   private:
    int sample_rate;
    float threshold;
    int min_silence_samples;
    int min_silence_samples_at_max_speech;
    int min_speech_samples;
    float max_speech_samples;
    int speech_pad_samples;

    std::vector<float> _context;
    std::vector<float> _state;
    bool triggered = false;
    unsigned int temp_end = 0;
    unsigned int current_sample = 0;
    int prev_end = 0;
    int next_start = 0;
    std::vector<timestamp_t> speeches;
    timestamp_t current_speech;

   public:
    SileroVadModel(const std::string& url,
                   int sample_rate = 16000,
                   float threshold = 0.5,
                   int min_silence_duration_ms = 100,
                   int speech_pad_ms = 30,
                   int min_speech_duration_ms = 250,
                   float max_speech_duration_s = std::numeric_limits<float>::infinity(),
                   const std::filesystem::path& base_dir = std::filesystem::path(getenv("HOME")) / ".libspeech");

    void process(const std::vector<float>& input_wav);
    const std::vector<timestamp_t> get_speech_timestamps() const;

   private:
    void predict(const std::vector<float>& data_chunk);
    void reset_states();
};

#endif  // LIBSPEECH_SILERO_VAD_H
