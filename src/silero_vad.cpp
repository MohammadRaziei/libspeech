//
// Created by mohammad on 3/16/25.
//

#include "silero_vad.h"

#include <iostream>
#include <onnxruntime_cxx_api.h>

SileroVadModel::SileroVadModel(const std::string& url,
                               int sample_rate,
                               float threshold,
                               int min_silence_duration_ms,
                               int speech_pad_ms,
                               int min_speech_duration_ms,
                               float max_speech_duration_s,
                               const std::filesystem::path& base_dir)
    : ONNXModel(url, base_dir),
      sample_rate(sample_rate),
      threshold(threshold),
      min_silence_samples(min_silence_duration_ms * sample_rate / 1000),
      min_silence_samples_at_max_speech(98 * sample_rate / 1000),
      min_speech_samples(min_speech_duration_ms * sample_rate / 1000),
      max_speech_samples(max_speech_duration_s * sample_rate),
      speech_pad_samples(speech_pad_ms * sample_rate / 1000) {
    _context.assign(64, 0.0f); // Context size for 16kHz
    _state.resize(2 * 1 * 128, 0.0f);
}

void SileroVadModel::process(const std::vector<float>& input_wav) {
    reset_states();
    int audio_length_samples = static_cast<int>(input_wav.size());

    for (size_t j = 0; j < static_cast<size_t>(audio_length_samples); j += 512) {
        if (j + 512 > static_cast<size_t>(audio_length_samples)) break;

        std::vector<float> chunk(input_wav.begin() + j, input_wav.begin() + j + 512);
        predict(chunk);
    }

    if (current_speech.start >= 0) {
        current_speech.end = audio_length_samples;
        speeches.push_back(current_speech);
        reset_states();
    }
}

const std::vector<timestamp_t> SileroVadModel::get_speech_timestamps() const {
    return speeches;
}

void SileroVadModel::predict(const std::vector<float>& data_chunk) {
    // Combine context and current chunk
    std::vector<float> new_data(576, 0.0f);
    std::copy(_context.begin(), _context.end(), new_data.begin());
    std::copy(data_chunk.begin(), data_chunk.end(), new_data.begin() + 64);

    // Run inference
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, new_data.data(), new_data.size(), nullptr, 2);
    auto output_tensors = session->Run(Ort::RunOptions{nullptr}, nullptr, &input_tensor, 1, nullptr, 1);

    float speech_prob = output_tensors[0].GetTensorMutableData<float>()[0];
    current_sample += 512;

    if (speech_prob >= threshold) {
        if (!triggered) {
            triggered = true;
            current_speech.start = current_sample - 512;
        }
    } else {
        if (triggered && (current_sample - temp_end) >= min_silence_samples) {
            current_speech.end = temp_end;
            if (current_speech.end - current_speech.start > min_speech_samples) {
                speeches.push_back(current_speech);
            }
            reset_states();
        }
    }

    std::copy(new_data.end() - 64, new_data.end(), _context.begin());
}

void SileroVadModel::reset_states() {
    std::fill(_context.begin(), _context.end(), 0.0f);
    std::fill(_state.begin(), _state.end(), 0.0f);
    triggered = false;
    temp_end = 0;
    current_sample = 0;
    prev_end = 0;
    next_start = 0;
    speeches.clear();
    current_speech = timestamp_t();
}