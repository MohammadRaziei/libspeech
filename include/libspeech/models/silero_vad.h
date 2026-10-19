//
// Created by mohammad on 3/16/25.
//

#ifndef LIBSPEECH_SILERO_VAD_H
#define LIBSPEECH_SILERO_VAD_H

#include "libspeech/models/onnx_model.h"

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "libspeech/utils/utils.h"
#include "libspeech/export.h"

namespace speech::models {

// Forward declaration of timestamp_t class
class speech_timestamp {
   public:
    speech_timestamp(int start = -1, int end = -1, const int sample_rate = 16000)
        : start(start), end(end), sample_rate(sample_rate) {}

    speech_timestamp& operator=(const speech_timestamp& a) {
        start = a.start;
        end = a.end;
        return *this;
    }

    bool operator==(const speech_timestamp& a) const {
        return (start == a.start && end == a.end);
    }

    std::string c_str() const {
        return speech::utils::format("{start:%08d(%0.2fs), end:%08d(%0.2fs)}", start, start_s(), end, end_s());
    }

    float start_s() const {
        return static_cast<float>(start) / static_cast<float>(sample_rate);
    }

    float end_s() const {
        return static_cast<float>(end) / static_cast<float>(sample_rate);
    }

   public:
    int start;
    int end;
    int sample_rate;

};




class SPEECH_API silero_vad_model : public onnx_model {

   public:
    // Constructor: sets model path, sample rate, window size (ms), and other parameters.
    silero_vad_model(const std::string& model_path = "silero-vad.onnx",
                   const int sample_rate = 16000, int window_frame_size = 32,
                   float threshold = 0.5, int min_silence_duration_ms = 100,
                   int speech_pad_ms = 30, int min_speech_duration_ms = 250,
                   float max_speech_duration_s = std::numeric_limits<float>::infinity(),
                   bool ignore_cache = false);

    // Process the entire audio input.
    void process_on_vector(const std::vector<float>& input_wav);

    // Returns the detected speech timestamps.
    const std::vector<speech_timestamp> get_speech_timestamps() const;

    // ---- Streaming -------------------------------------------------------------------------
    // Feed audio in chunks of any size (a microphone callback, a socket, ...) instead of one whole vector.
    // Typical loop:
    //     vad.reset();
    //     for each chunk: vad.feed(chunk.data(), chunk.size());
    //                     for (auto& seg : vad.pop_speech_timestamps()) ...   // segments that just ended
    //     vad.finish();                                                    // end of audio: closes an open segment
    //     for (auto& seg : vad.pop_speech_timestamps()) ...
    // Timestamps are in samples counted from the last reset(). process_on_vector() is implemented as
    // reset() + feed(all) + finish(), so streaming gives exactly the same segments as the batch call no matter
    // how the audio is split into chunks. Not thread-safe (same as the rest of this class).

    // Appends `n` samples (mono, [-1, 1], at the model's sample rate). Complete windows are processed
    // immediately; a partial window is kept until the next call.
    void feed(const float* data, std::size_t n);

    // Ends the stream: an open speech segment is closed at the last sample fed. A trailing partial window is
    // dropped (as in process_on_vector). Call reset() before feeding unrelated audio.
    void finish();

    // Segments completed since the previous call (each segment is returned once). A segment is only reported
    // after the silence that ends it has been seen, or by finish(). get_speech_timestamps() keeps returning all.
    std::vector<speech_timestamp> pop_speech_timestamps();

    // True while a speech segment is open (speech has started and its end has not been decided yet).
    bool in_speech() const { return triggered; }

    // Samples fed since the last reset().
    std::size_t samples_fed() const { return total_fed; }

    // Public method to reset the internal state.
    void reset();



   private:
    // Reads the input/output names from the session (inputs: audio, state[, sr]; outputs: probability, new state).
    void init_node_names();

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
    // The tensor names are read from the loaded model (init_node_names()): the released models call the
    // new state output "stateN", and silero-vad-half has no "sr" input at all.
    std::vector<std::string> input_name_storage;
    std::vector<std::string> output_name_storage;
    std::vector<const char*> input_node_names;
    bool has_sr_input = true;
    std::vector<float> input;
    unsigned int size_state = 2 * 1 * 128;
    std::vector<float> _state;
    std::vector<int64_t> sr;
    int64_t input_node_dims[2] = {};
    const int64_t state_node_dims[3] = { 2, 1, 128 };
    const int64_t sr_node_dims[1] = { 1 };
    std::vector<Ort::Value> ort_outputs;
    std::vector<const char*> output_node_names;

    // Model configuration parameters
    float threshold;
    int min_silence_samples;
    int min_silence_samples_at_max_speech;
    int min_speech_samples;
    float max_speech_samples;
    int speech_pad_samples;
    int audio_length_samples;

    // Streaming state: samples waiting for a complete window, total fed, and how many of `speeches` pop_*() returned.
    std::vector<float> pending;
    std::size_t total_fed = 0;
    std::size_t popped = 0;

    // State management
    bool triggered = false;
    unsigned int temp_end = 0;
    unsigned int current_sample = 0;
    int prev_end;
    int next_start = 0;
    std::vector<speech_timestamp> speeches;
    speech_timestamp current_speech;

    // Resets internal state (_state, _context, etc.)
    void reset_states();

    // Inference: runs inference on one chunk of input data.
    void predict(const std::vector<float>& data_chunk);
};

}  // namespace speech::models

#endif  // LIBSPEECH_SILERO_VAD_H
