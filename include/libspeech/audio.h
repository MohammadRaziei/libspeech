#ifndef LIBSPEECH_AUDIO_H
#define LIBSPEECH_AUDIO_H

#include <string>
#include <memory>
#include <vector>
#include <filesystem>

#include "libspeech/export.h"


// Forward declaration for the implementation class
namespace speech::io{

class audio_impl;

class SPEECH_API audio {
   public:
    audio();
    audio(const audio& other);
    ~audio();
    audio& operator=(const audio&);

    bool load(const std::filesystem::path& file_path);
    bool load(const std::vector<std::vector<float>>& input_data, int sample_rate);


    void play() const;
    bool save(const std::filesystem::path& output_path);

    [[nodiscard]] audio to_mono();

    [[nodiscard]] audio resample(int target_sample_rate) const;

    // Returns a copy scaled so that its RMS level is target_dbfs (dBFS, 0 = full scale), with the gain lowered if
    // that would push a sample above peak_ceiling (linear amplitude), so the result never clips. All channels get
    // the same gain, which keeps their balance. Silent audio is returned unchanged. See speech::dsp::rms_normalize.
    // Throws std::invalid_argument for a non-finite target_dbfs or a peak_ceiling that is not a finite number > 0.
    [[nodiscard]] audio normalize_rms(double target_dbfs = -20.0, double peak_ceiling = 0.99) const;

    [[nodiscard]] std::vector<std::vector<float>> data() const;
    [[nodiscard]] std::vector<float> data(int index) const;

    // Borrowed, copy-free view of one channel: returns a pointer to its samples
    // and stores the sample count in *length. Valid until this Audio is next
    // modified (load/assign). Throws std::out_of_range on a bad channel index.
    [[nodiscard]] const float* channel_data(int index, size_t* length) const;
    // All samples as one planar block: channel c is [c * size(), (c + 1) * size()). The returned
    // pointer co-owns the buffer, so it stays valid even if this Audio is loaded again or destroyed
    // (a load builds a new buffer and never touches the old one). Null when nothing is loaded.
    [[nodiscard]] std::shared_ptr<float> shared_samples() const;
    [[nodiscard]] int channels() const;
    [[nodiscard]] int sample_rate() const;
    [[nodiscard]] double duration() const;
    [[nodiscard]] size_t size() const;

   private:
    std::unique_ptr<audio_impl> impl_;
};
}  // namespace speech::io

#endif // LIBSPEECH_AUDIO_H
