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

    [[nodiscard]] std::vector<std::vector<float>> data() const;
    [[nodiscard]] std::vector<float> data(int index) const;

    // Borrowed, copy-free view of one channel: returns a pointer to its samples
    // and stores the sample count in *length. Valid until this Audio is next
    // modified (load/assign). Throws std::out_of_range on a bad channel index.
    [[nodiscard]] const float* channel_data(int index, size_t* length) const;
    [[nodiscard]] int channels() const;
    [[nodiscard]] int sample_rate() const;
    [[nodiscard]] double duration() const;
    [[nodiscard]] size_t size() const;

   private:
    std::unique_ptr<audio_impl> impl_;
};
}  // namespace speech::io

#endif // LIBSPEECH_AUDIO_H
