#ifndef LIBSPEECH_AUDIO_H
#define LIBSPEECH_AUDIO_H

#include <string>
#include <memory>
#include <vector>
#include <filesystem>

#include "resample.h"

// Forward declaration for the implementation class
namespace speech{

class AudioImpl;

class Audio {
   public:
    Audio();
    Audio(const Audio& other);
    ~Audio();

    bool load(const std::filesystem::path& filePath);
    bool load(const std::vector<std::vector<float>>& inputData, int sampleRate);

    Audio& to_mono();

    void play() const;
    bool save(const std::filesystem::path& outputPath);

    // Resample function
    Audio resample(int targetSampleRate) const;

    [[nodiscard]] std::vector<std::vector<float>> data() const;
    [[nodiscard]] std::vector<float> data(int index) const;
    [[nodiscard]] int sampleRate() const;
    [[nodiscard]] double duration() const;

   private:
    std::unique_ptr<AudioImpl> pImpl;
};
}

#endif // LIBSPEECH_AUDIO_H
