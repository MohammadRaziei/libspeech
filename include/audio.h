#ifndef LIBSPEECH_AUDIO_H
#define LIBSPEECH_AUDIO_H

#include <string>
#include <memory>
#include <vector>
#include <filesystem>

// Forward declaration for the implementation class
namespace speech{

class AudioImpl;

class Audio {
   public:
    Audio();
    ~Audio();

    bool load(const std::filesystem::path& filePath);
    bool load(const std::vector<float>& inputData, int sampleRate, int channels);

    void play();
    bool save(const std::filesystem::path& outputPath);

    [[nodiscard]] std::vector<float> data() const;
    [[nodiscard]] int sampleRate() const;

   private:
    std::unique_ptr<AudioImpl> pImpl;
};
}

#endif // LIBSPEECH_AUDIO_H
