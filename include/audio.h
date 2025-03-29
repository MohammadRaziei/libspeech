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
    Audio(const Audio& other);
    ~Audio();
    Audio& operator=(const Audio&);

    bool load(const std::filesystem::path& filePath);
    bool load(const std::vector<std::vector<float>>& inputData, int sampleRate);


    void play() const;
    bool save(const std::filesystem::path& outputPath);

    [[nodiscard]] Audio to_mono();

    [[nodiscard]] Audio resample(int targetSampleRate) const;

    [[nodiscard]] std::vector<std::vector<float>> data() const;
    [[nodiscard]] std::vector<float> data(int index) const;
    [[nodiscard]] int sample_rate() const;
    [[nodiscard]] double duration() const;
    [[nodiscard]] size_t size() const;

   private:
    std::unique_ptr<AudioImpl> pImpl;
};
}

#endif // LIBSPEECH_AUDIO_H
