#ifndef LIBSPEECH_AUDIO_H
#define LIBSPEECH_AUDIO_H

#include <string>
#include <memory> // For std::unique_ptr
#include <vector> // For std::vector

// Forward declaration of the implementation class
class AudioImpl;

class Audio {
   public:
    Audio();
    ~Audio();

    // Load an audio file
    bool load(const std::string& filePath);

    // Play the loaded audio
    void play();

    // Save the audio data to a new file
    bool save(const std::string& outputPath);

    // Get the audio data as a vector of floats
    std::vector<float> data() const;

    // Get the sample rate of the audio
    int sampleRate() const;

   private:
    std::unique_ptr<AudioImpl> pImpl; // Pointer to the implementation
};

#endif // LIBSPEECH_AUDIO_H