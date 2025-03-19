#include "models/silero_vad.h" // Include the SileroVAD model header
#include "audio.h"           // For speech::Audio
#include <iostream>
#include "utils/utils.h"           // For downloadFile utility


#include "dsp/resample.h"

int main2() {
    try {
        // Create a Resample object with default settings
        Resample resampler(44100, 16000); // Resample from 44.1kHz to 16kHz

        // Input audio data (example)
        std::vector<float> inputData = {0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0, -0.5};

        // Perform resampling
        std::cout << "Resampled Data:" << std::endl;

        std::vector<float> outputData = resampler.resample(inputData);

        // Print the resampled data
        for (float value : outputData) {
            std::cout << value << " ";
        }
        std::cout << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}

int main() {
    try {
        // Initialize the SileroVAD model
        SileroVadModel vad;

        // Download and load the audio file
        std::string url = "https://github.com/MohammadRaziei/libspeech/releases/download/resources/example-en-biden-large.mp3";
        std::filesystem::path tempDir = std::filesystem::temp_directory_path();
        std::filesystem::path fileName = speech::utils::downloadFile(url, tempDir, false, false);

        if (fileName.empty()) {
            std::cerr << "Download failed or file does not exist." << std::endl;
            return -1;
        }

        // Load the audio file
        speech::Audio audio;
        if (!audio.load(fileName)) {
            std::cerr << "Failed to load audio file." << std::endl;
            return -1;
        }
        const auto audio16k = audio.to_mono().resample(vad.sample_rate);

//        audio16k.play();


        // Process the audio
        vad.processOnVector(audio16k.data(0));

        // Retrieve and print speech timestamps
        std::vector<timestamp_t> stamps = vad.get_speech_timestamps();
        for (const auto& stamp : stamps) {
            std::cout << "Speech detected from " << std::setprecision(5) << stamp.start_s() << " to " << stamp.end_s() << " seconds." << std::endl;
        }

        // Optionally, reset the internal state
        vad.reset();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return 0;
}
