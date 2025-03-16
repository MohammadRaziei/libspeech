//#include <iostream>
//#include <algorithm> // for std::copy
//#include <iterator>  // for std::ostream_iterator
//
//#include "audio.h"
//#include "utils.h"
//#include "common.h"
//
//#include <vector>
//
//int main() {
//
//
//        // Initialize the ONNX Runtime environment
//    // Example usage
//    std::string url = "https://github.com/MohammadRaziei/libspeech/releases/download/resources/example-en-long.wav"; // Replace with your desired URL
////    std::string url = "https://github.com/MohammadRaziei/libspeech/releases/download/resources/example-en-short.mp3";
//    std::filesystem::path tempDir = std::filesystem::temp_directory_path();           // Get the system's temp directory
//
//    std::cout << "Temporary directory: " << tempDir << std::endl;
//    std::filesystem::path fileName = speech::utils::downloadFile(url, tempDir, false, false);
//    if (fileName.empty()) {
//        std::cerr << "Download failed or file does not exist." << std::endl;
//    } else {
//        std::cout << "File downloaded to: " << fileName << std::endl;
//    }
//
//    speech::Audio audio;
//
//        // Load an audio file
//    if (!audio.load(fileName)) {
//        return -1;
//    }
//
//    // Play the audio
//
//    const std::vector<float> data = audio.data();
//
//    std::cout << "Audio data: ";
//    std::copy(data.begin(), data.begin() + 5, std::ostream_iterator<float>(std::cout, " "));
//    std::cout << std::endl;
//
//    audio.play();
//
//
//    return 0;
//
//
//}
//

#include "silero_vad.h"
#include "audio.h" // For speech::Audio
#include <iostream>

int main() {
    try {
        // Initialize the SileroVAD model
        SileroVadModel vad("silero_vad.onnx");

        // Download and load the audio file
        std::string url = "https://github.com/MohammadRaziei/libspeech/releases/download/resources/example-en-long.wav";
        std::filesystem::path tempDir = std::filesystem::temp_directory_path();
        std::filesystem::path fileName = speech::utils::downloadFile(url, tempDir, false, false);

        if (fileName.empty()) {
            std::cerr << "Download failed or file does not exist." << std::endl;
            return -1;
        }

        speech::Audio audio;
        if (!audio.load(fileName)) {
            std::cerr << "Failed to load audio file." << std::endl;
            return -1;
        }

        // Get audio data
        const std::vector<float> input_wav = audio.data();

        // Process the audio
        vad.process(input_wav);

        // Retrieve and print speech timestamps
        std::vector<timestamp_t> stamps = vad.get_speech_timestamps();
        for (const auto& stamp : stamps) {
            std::cout << "Speech detected from " << stamp.start << " to " << stamp.end << " samples." << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return 0;
}
