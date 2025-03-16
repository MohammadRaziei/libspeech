#include <iostream>
#include <algorithm> // for std::copy
#include <iterator>  // for std::ostream_iterator

#include "audio.h"
#include "utils.h"
#include "common.h"

#include <iostream>
#include <thread>
#include <indicators/progress_bar.hpp>

int main() {
    // Example usage
//    std::string url = "https://github.com/MohammadRaziei/libspeech/releases/download/resources/example-en-long.wav"; // Replace with your desired URL
    std::string url = "https://github.com/MohammadRaziei/libspeech/releases/download/resources/example-en-short.mp3";
    std::filesystem::path tempDir = std::filesystem::temp_directory_path();           // Get the system's temp directory

    std::cout << "Temporary directory: " << tempDir << std::endl;
    std::filesystem::path fileName = speech::utils::downloadFile(url, tempDir, false, false);
    if (fileName.empty()) {
        std::cerr << "Download failed or file does not exist." << std::endl;
    } else {
        std::cout << "File downloaded to: " << fileName << std::endl;
    }

    speech::Audio audio;

        // Load an audio file
    if (!audio.load(fileName.string())) {
        return -1;
    }

    // Play the audio

    const std::vector<float> data = audio.data();

    std::cout << "Audio data: ";
    std::copy(data.begin(), data.begin() + 5, std::ostream_iterator<float>(std::cout, " "));
    std::cout << std::endl;

    audio.play();


    return 0;


}

