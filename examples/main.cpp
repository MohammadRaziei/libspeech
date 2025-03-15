#include <iostream>
#include <algorithm> // for std::copy
#include <iterator>  // for std::ostream_iterator

#include "audio.h"
#include "utils.h"
#include "common.h"

int main() {
    // Example usage
    std::string url = "https://models.silero.ai/vad_models/en.wav"; // Replace with your desired URL
    fs::path tempDir = speech::utils::getTempDirectory();           // Get the system's temp directory
    fs::path fileName = "en.wav";       // Name of the downloaded file

    std::cout << "Temporary directory: " << tempDir << std::endl;

    if (speech::utils::downloadFile(url, tempDir / fileName)) {
        std::cout << "File downloaded successfully to: " << (tempDir / fileName) << std::endl;
    } else {
        std::cout << "Failed to download the file." << std::endl;
    }

    Audio audio;

        // Load an audio file
    if (!audio.load((tempDir/fileName).string())) {
        return -1;
    }

    // Play the audio

    const std::vector<float> data = audio.data();

    std::cout << "Audio data: ";
    std::copy(data.begin(), data.begin() + 5, std::ostream_iterator<float>(std::cout, " "));
    std::cout << std::endl;

    audio.play();
    
    // Save the audio to a new file
    //    if (!audio.save("D:\\Desk\\Work\\Academy-hamrah\\courses\\assets\\output.wav")) {
    //        return -1;
    //    }


    return 0;
}

