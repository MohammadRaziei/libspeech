#include "audio.h"           // For speech::Audio
#include <iostream>

#include "models/silero_vad.h" // Include the SileroVAD model header
#include "utils/utils.h"           // For downloadFile utility

#include "models/facebook_denoiser.h"
#include "models/speechbrain_denoiser.h"
#include "dsp/resample.h"

#include <aixlog.hpp>


int main0() {
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



int main2() {
    AixLog::Log::init<AixLog::SinkCout>(AixLog::Severity::trace);
    try {

        // Initialize the SileroVAD model
        SileroVadModel vad;

        // Download and load the audio file
//        std::string url = "https://github.com/MohammadRaziei/libspeech/releases/download/resources/example-en-biden-large.mp3";
        std::string url = "https://github.com/MohammadRaziei/libspeech/releases/download/resources/example-en-biden-xlarge.mp3";

        std::filesystem::path tempDir = std::filesystem::temp_directory_path();
        std::filesystem::path fileName = speech::utils::downloadFile(url, tempDir, false, false);

        if (fileName.empty()) {
            LOG(FATAL) << "Download failed or file does not exist." << std::endl;
            return -1;
        }

        // Load the audio file
        speech::Audio audio;
        if (!audio.load(fileName)) {
            LOG(FATAL) << "Failed to load audio file." << std::endl;
            return -1;
        }
        const auto audio16k = audio.to_mono().resample(vad.sample_rate);

//        audio16k.play();


        // Process the audio
        vad.processOnVector(audio16k.data(0));

        // Retrieve and print speech timestamps
        std::vector<timestamp_t> stamps = vad.get_speech_timestamps();
        for (const auto& stamp : stamps) {
            LOG(INFO) << "Speech detected from " << AixLog::Color::cyan << stamp.start_s() << " to " << stamp.end_s() << AixLog::Color::none << " seconds." << std::endl;
        }

        // Optionally, reset the internal state
        vad.reset();
    } catch (const std::exception& e) {
        LOG(FATAL) << "Error: " << e.what() << std::endl;
    }
    return 0;
}


int main5() {
    try {
        AixLog::Log::init<AixLog::SinkCout>(AixLog::Severity::trace);

        // Define the model URL and base directory.


        // Prepare input audio data (example: 1 second of silence at 16000 Hz).
//        std::string url = "https://github.com/MohammadRaziei/libspeech/releases/download/resources/alex-noisy.mp3";
//        std::string url = "https://github.com/MohammadRaziei/libspeech/releases/download/resources/example-en-biden-xlarge.mp3";
        std::string url = "https://github.com/MohammadRaziei/libspeech/releases/download/resources/audio-fa-noisy.mp3";


        std::filesystem::path fileName = speech::utils::downloadFile(url,
                 std::filesystem::temp_directory_path(), false, false);

        if (fileName.empty()) {
            LOG(FATAL) << "Download failed or file does not exist." << std::endl;
            return -1;
        }

        // Load the audio file
        speech::Audio audio;
        if (!audio.load(fileName)) {
            LOG(FATAL) << "Failed to load audio file." << std::endl;
            return -1;
        }
//        audio.play();



        SpeechBrainDenoiser denoiser("speechbrain-sepformer-wham16k-enhancement.onnx", 16000);

        auto audio_process = audio.to_mono().resample(denoiser.sample_rate);
        auto process_data = audio_process.data(0);
        process_data.resize(denoiser.sample_rate * 32);

//        audio_process.play();

        // Process the audio.
        std::vector<float> denoised_audio_data = denoiser.process(process_data);

        audio_process.load({denoised_audio_data}, denoiser.sample_rate);
        audio_process = audio_process.resample(audio.sample_rate());
        // Print the size of the denoised audio.
        LOG(INFO) << "Denoised audio size: " << audio_process.size() << std::endl;
        audio_process.play();

    } catch (const std::exception& e) {
        LOG(FATAL) << "Error: " << e.what() << std::endl;
    }

    return 0;
}


int main1() {
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



int main4() {
    AixLog::Log::init<AixLog::SinkCout>(AixLog::Severity::trace);
    try {

        // Initialize the SileroVAD model
        SileroVadModel vad;

        // Download and load the audio file
//        std::string url = "https://github.com/MohammadRaziei/libspeech/releases/download/resources/example-en-biden-large.mp3";
        std::string url = "https://github.com/MohammadRaziei/libspeech/releases/download/resources/example-en-biden-xlarge.mp3";

        std::filesystem::path tempDir = std::filesystem::temp_directory_path();
        std::filesystem::path fileName = speech::utils::downloadFile(url, tempDir, false, false);

        if (fileName.empty()) {
            LOG(FATAL) << "Download failed or file does not exist." << std::endl;
            return -1;
        }

        // Load the audio file
        speech::Audio audio;
        if (!audio.load(fileName)) {
            LOG(FATAL) << "Failed to load audio file." << std::endl;
            return -1;
        }
        const auto audio16k = audio.to_mono().resample(vad.sample_rate);

//        audio16k.play();


        // Process the audio
        vad.processOnVector(audio16k.data(0));

        // Retrieve and print speech timestamps
        std::vector<timestamp_t> stamps = vad.get_speech_timestamps();
        for (const auto& stamp : stamps) {
            LOG(INFO) << "Speech detected from " << AixLog::Color::cyan << stamp.start_s() << " to " << stamp.end_s() << AixLog::Color::none << " seconds." << std::endl;
        }

        // Optionally, reset the internal state
        vad.reset();
    } catch (const std::exception& e) {
        LOG(FATAL) << "Error: " << e.what() << std::endl;
    }
    return 0;
}


int main() {
    try {
        AixLog::Log::init<AixLog::SinkCout>(AixLog::Severity::trace);

        // Define the model URL and base directory.

        // Create an instance of FacebookDenoiser.

        // Prepare input audio data (example: 1 second of silence at 16000 Hz).
//        std::string url = "https://github.com/MohammadRaziei/libspeech/releases/download/resources/alex-noisy.mp3";
//        std::string url = "https://github.com/MohammadRaziei/libspeech/releases/download/resources/example-en-biden-xlarge.mp3";
        std::string url = "https://github.com/MohammadRaziei/libspeech/releases/download/resources/audio-fa-noisy.mp3";


        std::filesystem::path fileName = speech::utils::downloadFile(url,
                                                                     std::filesystem::temp_directory_path(), false, false);

        if (fileName.empty()) {
            LOG(FATAL) << "Download failed or file does not exist." << std::endl;
            return -1;
        }

        // Load the audio file
        speech::Audio audio;
        if (!audio.load(fileName)) {
            LOG(FATAL) << "Failed to load audio file." << std::endl;
            return -1;
        }
//        audio.play();

        std::string model_url = "facebook-denoiser-dns64.onnx";

        FacebookDenoiser denoiser(model_url);

        auto audio_process = audio.to_mono().resample(denoiser.sample_rate);

        // Process the audio.
        std::vector<float> denoised_audio_data = denoiser.process(audio_process.data(0));

        audio_process.load({denoised_audio_data}, denoiser.sample_rate);
        audio_process.play();

        // Print the size of the denoised audio.
        LOG(INFO) << "Denoised audio size: " << audio_process.size() << std::endl;
    } catch (const std::exception& e) {
        LOG(FATAL) << "Error: " << e.what() << std::endl;
    }

    return 0;
}