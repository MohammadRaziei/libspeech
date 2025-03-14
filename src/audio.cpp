#include "audio.h"
#include <iostream>
#include <miniaudio.h> // Include miniaudio in the source file, not the header

// Implementation class
class AudioImpl {
   public:
    ma_decoder decoder;           // Decoder for loading and decoding audio files
    ma_device device;             // Audio device for playback
    ma_device_config deviceConfig; // Configuration for the audio device
    bool isLoaded;                // Indicates if an audio file is successfully loaded
    std::vector<float> audioData; // Buffer to store PCM data

    AudioImpl() : isLoaded(false) {}
    ~AudioImpl() {
        if (isLoaded) {
            ma_decoder_uninit(&decoder); // Uninitialize the decoder if it was initialized
        }
        ma_device_uninit(&device); // Uninitialize the audio device
    }

    // Load an audio file
    bool load(const std::string& filePath) {
        ma_result result = ma_decoder_init_file(filePath.c_str(), NULL, &decoder);
        if (result != MA_SUCCESS) {
            std::cerr << "Failed to load audio file: " << filePath << std::endl;
            return false;
        }

        // Read all PCM data into the buffer
        ma_uint64 totalFrames = 0;
        ma_uint64 framesToRead = 1024;
        float buffer[1024 * decoder.outputChannels];

        while ((totalFrames = ma_decoder_read_pcm_frames(&decoder, buffer, framesToRead, NULL)) > 0) {
            audioData.insert(audioData.end(), buffer, buffer + totalFrames * decoder.outputChannels);
        }

        isLoaded = true;
        std::cout << "Audio file loaded successfully: " << filePath << std::endl;
        return true;
    }

    // Get the sample rate of the audio
    int getSampleRate() const {
        return decoder.outputSampleRate;
    }

    // Get the audio data as a vector of floats
    std::vector<float> getData() const {
        return audioData;
    }


        // Play the loaded audio
        void play() {
            if (!isLoaded) {
                std::cerr << "No audio file is loaded. Please load an audio file first." << std::endl;
                return;
            }

            // Initialize the audio device
            deviceConfig = ma_device_config_init(ma_device_type_playback);
            deviceConfig.playback.format = decoder.outputFormat;
            deviceConfig.playback.channels = decoder.outputChannels;
            deviceConfig.sampleRate = decoder.outputSampleRate;

            // Set the callback function for audio playback
            deviceConfig.dataCallback = [](ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount) {
                ma_decoder* pDecoder = (ma_decoder*)pDevice->pUserData;
                ma_decoder_read_pcm_frames(pDecoder, pOutput, frameCount, NULL);
            };

            deviceConfig.pUserData = &decoder;

            ma_result result = ma_device_init(NULL, &deviceConfig, &device);
            if (result != MA_SUCCESS) {
                std::cerr << "Failed to initialize audio device." << std::endl;
                return;
            }

            // Start the audio playback
            result = ma_device_start(&device);
            if (result != MA_SUCCESS) {
                std::cerr << "Failed to start audio playback." << std::endl;
                ma_device_uninit(&device);
                return;
            }

            std::cout << "Playing audio... Press Enter to stop." << std::endl;
            std::cin.get(); // Wait for user input to stop playback

            // Stop and uninitialize the audio device
            ma_device_stop(&device);
            ma_device_uninit(&device);
        }

        // Save the audio data to a new file
        bool save(const std::string& outputPath) {
            if (!isLoaded) {
                std::cerr << "No audio file is loaded. Please load an audio file first." << std::endl;
                return false;
            }

            // Initialize the encoder for saving audio
            ma_encoder encoder;
            ma_encoder_config encoderConfig = ma_encoder_config_init(ma_encoding_format_wav, decoder.outputFormat, decoder.outputChannels, decoder.outputSampleRate);

            ma_result result = ma_encoder_init_file(outputPath.c_str(), &encoderConfig, &encoder);
            if (result != MA_SUCCESS) {
                std::cerr << "Failed to initialize encoder for saving audio." << std::endl;
                return false;
            }

            // Write audio frames from the buffer
            ma_encoder_write_pcm_frames(&encoder, audioData.data(), audioData.size() / decoder.outputChannels, NULL);

            ma_encoder_uninit(&encoder);
            std::cout << "Audio saved successfully: " << outputPath << std::endl;
            return true;
        }

};

// Constructor: Initialize the implementation
Audio::Audio() : pImpl(std::make_unique<AudioImpl>()) {}

// Destructor: Clean up the implementation
Audio::~Audio() = default;

// Load an audio file
bool Audio::load(const std::string& filePath) {
    return pImpl->load(filePath);
}

// Play the loaded audio
void Audio::play() {
    pImpl->play();
}

// Save the audio data to a new file
bool Audio::save(const std::string& outputPath) {
    return pImpl->save(outputPath);
}

// Get the audio data as a vector of floats
std::vector<float> Audio::data() const {
    return pImpl->getData();
}

// Get the sample rate of the audio
int Audio::sampleRate() const {
    return pImpl->getSampleRate();
}