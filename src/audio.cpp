#include "audio.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <cctype>    // For std::tolower
#include <cstring>   // For memcpy
#include <algorithm> // For std::transform
#include <thread>
#include <chrono>
#include "progressbar.h"
// Include `dr_wav`, `dr_mp3`, and `dr_flac`
#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"
#define DR_MP3_IMPLEMENTATION
#include "dr_mp3.h"
#define DR_FLAC_IMPLEMENTATION
#include "dr_flac.h"
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

namespace speech {

class AudioImpl {
   public:
    std::vector<std::vector<float>> audioData; // Each channel has its own data
    int sampleRate = 0;
    int channels = 0;
    bool loaded = false;

    bool loadWAV(const std::filesystem::path& filePath);
    bool loadMP3(const std::filesystem::path& filePath);
    bool loadFLAC(const std::filesystem::path& filePath);
    bool loadBin(const std::filesystem::path& filePath);
    bool loadVector(const std::vector<std::vector<float>>& inputData, int sampleRate);
    bool saveWAV(const std::filesystem::path& outputPath);
    void play();
    double duration() const;

    void to_mono();
};

} // namespace speech

// **Load WAV**
bool speech::AudioImpl::loadWAV(const std::filesystem::path& filePath) {
    drwav wav;
    if (!drwav_init_file(&wav, filePath.string().c_str(), NULL)) {
        std::cerr << "Failed to open WAV file: " << filePath << std::endl;
        return false;
    }
    sampleRate = wav.sampleRate;
    channels = wav.channels;
    size_t numFrames = wav.totalPCMFrameCount;
    audioData.resize(channels, std::vector<float>(numFrames)); // Allocate memory for each channel

    std::vector<float> interleavedData(numFrames * channels);
    drwav_read_pcm_frames_f32(&wav, numFrames, interleavedData.data());

    // Deinterleave the data into separate channels
    for (size_t i = 0; i < numFrames; ++i) {
        for (int c = 0; c < channels; ++c) {
            audioData[c][i] = interleavedData[i * channels + c];
        }
    }

    drwav_uninit(&wav);
    loaded = true;
    std::cout << "Loaded WAV: " << filePath << ", Sample Rate: " << sampleRate << ", Channels: " << channels << std::endl;
    return true;
}

// **Load MP3**
bool speech::AudioImpl::loadMP3(const std::filesystem::path& filePath) {
    drmp3 mp3;
    if (!drmp3_init_file(&mp3, filePath.string().c_str(), NULL)) {
        std::cerr << "Failed to open MP3 file: " << filePath << std::endl;
        return false;
    }
    sampleRate = mp3.sampleRate;
    channels = mp3.channels;
    size_t numFrames = drmp3_get_pcm_frame_count(&mp3);
    audioData.resize(channels, std::vector<float>(numFrames)); // Allocate memory for each channel

    std::vector<float> interleavedData(numFrames * channels);
    drmp3_read_pcm_frames_f32(&mp3, numFrames, interleavedData.data());

    // Deinterleave the data into separate channels
    for (size_t i = 0; i < numFrames; ++i) {
        for (int c = 0; c < channels; ++c) {
            audioData[c][i] = interleavedData[i * channels + c];
        }
    }

    drmp3_uninit(&mp3);
    loaded = true;
    std::cout << "Loaded MP3: " << filePath << ", Sample Rate: " << sampleRate << ", Channels: " << channels << std::endl;
    return true;
}

// **Load FLAC**
bool speech::AudioImpl::loadFLAC(const std::filesystem::path& filePath) {
    drflac* flac = drflac_open_file(filePath.string().c_str(), NULL);
    if (!flac) {
        std::cerr << "Failed to open FLAC file: " << filePath << std::endl;
        return false;
    }
    sampleRate = flac->sampleRate;
    channels = flac->channels;
    size_t numFrames = flac->totalPCMFrameCount;
    audioData.resize(channels, std::vector<float>(numFrames)); // Allocate memory for each channel

    std::vector<float> interleavedData(numFrames * channels);
    drflac_read_pcm_frames_f32(flac, numFrames, interleavedData.data());

    // Deinterleave the data into separate channels
    for (size_t i = 0; i < numFrames; ++i) {
        for (int c = 0; c < channels; ++c) {
            audioData[c][i] = interleavedData[i * channels + c];
        }
    }

    drflac_close(flac);
    loaded = true;
    std::cout << "Loaded FLAC: " << filePath << ", Sample Rate: " << sampleRate << ", Channels: " << channels << std::endl;
    return true;
}

// **Load PCM from Binary File**
bool speech::AudioImpl::loadBin(const std::filesystem::path& filePath) {
    std::ifstream file(filePath, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Failed to open BIN file: " << filePath << std::endl;
        return false;
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    if (size % sizeof(float) != 0) {
        std::cerr << "Invalid BIN file format: " << filePath << std::endl;
        return false;
    }
    size_t totalSamples = size / sizeof(float);
    size_t numFrames = totalSamples / channels; // Assuming channels is known beforehand
    audioData.resize(channels, std::vector<float>(numFrames)); // Allocate memory for each channel

    std::vector<float> interleavedData(totalSamples);
    file.read(reinterpret_cast<char*>(interleavedData.data()), size);

    // Deinterleave the data into separate channels
    for (size_t i = 0; i < numFrames; ++i) {
        for (int c = 0; c < channels; ++c) {
            audioData[c][i] = interleavedData[i * channels + c];
        }
    }

    file.close();
    loaded = true;
    std::cout << "Loaded BIN: " << filePath << ", Total Samples: " << totalSamples << std::endl;
    return true;
}

// **Load Audio from Vector**
bool speech::AudioImpl::loadVector(const std::vector<std::vector<float>>& inputData, int inputSampleRate) {
    if (inputData.empty() || inputSampleRate <= 0) {
        std::cerr << "Invalid audio data provided to loadVector!" << std::endl;
        return false;
    }
    audioData = inputData;
    sampleRate = inputSampleRate;
    channels = inputData.size();
    loaded = true;
    std::cout << "Loaded audio from vector, Sample Rate: " << sampleRate << ", Channels: " << channels << std::endl;
    return true;
}

// **Play (Simulation)**
void audioCallback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount) {
    speech::AudioImpl* audio = reinterpret_cast<speech::AudioImpl*>(pDevice->pUserData);
    if (!audio || audio->audioData.empty()) return;

    static size_t currentFrame = 0;
    size_t framesToCopy = std::min(static_cast<size_t>(frameCount), audio->audioData[0].size() - currentFrame);

    if (framesToCopy > 0) {
        float* output = static_cast<float*>(pOutput);
        for (size_t i = 0; i < framesToCopy; ++i) {
            for (int c = 0; c < audio->channels; ++c) {
                output[i * audio->channels + c] = audio->audioData[c][currentFrame + i];
            }
        }
        currentFrame += framesToCopy;
    } else {
        std::memset(pOutput, 0, frameCount * audio->channels * sizeof(float)); // Silence if no data left
    }
}

void simulateWorkWithProgressBar(double durationInSeconds) {
    // Create a progress bar
    auto progressBar = speech::bar::createProgressBar("Playing audio ", indicators::Color::yellow);
    // Divide the total duration into n small intervals
    const size_t n = 50;
    const double intervalDuration = durationInSeconds / n;
    const double intervalStep = 100.0 / n;
    for (int i = 0; i < n; ++i ) {
        // Update the progress bar
        progressBar->set_progress(int(i * intervalStep));
        std::this_thread::sleep_for(std::chrono::duration<double>(intervalDuration));
    }
    progressBar->set_progress(100);
}

void speech::AudioImpl::play(){
//    if (audioData.empty()) {
//        std::cerr << "No audio loaded to play!" << std::endl;
//        return;
//    }

    ma_device_config deviceConfig = ma_device_config_init(ma_device_type_playback);
    deviceConfig.playback.format = ma_format_f32;
    deviceConfig.playback.channels = channels;
    deviceConfig.sampleRate = sampleRate;
    deviceConfig.dataCallback = audioCallback;
    deviceConfig.pUserData = this;

    ma_device device;
    if (ma_device_init(NULL, &deviceConfig, &device) != MA_SUCCESS) {
        std::cerr << "Failed to initialize audio device!" << std::endl;
        return;
    }

    if (ma_device_start(&device) != MA_SUCCESS) {
        std::cerr << "Failed to start audio playback!" << std::endl;
        ma_device_uninit(&device);
        return;
    }

    simulateWorkWithProgressBar(duration());
    ma_device_stop(&device);
    ma_device_uninit(&device);
}

// **Save as WAV**
bool speech::AudioImpl::saveWAV(const std::filesystem::path& outputPath) {
    if (!loaded) {
        std::cerr << "No audio loaded to save!" << std::endl;
        return false;
    }

    size_t numFrames = audioData[0].size();
    std::vector<float> interleavedData(numFrames * channels);

    // Interleave the data from separate channels
    for (size_t i = 0; i < numFrames; ++i) {
        for (int c = 0; c < channels; ++c) {
            interleavedData[i * channels + c] = audioData[c][i];
        }
    }

    drwav_data_format format = {};
    format.container = drwav_container_riff;
    format.format = DR_WAVE_FORMAT_IEEE_FLOAT;
    format.channels = channels;
    format.sampleRate = sampleRate;
    format.bitsPerSample = 32;

    drwav wav;
    if (!drwav_init_file_write(&wav, outputPath.string().c_str(), &format, NULL)) {
        std::cerr << "Failed to save WAV file: " << outputPath << std::endl;
        return false;
    }

    drwav_write_pcm_frames(&wav, numFrames, interleavedData.data());
    drwav_uninit(&wav);
    std::cout << "Saved WAV file: " << outputPath << std::endl;
    return true;
}

double speech::AudioImpl::duration() const {
    if (audioData.empty() || audioData[0].empty()) return 0.0;
    return static_cast<double>(audioData[0].size()) / sampleRate;
}

void speech::AudioImpl::to_mono() {
    if (channels <= 1) {
        // If the audio is already mono, no need to process.
        return;
    }

    // Calculate the number of samples per channel.
    size_t numFrames = audioData[0].size();

    // Create a new mono channel to store the averaged data.
    std::vector<float> monoData(numFrames, 0.0f);

    // Average the samples across all channels.
    for (size_t i = 0; i < numFrames; ++i) {
        float sum = 0.0f;
        for (int c = 0; c < channels; ++c) {
            sum += audioData[c][i];
        }
        monoData[i] = sum / static_cast<float>(channels);
    }

    // Replace the multi-channel data with the mono channel.
    audioData.clear();
    audioData.push_back(monoData);

    // Update the number of channels.
    channels = 1;

    std::cout << "Converted to mono audio." << std::endl;
}








// **Audio Class Implementation**
speech::Audio::Audio() : pImpl(std::make_unique<AudioImpl>()) {}

speech::Audio::Audio(const Audio& other) {
    pImpl = std::make_unique<AudioImpl>();
    pImpl->audioData = other.pImpl->audioData;
    pImpl->sampleRate = other.pImpl->sampleRate;
    pImpl->channels = other.pImpl->channels;
}

speech::Audio::~Audio() = default;

// **Load Audio File**
bool speech::Audio::load(const std::filesystem::path& filePath) {
    if (!std::filesystem::exists(filePath)) {
        std::cerr << "File not found: " << filePath << std::endl;
        return false;
    }
    std::string extension = filePath.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) { return std::tolower(c); });
    static const std::unordered_map<std::string, bool (AudioImpl::*)(const std::filesystem::path&)> loaders = {
        {".wav", &AudioImpl::loadWAV},
        {".mp3", &AudioImpl::loadMP3},
        {".flac", &AudioImpl::loadFLAC},
        {".bin", &AudioImpl::loadBin}
    };
    auto it = loaders.find(extension);
    if (it != loaders.end()) {
        return (pImpl.get()->*(it->second))(filePath);
    }
    std::cerr << "Unsupported file format: " << filePath << std::endl;
    return false;
}

// **Load from Vector**
bool speech::Audio::load(const std::vector<std::vector<float>>& inputData, int sampleRate) {
    return pImpl->loadVector(inputData, sampleRate);
}

// **Play Audio**
void speech::Audio::play() const{
    pImpl->play();
}

// **Save Audio as WAV**
bool speech::Audio::save(const std::filesystem::path& outputPath) {
    return pImpl->saveWAV(outputPath);
}

// **Get Audio Data**
std::vector<std::vector<float>> speech::Audio::data() const {
    return pImpl->audioData;
}

// **Get Sample Rate**
int speech::Audio::sampleRate() const {
    return pImpl->sampleRate;
}

double speech::Audio::duration() const {
    return pImpl->duration();
}

speech::Audio speech::Audio::resample(int targetSampleRate) const {
    if (targetSampleRate <= 0 || targetSampleRate == pImpl->sampleRate) {
        // If the target sample rate is invalid or the same as the current rate, return a copy of the current object
        return *this;
    }

    // Create a new Audio object for the resampled data
    Audio resampledAudio;

    // Initialize the resampler for each channel
    Resample resampler(pImpl->sampleRate, targetSampleRate);

    // Perform resampling for each channel
    std::vector<std::vector<float>> resampledData(pImpl->channels);
    for (int c = 0; c < pImpl->channels; ++c) {
        resampledData[c] = resampler.resample(pImpl->audioData[c]);
    }

    resampledAudio.load(resampledData, targetSampleRate);

    return resampledAudio;
}

speech::Audio& speech::Audio::to_mono() {
    pImpl->to_mono();
    return *this;
}

std::vector<float> speech::Audio::data(int index) const {
    if (index < 0 || index >= pImpl->channels) {
        throw std::out_of_range("Invalid channel index: " + std::to_string(index));
    }
    return pImpl->audioData[index];
}