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
    std::vector<float> audioData;
    int sampleRate = 0;
    int channels = 0;
    bool loaded = false;

    bool loadWAV(const std::filesystem::path& filePath);
    bool loadMP3(const std::filesystem::path& filePath);
    bool loadFLAC(const std::filesystem::path& filePath);
    bool loadBin(const std::filesystem::path& filePath);
    bool loadVector(const std::vector<float>& inputData, int sampleRate, int channels);
    bool saveWAV(const std::filesystem::path& outputPath);
    void play();
    double duration() const;
};
}

// **Load WAV**
bool speech::AudioImpl::loadWAV(const std::filesystem::path& filePath) {
    drwav wav;
    if (!drwav_init_file(&wav, filePath.string().c_str(), NULL)) {
        std::cerr << "Failed to open WAV file: " << filePath << std::endl;
        return false;
    }

    sampleRate = wav.sampleRate;
    channels = wav.channels;
    size_t numFrames = wav.totalPCMFrameCount * channels;

    audioData.resize(numFrames);
    drwav_read_pcm_frames_f32(&wav, wav.totalPCMFrameCount, audioData.data());
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
    size_t numFrames = drmp3_get_pcm_frame_count(&mp3) * channels;

    audioData.resize(numFrames);
    drmp3_read_pcm_frames_f32(&mp3, numFrames / channels, audioData.data());
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
    size_t numFrames = flac->totalPCMFrameCount * channels;

    audioData.resize(numFrames);
    drflac_read_pcm_frames_f32(flac, flac->totalPCMFrameCount, audioData.data());
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

    audioData.resize(size / sizeof(float));
    file.read(reinterpret_cast<char*>(audioData.data()), size);
    file.close();

    loaded = true;
    std::cout << "Loaded BIN: " << filePath << ", Total Samples: " << audioData.size() << std::endl;
    return true;
}

// **Load Audio from Vector**
bool speech::AudioImpl::loadVector(const std::vector<float>& inputData, int inputSampleRate, int inputChannels) {
    if (inputData.empty() || inputSampleRate <= 0 || inputChannels <= 0) {
        std::cerr << "Invalid audio data provided to loadVector!" << std::endl;
        return false;
    }

    audioData = inputData;
    sampleRate = inputSampleRate;
    channels = inputChannels;
    loaded = true;

    std::cout << "Loaded audio from vector, Sample Rate: " << sampleRate << ", Channels: " << channels << std::endl;
    return true;
}

// **Play (Simulation)**
void audioCallback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount) {
    speech::AudioImpl* audio = reinterpret_cast<speech::AudioImpl*>(pDevice->pUserData);
    if (!audio || audio->audioData.empty()) return;

    static size_t currentFrame = 0;
    size_t framesToCopy = std::min(static_cast<size_t>(frameCount * audio->channels), audio->audioData.size() - currentFrame);

    if (framesToCopy > 0) {
        std::memcpy(pOutput, audio->audioData.data() + currentFrame, framesToCopy * sizeof(float));
        currentFrame += framesToCopy;
    } else {
        std::memset(pOutput, 0, frameCount * audio->channels * sizeof(float));  // Silence if no data left
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

void speech::AudioImpl::play() {
    if (!loaded || audioData.empty()) {
        std::cerr << "No audio loaded to play!" << std::endl;
        return;
    }

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

    drwav_write_pcm_frames(&wav, audioData.size() / channels, audioData.data());
    drwav_uninit(&wav);

    std::cout << "Saved WAV file: " << outputPath << std::endl;
    return true;
}

double speech::AudioImpl::duration() const {
    return static_cast<double>(audioData.size()) / sampleRate;
}

// **Audio Class Implementation**
speech::Audio::Audio() : pImpl(std::make_unique<AudioImpl>()) {}
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
bool speech::Audio::load(const std::vector<float>& inputData, int sampleRate, int channels) {
    return pImpl->loadVector(inputData, sampleRate, channels);
}

// **Play Audio**
void speech::Audio::play() {
    pImpl->play();
}

// **Save Audio as WAV**
bool speech::Audio::save(const std::filesystem::path& outputPath) {
    return pImpl->saveWAV(outputPath);
}

// **Get Audio Data**
std::vector<float> speech::Audio::data() const {
    return pImpl->audioData;
}

// **Get Sample Rate**
int speech::Audio::sampleRate() const {
    return pImpl->sampleRate;
}

double speech::Audio::duration() const {
    return pImpl->duration();
}
