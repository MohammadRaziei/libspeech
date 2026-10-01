#include "libspeech/audio.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <cctype>    // For std::tolower
#include <cstring>   // For memcpy
#include <algorithm> // For std::transform
#include <thread>
#include <chrono>


#include <httpp/progress.hpp>
// miniaudio.h embeds its own renamed copy of dr_wav/dr_mp3/dr_flac
// internally (see "dr_wav_h begin" etc. inside it) and exposes them
// uniformly through ma_decoder/ma_encoder -- no need for the separate
// dr_libs submodule.
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#include "libspeech/dsp/resample.h"

#include "aixlog.hpp"


namespace speech::io {

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

   private:
    // Shared by loadWAV/loadMP3/loadFLAC: miniaudio's ma_decoder already
    // embeds its own (renamed, `ma_`-prefixed) copy of dr_wav/dr_mp3/dr_flac
    // internally -- see "dr_wav_h begin" etc. in miniaudio.h -- so there is
    // no need for the separate dr_libs submodule to get identical decoding
    // for all three formats through one API. `encodingFormat` is passed
    // explicitly (rather than left to miniaudio's file-extension/trial-and-
    // error sniffing) so a mismatched extension still fails the same way
    // dr_wav/dr_mp3/dr_flac's own type-specific init functions did.
    bool loadWithMiniaudio(const std::filesystem::path& filePath, ma_encoding_format format, const char* formatName);
};

} // namespace speech::io

bool speech::io::AudioImpl::loadWithMiniaudio(const std::filesystem::path& filePath, ma_encoding_format format, const char* formatName) {
    ma_decoder_config config = ma_decoder_config_init(ma_format_f32, /*outputChannels=*/0, /*outputSampleRate=*/0);
    config.encodingFormat = format;

    ma_decoder decoder;
    if (ma_decoder_init_file(filePath.string().c_str(), &config, &decoder) != MA_SUCCESS) {
        LOG(ERROR) << TAG("speech::io::Audio") << "Failed to open " << formatName << " file: " << filePath << std::endl;
        return false;
    }

    ma_uint32 outChannels = 0, outSampleRate = 0;
    ma_decoder_get_data_format(&decoder, nullptr, &outChannels, &outSampleRate, nullptr, 0);
    channels = static_cast<int>(outChannels);
    sampleRate = static_cast<int>(outSampleRate);

    ma_uint64 numFrames = 0;
    ma_decoder_get_length_in_pcm_frames(&decoder, &numFrames);
    audioData.resize(channels, std::vector<float>(numFrames)); // Allocate memory for each channel

    ma_uint64 framesRead = 0;
    if (channels == 1) {
        // Mono is already "deinterleaved": decode straight into the channel buffer
        // (no intermediate interleaved buffer, no copy loop).
        ma_decoder_read_pcm_frames(&decoder, audioData[0].data(), numFrames, &framesRead);
    } else {
        std::vector<float> interleavedData(numFrames * channels);
        ma_decoder_read_pcm_frames(&decoder, interleavedData.data(), numFrames, &framesRead);

        // Deinterleave the data into separate channels
        for (int c = 0; c < channels; ++c) {
            float* dst = audioData[c].data();
            const float* src = interleavedData.data() + c;
            for (ma_uint64 i = 0; i < framesRead; ++i) {
                dst[i] = src[i * channels];
            }
        }
    }

    ma_decoder_uninit(&decoder);
    loaded = true;
    LOG(DEBUG) << TAG("speech::io::Audio") << "Loaded " << formatName << ": " << filePath << ", Sample Rate: " << sampleRate << ", Channels: " << channels << std::endl;
    return true;
}

// **Load WAV**
bool speech::io::AudioImpl::loadWAV(const std::filesystem::path& filePath) {
    return loadWithMiniaudio(filePath, ma_encoding_format_wav, "WAV");
}

// **Load MP3**
bool speech::io::AudioImpl::loadMP3(const std::filesystem::path& filePath) {
    return loadWithMiniaudio(filePath, ma_encoding_format_mp3, "MP3");
}

// **Load FLAC**
bool speech::io::AudioImpl::loadFLAC(const std::filesystem::path& filePath) {
    return loadWithMiniaudio(filePath, ma_encoding_format_flac, "FLAC");
}

// **Load PCM from Binary File**
bool speech::io::AudioImpl::loadBin(const std::filesystem::path& filePath) {
    std::ifstream file(filePath, std::ios::binary | std::ios::ate);
    if (!file) {
        LOG(ERROR) << TAG("speech::io::Audio") << "Failed to open BIN file: " << filePath << std::endl;
        return false;
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    if (size % sizeof(float) != 0) {
        LOG(ERROR) << TAG("speech::io::Audio") << "Invalid BIN file format: " << filePath << std::endl;
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
    LOG(DEBUG) << TAG("speech::io::Audio") << "Loaded BIN: " << filePath << ", Total Samples: " << totalSamples << std::endl;
    return true;
}

// **Load Audio from Vector**
bool speech::io::AudioImpl::loadVector(const std::vector<std::vector<float>>& inputData, int inputSampleRate) {
    if (inputData.empty() || inputSampleRate <= 0) {
        LOG(ERROR) << TAG("speech::io::Audio") << "Invalid audio data provided to loadVector!" << std::endl;
        return false;
    }
    audioData = inputData;
    sampleRate = inputSampleRate;
    channels = inputData.size();
    loaded = true;
    LOG(DEBUG) << TAG("speech::io::Audio") << "Loaded audio from vector, Sample Rate: " << sampleRate << ", Channels: " << channels << std::endl;
    return true;
}

// **Play (Simulation)**
void audioCallback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount) {
    speech::io::AudioImpl* audio = reinterpret_cast<speech::io::AudioImpl*>(pDevice->pUserData);
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
    auto progressBar = std::make_shared<httpp::progress::bar>(100, "Playing audio ");
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

void speech::io::AudioImpl::play(){
//    if (audioData.empty()) {
//        LOG(ERROR) << TAG("speech::io::Audio") << "No audio loaded to play!" << std::endl;
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
        LOG(ERROR) << TAG("speech::io::Audio") << "Failed to initialize audio device!" << std::endl;
        return;
    }

    if (ma_device_start(&device) != MA_SUCCESS) {
        LOG(ERROR) << TAG("speech::io::Audio") << "Failed to start audio playback!" << std::endl;
        ma_device_uninit(&device);
        return;
    }

    simulateWorkWithProgressBar(duration());
    ma_device_stop(&device);
    ma_device_uninit(&device);
}

// **Save as WAV**
bool speech::io::AudioImpl::saveWAV(const std::filesystem::path& outputPath) {
    if (!loaded) {
        LOG(WARNING) << TAG("speech::io::Audio") << "No audio loaded to save!" << std::endl;
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

    ma_encoder_config config = ma_encoder_config_init(ma_encoding_format_wav, ma_format_f32, channels, sampleRate);

    ma_encoder encoder;
    if (ma_encoder_init_file(outputPath.string().c_str(), &config, &encoder) != MA_SUCCESS) {
        LOG(ERROR) << TAG("speech::io::Audio") << "Failed to save WAV file: " << outputPath << std::endl;
        return false;
    }

    ma_uint64 framesWritten = 0;
    ma_encoder_write_pcm_frames(&encoder, interleavedData.data(), numFrames, &framesWritten);
    ma_encoder_uninit(&encoder);
    LOG(DEBUG) << TAG("speech::io::Audio") << "Saved WAV file: " << outputPath << std::endl;
    return true;
}

double speech::io::AudioImpl::duration() const {
    if (audioData.empty() || audioData[0].empty()) return 0.0;
    return static_cast<double>(audioData[0].size()) / sampleRate;
}

void speech::io::AudioImpl::to_mono() {
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

    LOG(DEBUG) << TAG("speech::io::Audio") << "Converted to mono audio." << std::endl;
}








// **Audio Class Implementation**
speech::io::Audio::Audio() : pImpl(std::make_unique<AudioImpl>()) {}

speech::io::Audio::Audio(const Audio& other) {
    pImpl = std::make_unique<AudioImpl>();
    pImpl->audioData = other.pImpl->audioData;
    pImpl->sampleRate = other.pImpl->sampleRate;
    pImpl->channels = other.pImpl->channels;
    pImpl->loaded = other.pImpl->loaded;
}

speech::io::Audio::~Audio() = default;

// **Load Audio File**
bool speech::io::Audio::load(const std::filesystem::path& filePath) {
    if (!std::filesystem::exists(filePath)) {
        LOG(ERROR) << TAG("speech::io::Audio") << "File not found: " << filePath << std::endl;
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
    LOG(ERROR) << TAG("speech::io::Audio") << "Unsupported file format: " << filePath << std::endl;
    return false;
}

// **Load from Vector**
bool speech::io::Audio::load(const std::vector<std::vector<float>>& inputData, int sampleRate) {
    return pImpl->loadVector(inputData, sampleRate);
}

// **Play Audio**
void speech::io::Audio::play() const{
    pImpl->play();
}

// **Save Audio as WAV**
bool speech::io::Audio::save(const std::filesystem::path& outputPath) {
    return pImpl->saveWAV(outputPath);
}

// **Get Audio Data**
std::vector<std::vector<float>> speech::io::Audio::data() const {
    return pImpl->audioData;
}

// **Get Sample Rate**
int speech::io::Audio::sample_rate() const {
    return pImpl->sampleRate;
}

double speech::io::Audio::duration() const {
    return pImpl->duration();
}

speech::io::Audio speech::io::Audio::resample(int targetSampleRate) const {
    if (targetSampleRate <= 0 || targetSampleRate == pImpl->sampleRate) {
        // If the target sample rate is invalid or the same as the current rate, return a copy of the current object
        return *this;
    }

    // Create a new Audio object for the resampled data
    Audio resampledAudio;

    // Initialize the resampler for each channel
    speech::dsp::Resample resampler(pImpl->sampleRate, targetSampleRate);

    // Perform resampling for each channel
    std::vector<std::vector<float>> resampledData(pImpl->channels);
    for (int c = 0; c < pImpl->channels; ++c) {
        resampledData[c] = resampler.resample(pImpl->audioData[c]);
    }

    resampledAudio.load(resampledData, targetSampleRate);

    return resampledAudio;
}

speech::io::Audio speech::io::Audio::to_mono() {
    Audio audio(*this);
    audio.pImpl->to_mono();
    return audio;
}

std::vector<float> speech::io::Audio::data(int index) const {
    if (index < 0 || index >= pImpl->channels) {
        LOG(ERROR) << TAG("speech::io::Audio") <<
            "Invalid channel index: " + std::to_string(index) << std::endl;
        throw std::out_of_range("Invalid channel index: " + std::to_string(index));
    }
    return pImpl->audioData[index];
}
size_t speech::io::Audio::size() const {
    // Was `this->data(0).size()`, which copied the whole channel just to read its length.
    return (pImpl->channels > 0 && !pImpl->audioData.empty()) ? pImpl->audioData[0].size() : 0;
}

const float* speech::io::Audio::channelData(int index, size_t* length) const {
    if (index < 0 || index >= pImpl->channels || index >= static_cast<int>(pImpl->audioData.size())) {
        LOG(ERROR) << TAG("speech::io::Audio") <<
            "Invalid channel index: " + std::to_string(index) << std::endl;
        throw std::out_of_range("Invalid channel index: " + std::to_string(index));
    }
    if (length) *length = pImpl->audioData[index].size();
    return pImpl->audioData[index].data();
}

int speech::io::Audio::channels() const {
    return pImpl->channels;
}

speech::io::Audio &speech::io::Audio::operator=(const speech::io::Audio &other) {
    pImpl.reset();
    pImpl = std::make_unique<AudioImpl>();
    pImpl->audioData = other.pImpl->audioData;
    pImpl->sampleRate = other.pImpl->sampleRate;
    pImpl->channels = other.pImpl->channels;
    pImpl->loaded = other.pImpl->loaded;
    return *this;
}
