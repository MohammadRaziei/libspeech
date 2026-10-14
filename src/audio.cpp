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
#include "libspeech/detail/log.h"


namespace speech::io {

class audio_impl {
   public:
    std::vector<std::vector<float>> audio_data; // Each channel has its own data
    int sample_rate = 0;
    int channels = 0;
    bool loaded = false;

    bool load_wav(const std::filesystem::path& file_path);
    bool load_mp3(const std::filesystem::path& file_path);
    bool load_flac(const std::filesystem::path& file_path);
    bool load_bin(const std::filesystem::path& file_path);
    bool load_vector(const std::vector<std::vector<float>>& input_data, int sample_rate);
    bool save_wav(const std::filesystem::path& output_path);
    void play();
    double duration() const;

    void to_mono();

   private:
    // Shared by load_wav/load_mp3/load_flac: miniaudio's ma_decoder already
    // embeds its own (renamed, `ma_`-prefixed) copy of dr_wav/dr_mp3/dr_flac
    // internally -- see "dr_wav_h begin" etc. in miniaudio.h -- so there is
    // no need for the separate dr_libs submodule to get identical decoding
    // for all three formats through one API. `encodingFormat` is passed
    // explicitly (rather than left to miniaudio's file-extension/trial-and-
    // error sniffing) so a mismatched extension still fails the same way
    // dr_wav/dr_mp3/dr_flac's own type-specific init functions did.
    bool load_with_miniaudio(const std::filesystem::path& file_path, ma_encoding_format format, const char* format_name);
};

} // namespace speech::io

bool speech::io::audio_impl::load_with_miniaudio(const std::filesystem::path& file_path, ma_encoding_format format, const char* format_name) {
    ma_decoder_config config = ma_decoder_config_init(ma_format_f32, /*outputChannels=*/0, /*outputSampleRate=*/0);
    config.encodingFormat = format;

    ma_decoder decoder;
    if (ma_decoder_init_file(file_path.string().c_str(), &config, &decoder) != MA_SUCCESS) {
        SPEECH_LOG(ERROR) << TAG("speech::io::Audio") << "Failed to open " << format_name << " file: " << file_path << std::endl;
        return false;
    }

    ma_uint32 out_channels = 0, out_sample_rate = 0;
    ma_decoder_get_data_format(&decoder, nullptr, &out_channels, &out_sample_rate, nullptr, 0);
    channels = static_cast<int>(out_channels);
    sample_rate = static_cast<int>(out_sample_rate);

    ma_uint64 num_frames = 0;
    ma_decoder_get_length_in_pcm_frames(&decoder, &num_frames);
    audio_data.resize(channels, std::vector<float>(num_frames)); // Allocate memory for each channel

    ma_uint64 frames_read = 0;
    if (channels == 1) {
        // Mono is already "deinterleaved": decode straight into the channel buffer
        // (no intermediate interleaved buffer, no copy loop).
        ma_decoder_read_pcm_frames(&decoder, audio_data[0].data(), num_frames, &frames_read);
    } else {
        std::vector<float> interleaved_data(num_frames * channels);
        ma_decoder_read_pcm_frames(&decoder, interleaved_data.data(), num_frames, &frames_read);

        // Deinterleave the data into separate channels
        for (int c = 0; c < channels; ++c) {
            float* dst = audio_data[c].data();
            const float* src = interleaved_data.data() + c;
            for (ma_uint64 i = 0; i < frames_read; ++i) {
                dst[i] = src[i * channels];
            }
        }
    }

    ma_decoder_uninit(&decoder);
    loaded = true;
    SPEECH_LOG(DEBUG) << TAG("speech::io::Audio") << "Loaded " << format_name << ": " << file_path << ", Sample Rate: " << sample_rate << ", Channels: " << channels << std::endl;
    return true;
}

// **Load WAV**
bool speech::io::audio_impl::load_wav(const std::filesystem::path& file_path) {
    return load_with_miniaudio(file_path, ma_encoding_format_wav, "WAV");
}

// **Load MP3**
bool speech::io::audio_impl::load_mp3(const std::filesystem::path& file_path) {
    return load_with_miniaudio(file_path, ma_encoding_format_mp3, "MP3");
}

// **Load FLAC**
bool speech::io::audio_impl::load_flac(const std::filesystem::path& file_path) {
    return load_with_miniaudio(file_path, ma_encoding_format_flac, "FLAC");
}

// **Load PCM from Binary File**
bool speech::io::audio_impl::load_bin(const std::filesystem::path& file_path) {
    std::ifstream file(file_path, std::ios::binary | std::ios::ate);
    if (!file) {
        SPEECH_LOG(ERROR) << TAG("speech::io::Audio") << "Failed to open BIN file: " << file_path << std::endl;
        return false;
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    if (size % sizeof(float) != 0) {
        SPEECH_LOG(ERROR) << TAG("speech::io::Audio") << "Invalid BIN file format: " << file_path << std::endl;
        return false;
    }
    size_t total_samples = size / sizeof(float);
    size_t num_frames = total_samples / channels; // Assuming channels is known beforehand
    audio_data.resize(channels, std::vector<float>(num_frames)); // Allocate memory for each channel

    std::vector<float> interleaved_data(total_samples);
    file.read(reinterpret_cast<char*>(interleaved_data.data()), size);

    // Deinterleave the data into separate channels
    for (size_t i = 0; i < num_frames; ++i) {
        for (int c = 0; c < channels; ++c) {
            audio_data[c][i] = interleaved_data[i * channels + c];
        }
    }

    file.close();
    loaded = true;
    SPEECH_LOG(DEBUG) << TAG("speech::io::Audio") << "Loaded BIN: " << file_path << ", Total Samples: " << total_samples << std::endl;
    return true;
}

// **Load Audio from Vector**
bool speech::io::audio_impl::load_vector(const std::vector<std::vector<float>>& input_data, int input_sample_rate) {
    if (input_data.empty() || input_sample_rate <= 0) {
        SPEECH_LOG(ERROR) << TAG("speech::io::Audio") << "Invalid audio data provided to load_vector!" << std::endl;
        return false;
    }
    audio_data = input_data;
    sample_rate = input_sample_rate;
    channels = input_data.size();
    loaded = true;
    SPEECH_LOG(DEBUG) << TAG("speech::io::Audio") << "Loaded audio from vector, Sample Rate: " << sample_rate << ", Channels: " << channels << std::endl;
    return true;
}

// **Play (Simulation)**
void audio_callback(ma_device* device, void* output, const void* input, ma_uint32 frame_count) {
    speech::io::audio_impl* audio = reinterpret_cast<speech::io::audio_impl*>(device->pUserData);
    if (!audio || audio->audio_data.empty()) return;

    static size_t current_frame = 0;
    size_t frames_to_copy = std::min(static_cast<size_t>(frame_count), audio->audio_data[0].size() - current_frame);

    if (frames_to_copy > 0) {
        float* out = static_cast<float*>(output);
        for (size_t i = 0; i < frames_to_copy; ++i) {
            for (int c = 0; c < audio->channels; ++c) {
                out[i * audio->channels + c] = audio->audio_data[c][current_frame + i];
            }
        }
        current_frame += frames_to_copy;
    } else {
        std::memset(output, 0, frame_count * audio->channels * sizeof(float)); // Silence if no data left
    }
}

void simulate_work_with_progress_bar(double duration_in_seconds) {
    // Create a progress bar
    auto progress_bar = std::make_shared<httpp::progress::bar>(100, "Playing audio ");
    // Divide the total duration into n small intervals
    const size_t n = 50;
    const double interval_duration = duration_in_seconds / n;
    const double interval_step = 100.0 / n;
    for (int i = 0; i < n; ++i ) {
        // Update the progress bar
        progress_bar->set_progress(int(i * interval_step));
        std::this_thread::sleep_for(std::chrono::duration<double>(interval_duration));
    }
    progress_bar->set_progress(100);
}

void speech::io::audio_impl::play(){
//    if (audio_data.empty()) {
//        SPEECH_LOG(ERROR) << TAG("speech::io::Audio") << "No audio loaded to play!" << std::endl;
//        return;
//    }

    ma_device_config device_config = ma_device_config_init(ma_device_type_playback);
    device_config.playback.format = ma_format_f32;
    device_config.playback.channels = channels;
    device_config.sampleRate = sample_rate;  // miniaudio's own field name
    device_config.dataCallback = audio_callback;
    device_config.pUserData = this;

    ma_device device;
    if (ma_device_init(NULL, &device_config, &device) != MA_SUCCESS) {
        SPEECH_LOG(ERROR) << TAG("speech::io::Audio") << "Failed to initialize audio device!" << std::endl;
        return;
    }

    if (ma_device_start(&device) != MA_SUCCESS) {
        SPEECH_LOG(ERROR) << TAG("speech::io::Audio") << "Failed to start audio playback!" << std::endl;
        ma_device_uninit(&device);
        return;
    }

    simulate_work_with_progress_bar(duration());
    ma_device_stop(&device);
    ma_device_uninit(&device);
}

// **Save as WAV**
bool speech::io::audio_impl::save_wav(const std::filesystem::path& output_path) {
    if (!loaded) {
        SPEECH_LOG(WARNING) << TAG("speech::io::Audio") << "No audio loaded to save!" << std::endl;
        return false;
    }

    size_t num_frames = audio_data[0].size();
    std::vector<float> interleaved_data(num_frames * channels);

    // Interleave the data from separate channels
    for (size_t i = 0; i < num_frames; ++i) {
        for (int c = 0; c < channels; ++c) {
            interleaved_data[i * channels + c] = audio_data[c][i];
        }
    }

    ma_encoder_config config = ma_encoder_config_init(ma_encoding_format_wav, ma_format_f32, channels, sample_rate);

    ma_encoder encoder;
    if (ma_encoder_init_file(output_path.string().c_str(), &config, &encoder) != MA_SUCCESS) {
        SPEECH_LOG(ERROR) << TAG("speech::io::Audio") << "Failed to save WAV file: " << output_path << std::endl;
        return false;
    }

    ma_uint64 frames_written = 0;
    ma_encoder_write_pcm_frames(&encoder, interleaved_data.data(), num_frames, &frames_written);
    ma_encoder_uninit(&encoder);
    SPEECH_LOG(DEBUG) << TAG("speech::io::Audio") << "Saved WAV file: " << output_path << std::endl;
    return true;
}

double speech::io::audio_impl::duration() const {
    if (audio_data.empty() || audio_data[0].empty()) return 0.0;
    return static_cast<double>(audio_data[0].size()) / sample_rate;
}

void speech::io::audio_impl::to_mono() {
    if (channels <= 1) {
        // If the audio is already mono, no need to process.
        return;
    }

    // Calculate the number of samples per channel.
    size_t num_frames = audio_data[0].size();

    // Create a new mono channel to store the averaged data.
    std::vector<float> mono_data(num_frames, 0.0f);

    // Average the samples across all channels.
    for (size_t i = 0; i < num_frames; ++i) {
        float sum = 0.0f;
        for (int c = 0; c < channels; ++c) {
            sum += audio_data[c][i];
        }
        mono_data[i] = sum / static_cast<float>(channels);
    }

    // Replace the multi-channel data with the mono channel.
    audio_data.clear();
    audio_data.push_back(mono_data);

    // Update the number of channels.
    channels = 1;

    SPEECH_LOG(DEBUG) << TAG("speech::io::Audio") << "Converted to mono audio." << std::endl;
}








// **Audio Class Implementation**
speech::io::audio::audio() : impl_(std::make_unique<audio_impl>()) {}

speech::io::audio::audio(const audio& other) {
    impl_ = std::make_unique<audio_impl>();
    impl_->audio_data = other.impl_->audio_data;
    impl_->sample_rate = other.impl_->sample_rate;
    impl_->channels = other.impl_->channels;
    impl_->loaded = other.impl_->loaded;
}

speech::io::audio::~audio() = default;

// **Load Audio File**
bool speech::io::audio::load(const std::filesystem::path& file_path) {
    if (!std::filesystem::exists(file_path)) {
        SPEECH_LOG(ERROR) << TAG("speech::io::Audio") << "File not found: " << file_path << std::endl;
        return false;
    }
    std::string extension = file_path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) { return std::tolower(c); });
    static const std::unordered_map<std::string, bool (audio_impl::*)(const std::filesystem::path&)> loaders = {
        {".wav", &audio_impl::load_wav},
        {".mp3", &audio_impl::load_mp3},
        {".flac", &audio_impl::load_flac},
        {".bin", &audio_impl::load_bin}
    };
    auto it = loaders.find(extension);
    if (it != loaders.end()) {
        return (impl_.get()->*(it->second))(file_path);
    }
    SPEECH_LOG(ERROR) << TAG("speech::io::Audio") << "Unsupported file format: " << file_path << std::endl;
    return false;
}

// **Load from Vector**
bool speech::io::audio::load(const std::vector<std::vector<float>>& input_data, int sample_rate) {
    return impl_->load_vector(input_data, sample_rate);
}

// **Play Audio**
void speech::io::audio::play() const{
    impl_->play();
}

// **Save Audio as WAV**
bool speech::io::audio::save(const std::filesystem::path& output_path) {
    return impl_->save_wav(output_path);
}

// **Get Audio Data**
std::vector<std::vector<float>> speech::io::audio::data() const {
    return impl_->audio_data;
}

// **Get Sample Rate**
int speech::io::audio::sample_rate() const {
    return impl_->sample_rate;
}

double speech::io::audio::duration() const {
    return impl_->duration();
}

speech::io::audio speech::io::audio::resample(int target_sample_rate) const {
    if (target_sample_rate <= 0 || target_sample_rate == impl_->sample_rate) {
        // If the target sample rate is invalid or the same as the current rate, return a copy of the current object
        return *this;
    }

    // Create a new Audio object for the resampled data
    audio resampled_audio;

    // Initialize the resampler for each channel
    speech::dsp::resample resampler(impl_->sample_rate, target_sample_rate);

    // Perform resampling for each channel
    std::vector<std::vector<float>> resampled_data(impl_->channels);
    for (int c = 0; c < impl_->channels; ++c) {
        resampled_data[c] = resampler.process(impl_->audio_data[c]);
    }

    resampled_audio.load(resampled_data, target_sample_rate);

    return resampled_audio;
}

speech::io::audio speech::io::audio::to_mono() {
    audio audio(*this);
    audio.impl_->to_mono();
    return audio;
}

std::vector<float> speech::io::audio::data(int index) const {
    if (index < 0 || index >= impl_->channels) {
        SPEECH_LOG(ERROR) << TAG("speech::io::Audio") <<
            "Invalid channel index: " + std::to_string(index) << std::endl;
        throw std::out_of_range("Invalid channel index: " + std::to_string(index));
    }
    return impl_->audio_data[index];
}
size_t speech::io::audio::size() const {
    // Was `this->data(0).size()`, which copied the whole channel just to read its length.
    return (impl_->channels > 0 && !impl_->audio_data.empty()) ? impl_->audio_data[0].size() : 0;
}

const float* speech::io::audio::channel_data(int index, size_t* length) const {
    if (index < 0 || index >= impl_->channels || index >= static_cast<int>(impl_->audio_data.size())) {
        SPEECH_LOG(ERROR) << TAG("speech::io::Audio") <<
            "Invalid channel index: " + std::to_string(index) << std::endl;
        throw std::out_of_range("Invalid channel index: " + std::to_string(index));
    }
    if (length) *length = impl_->audio_data[index].size();
    return impl_->audio_data[index].data();
}

int speech::io::audio::channels() const {
    return impl_->channels;
}

speech::io::audio &speech::io::audio::operator=(const speech::io::audio &other) {
    impl_.reset();
    impl_ = std::make_unique<audio_impl>();
    impl_->audio_data = other.impl_->audio_data;
    impl_->sample_rate = other.impl_->sample_rate;
    impl_->channels = other.impl_->channels;
    impl_->loaded = other.impl_->loaded;
    return *this;
}
