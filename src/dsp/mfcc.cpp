#include "libspeech/dsp/mfcc.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "aixlog.hpp"
#include "libspeech/detail/log.h"
#include "libspeech/dsp/dct.h"
#include "libspeech/dsp/stft.h"

namespace speech::dsp {

namespace {
constexpr const char* log_tag = "speech::dsp::MFCC";
constexpr int block_frames = 64;  // 64 * 512 * 4 B * 2 planes = 256 KiB: stays in L2

// HTK-style conversion, matching AudioFlux's auditory_freToMel/auditory_melToFre.
float hz_to_mel(float hz) { return 2595.0f * std::log10(1.0f + hz / 700.0f); }
float mel_to_hz(float mel) { return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f); }

}  // namespace

mfcc::mfcc(params params) : params_(params) {
    if (params_.num_mel_filters < 2) {
        SPEECH_LOG(ERROR) << TAG(log_tag) << "num_mel_filters must be >= 2 (got " << params_.num_mel_filters
                   << ")." << std::endl;
        throw std::invalid_argument("num_mel_filters must be >= 2.");
    }
    if (params_.num_coefficients < 1 || params_.num_coefficients > params_.num_mel_filters) {
        SPEECH_LOG(ERROR) << TAG(log_tag) << "num_coefficients must be in [1, num_mel_filters] (got "
                   << params_.num_coefficients << " with num_mel_filters=" << params_.num_mel_filters
                   << ")." << std::endl;
        throw std::invalid_argument("num_coefficients must be in [1, num_mel_filters].");
    }

    stft_ = std::make_unique<stft>(params_.radix2_exp, window_type::hann, params_.slide_length);

    const int fft_length = stft_->fft_length();
    const float high_freq =
        (params_.high_freq_hz > 0.0f) ? params_.high_freq_hz : params_.sample_rate / 2.0f;

    mel_filter_bank_ = build_mel_filter_bank(params_.num_mel_filters, fft_length, params_.sample_rate,
                                         params_.low_freq_hz, high_freq);

    // Sparse mel filters: identical weights to the dense bank, minus the zeros.
    const int num_fft_bins = fft_length / 2 + 1;
    mel_start_.assign(params_.num_mel_filters, 0);
    mel_len_.assign(params_.num_mel_filters, 0);
    mel_offset_.assign(params_.num_mel_filters, 0);
    min_bin_ = num_fft_bins;
    max_bin_ = -1;
    for (int m = 0; m < params_.num_mel_filters; ++m) {
        int first = -1;
        int last = -1;
        for (int k = 0; k < num_fft_bins; ++k) {
            if (mel_filter_bank_[m][k] != 0.0f) {
                if (first < 0) first = k;
                last = k;
            }
        }
        mel_offset_[m] = static_cast<int>(mel_weights_.size());
        if (first >= 0) {
            mel_start_[m] = first;
            mel_len_[m] = last - first + 1;
            mel_weights_.insert(mel_weights_.end(), mel_filter_bank_[m].begin() + first,
                               mel_filter_bank_[m].begin() + last + 1);
            min_bin_ = std::min(min_bin_, first);
            max_bin_ = std::max(max_bin_, last);
        }
    }
    if (max_bin_ < 0) {
        min_bin_ = 0;
        max_bin_ = -1;
    }

    dct_ = std::make_unique<dct_ii_plan>(params_.num_mel_filters, params_.num_coefficients,
                                   /*orthonormal=*/true);

    SPEECH_LOG(DEBUG) << TAG(log_tag) << "Created MFCC: sample_rate=" << params_.sample_rate
               << ", num_mel_filters=" << params_.num_mel_filters
               << ", num_coefficients=" << params_.num_coefficients
               << ", fft_length=" << fft_length << std::endl;
}

mfcc::~mfcc() = default;

std::vector<std::vector<float>> mfcc::build_mel_filter_bank(int num_mel_filters, int fft_length,
                                                           int sample_rate, float low_freq_hz,
                                                           float high_freq_hz) {
    const int num_fft_bins = fft_length / 2 + 1;

    const float mel_low = hz_to_mel(low_freq_hz);
    const float mel_high = hz_to_mel(high_freq_hz);

    // num_mel_filters+2 boundary points define num_mel_filters triangular filters.
    std::vector<int> bin_points(num_mel_filters + 2);
    for (int i = 0; i < num_mel_filters + 2; ++i) {
        float mel = mel_low + (mel_high - mel_low) * i / (num_mel_filters + 1);
        float hz = mel_to_hz(mel);
        int bin = static_cast<int>(std::floor((fft_length + 1) * hz / sample_rate));
        bin_points[i] = std::clamp(bin, 0, num_fft_bins - 1);
    }

    std::vector<std::vector<float>> filter_bank(num_mel_filters,
                                                std::vector<float>(num_fft_bins, 0.0f));
    for (int m = 0; m < num_mel_filters; ++m) {
        int left = bin_points[m];
        int center = bin_points[m + 1];
        int right = bin_points[m + 2];

        // Degenerate (zero-width) filters can happen when fft_length/sample_rate
        // is too coarse to resolve the requested number of mel filters at the
        // low end of the spectrum -- leave them all-zero rather than dividing
        // by zero. compute() still produces a value for these filters (0,
        // feeding log(epsilon) downstream), it's just not a meaningful one;
        // callers hitting this should use a larger fft_length or fewer filters.
        if (center > left) {
            for (int k = left; k < center; ++k) {
                filter_bank[m][k] = static_cast<float>(k - left) / (center - left);
            }
        }
        if (right > center) {
            for (int k = center; k < right; ++k) {
                filter_bank[m][k] = static_cast<float>(right - k) / (right - center);
            }
        }
    }

    return filter_bank;
}

mfcc_matrix mfcc::compute_flat(const float* signal, std::size_t n) {
    return run(signal, n, /*apply_dct=*/true);
}

mfcc_matrix mfcc::compute_log_mel_flat(const float* signal, std::size_t n) {
    return run(signal, n, /*apply_dct=*/false);
}

mfcc_matrix mfcc::run(const float* signal, std::size_t n, bool apply_dct) {
    constexpr float log_epsilon = 1e-10f;
    const int fft_length = stft_->fft_length();
    const int num_bins = fft_length / 2 + 1;
    const int num_frames = stft_->cal_time_length(static_cast<int>(n));
    const int num_mel = params_.num_mel_filters;
    const int num_coef = apply_dct ? params_.num_coefficients : num_mel;  // columns per output frame

    mfcc_matrix out;
    out.num_coefficients = num_coef;
    if (num_frames <= 0) {
        return out;
    }
    out.num_frames = num_frames;
    out.data.resize(static_cast<std::size_t>(num_frames) * num_coef);  // fully overwritten below

    const int num_blocks = (num_frames + block_frames - 1) / block_frames;
    stft* stft = stft_.get();
    float* out_data = out.data.data();

    // Blocks are independent, so they can run on separate threads; each thread
    // owns its power-spectrum block buffer and per-frame scratch.
#pragma omp parallel if (num_frames >= 256)
    {
        std::vector<float> power_block(static_cast<std::size_t>(block_frames) * num_bins);
        std::vector<float> mel_heap(static_cast<std::size_t>(num_mel));
        float* mel_buf = mel_heap.data();

#pragma omp for schedule(static)
        for (int blk = 0; blk < num_blocks; ++blk) {
            const int f0 = blk * block_frames;
            const int frames = std::min(block_frames, num_frames - f0);
            stft->power_spectra(signal, n, f0, frames, power_block.data());

            for (int fr = 0; fr < frames; ++fr) {
                const float* power = power_block.data() + static_cast<std::size_t>(fr) * num_bins;
                for (int m = 0; m < num_mel; ++m) {
                    const float* w = mel_weights_.data() + mel_offset_[m];
                    const float* p = power + mel_start_[m];
                    float energy = 0.0f;
                    for (int j = 0; j < mel_len_[m]; ++j) {
                        energy += w[j] * p[j];
                    }
                    mel_buf[m] = std::log(energy + log_epsilon);
                }
                float* dst = out_data + static_cast<std::size_t>(f0 + fr) * num_coef;
                if (apply_dct) {
                    dct_->apply(mel_buf, dst);
                } else {
                    std::copy(mel_buf, mel_buf + num_mel, dst);
                }
            }
        }
    }

    SPEECH_LOG(DEBUG) << TAG(log_tag) << (apply_dct ? "Computed MFCC: " : "Computed log-mel: ") << n << " samples -> " << num_frames
               << " frames x " << num_coef << (apply_dct ? " coefficients." : " mel bands.") << std::endl;
    return out;
}

std::vector<std::vector<float>> mfcc::compute(const std::vector<float>& signal) {
    // Legacy nested-vector API: one copy out of the flat result.
    mfcc_matrix flat = compute_flat(signal.data(), signal.size());
    std::vector<std::vector<float>> result(flat.num_frames);
    for (int f = 0; f < flat.num_frames; ++f) {
        const auto begin = flat.data.begin() + static_cast<std::ptrdiff_t>(f) * flat.num_coefficients;
        result[f].assign(begin, begin + flat.num_coefficients);
    }
    return result;
}

}  // namespace speech::dsp
