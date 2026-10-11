#include "utest.h"

#include <cmath>
#include <stdexcept>
#include <vector>

#include "libspeech/dsp/mfcc.h"

namespace {
std::vector<float> make_sine(int n, float freq_hz, int sample_rate) {
    std::vector<float> out(n);
    for (int i = 0; i < n; ++i) {
        out[i] = std::sin(2.0 * M_PI * freq_hz * i / sample_rate);
    }
    return out;
}
}  // namespace

UTEST(mfcc, output_shape_matches_params) {
    speech::dsp::mfcc::params params;
    params.sample_rate = 16000;
    params.num_mel_filters = 26;
    params.num_coefficients = 13;
    params.radix2_exp = 9;  // fft_length = 512

    speech::dsp::mfcc mfcc(params);

    int data_length = 512 + 128 * 9;  // 10 frames at default slide_length=fft_length/4
    auto signal = make_sine(data_length, 440.0f, 16000);

    auto coeffs = mfcc.compute(signal);

    ASSERT_EQ(coeffs.size(), static_cast<size_t>(10));
    for (const auto& frame : coeffs) {
        ASSERT_EQ(frame.size(), static_cast<size_t>(13));
    }
}

UTEST(mfcc, output_is_finite_for_real_signal) {
    speech::dsp::mfcc::params params;
    params.sample_rate = 16000;
    speech::dsp::mfcc mfcc(params);

    int data_length = 1024 + 256 * 9;
    auto signal = make_sine(data_length, 220.0f, 16000);
    auto coeffs = mfcc.compute(signal);

    for (const auto& frame : coeffs) {
        for (float v : frame) {
            ASSERT_TRUE(std::isfinite(v));
        }
    }
}

UTEST(mfcc, output_is_finite_for_silence) {
    // Regression-style test: log(0 + epsilon) must not produce NaN/Inf.
    speech::dsp::mfcc::params params;
    params.sample_rate = 16000;
    speech::dsp::mfcc mfcc(params);

    int data_length = 1024 + 256 * 5;
    std::vector<float> silence(data_length, 0.0f);
    auto coeffs = mfcc.compute(silence);

    for (const auto& frame : coeffs) {
        for (float v : frame) {
            ASSERT_TRUE(std::isfinite(v));
        }
    }
}

UTEST(mfcc, stationary_signal_gives_consistent_coefficients_across_frames) {
    // For a sinusoid at a frequency EXACTLY aligned to an FFT bin (k *
    // sample_rate / fft_length), shifting the analysis window only rotates
    // the phase of that bin's DFT coefficient -- magnitude is provably
    // invariant to the shift. That makes log-mel-energy (and therefore the
    // MFCC output) essentially identical across interior frames, which is a
    // real, backend-agnostic correctness signal (unlike an arbitrary
    // frequency, where windowed-FFT magnitude legitimately varies frame to
    // frame due to spectral leakage -- that's expected DSP behavior, not a
    // bug, so it would make a bad test).
    speech::dsp::mfcc::params params;
    params.sample_rate = 16000;
    params.radix2_exp = 9;  // fft_length = 512, bin spacing = 16000/512 = 31.25 Hz
    speech::dsp::mfcc mfcc(params);

    const float bin_aligned_freq = 14.0f * 16000.0f / 512.0f;  // = 437.5 Hz, exactly bin 14
    int data_length = 512 + 128 * 19;  // 20 frames
    auto signal = make_sine(data_length, bin_aligned_freq, 16000);
    auto coeffs = mfcc.compute(signal);

    // Compare frame 5 to frame 15 (both well away from edge-taper effects).
    const auto& a = coeffs[5];
    const auto& b = coeffs[15];
    for (size_t i = 0; i < a.size(); ++i) {
        ASSERT_NEAR(static_cast<double>(a[i]), static_cast<double>(b[i]), 0.05);
    }
}

UTEST(mfcc, too_few_mel_filters_throws) {
    speech::dsp::mfcc::params params;
    params.sample_rate = 16000;
    params.num_mel_filters = 1;

    bool threw = false;
    try {
        speech::dsp::mfcc mfcc(params);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    ASSERT_TRUE(threw);
}

UTEST(mfcc, too_many_coefficients_throws) {
    speech::dsp::mfcc::params params;
    params.sample_rate = 16000;
    params.num_mel_filters = 13;
    params.num_coefficients = 20;  // more than num_mel_filters

    bool threw = false;
    try {
        speech::dsp::mfcc mfcc(params);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    ASSERT_TRUE(threw);
}

