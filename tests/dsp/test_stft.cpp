#include "utest.h"

#include <cmath>
#include <stdexcept>
#include <vector>

#include "libspeech/dsp/stft.h"

namespace {
std::vector<float> make_sine(int n, float freq_hz, int sample_rate) {
    std::vector<float> out(n);
    for (int i = 0; i < n; ++i) {
        out[i] = std::sin(2.0 * M_PI * freq_hz * i / sample_rate);
    }
    return out;
}
}  // namespace

UTEST(stft, fft_length_and_slide_length_defaults) {
    speech::dsp::stft stft(9);  // fft_length = 512
    ASSERT_EQ(stft.fft_length(), 512);
    ASSERT_EQ(stft.slide_length(), 128);  // default: fft_length/4
}

UTEST(stft, cal_time_length_matches_frame_count) {
    speech::dsp::stft stft(9);  // fft_length=512, slide_length=128
    // data_length chosen so (data_length - fft_length) is an exact multiple of slide_length.
    int data_length = 512 + 128 * 9;  // 10 frames
    ASSERT_EQ(stft.cal_time_length(data_length), 10);
}

UTEST(stft, stft_produces_expected_frame_count_and_shape) {
    speech::dsp::stft stft(9);
    int data_length = 512 + 128 * 9;
    auto signal = make_sine(data_length, 440.0f, 16000);

    auto [real, imag] = stft.compute(signal);

    ASSERT_EQ(real.size(), static_cast<size_t>(10));
    ASSERT_EQ(imag.size(), static_cast<size_t>(10));
    for (const auto& frame : real) {
        ASSERT_EQ(frame.size(), static_cast<size_t>(512));
    }
}

UTEST(stft, too_short_input_produces_no_frames) {
    speech::dsp::stft stft(9);  // fft_length=512
    std::vector<float> short_signal(100, 0.0f);

    auto [real, imag] = stft.compute(short_signal);
    ASSERT_EQ(real.size(), static_cast<size_t>(0));
}

UTEST(stft, istft_reconstructs_interior_of_signal) {
    // Round-trip test: Hann window + 75% overlap (the defaults) satisfies
    // the constant-overlap-add condition, so istft(stft(x)) should closely
    // match x -- EXCEPT within one fft_length of each edge, where the
    // analysis window necessarily tapers toward zero (a property of
    // windowed STFT in general, not a bug -- see stft.h's istft() docs).
    speech::dsp::stft stft(9);  // fft_length=512, slide_length=128
    int data_length = 512 + 128 * 19;  // 20 frames, comfortably long
    auto signal = make_sine(data_length, 440.0f, 16000);

    auto [real, imag] = stft.compute(signal);
    auto reconstructed = stft.istft(real, imag);

    ASSERT_EQ(reconstructed.size(), static_cast<size_t>(data_length));

    // Only check the interior, away from edge taper effects.
    int margin = stft.fft_length();
    double max_abs_err = 0.0;
    for (int i = margin; i < data_length - margin; ++i) {
        double err = std::fabs(reconstructed[i] - signal[i]);
        max_abs_err = std::max(max_abs_err, err);
    }
    ASSERT_TRUE(max_abs_err < 1e-2);
}

UTEST(stft, mismatched_frame_length_throws) {
    speech::dsp::stft stft(9);  // fft_length=512
    std::vector<std::vector<float>> bad_real = {std::vector<float>(256, 0.0f)};  // wrong length
    std::vector<std::vector<float>> bad_imag = {std::vector<float>(256, 0.0f)};

    bool threw = false;
    try {
        stft.istft(bad_real, bad_imag);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    ASSERT_TRUE(threw);
}

UTEST(stft, invalid_radix2_exp_throws) {
    bool threw = false;
    try {
        speech::dsp::stft stft(0);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    ASSERT_TRUE(threw);
}

