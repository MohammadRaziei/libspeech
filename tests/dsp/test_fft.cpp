#include "utest.h"

#include <cmath>
#include <stdexcept>
#include <vector>

#include "libspeech/dsp/fft.h"

namespace {
std::vector<float> make_sine(int n, float cycles_over_n) {
    std::vector<float> out(n);
    for (int i = 0; i < n; ++i) {
        out[i] = std::sin(2.0 * M_PI * cycles_over_n * i / n);
    }
    return out;
}
}  // namespace

UTEST(fft, size_matches_radix2_exp) {
    speech::dsp::fft fft(8);  // 2^8 = 256
    ASSERT_EQ(fft.size(), 256);
}

UTEST(fft, impulse_has_flat_spectrum) {
    // FFT of a unit impulse (delta function) is a constant (all-ones) spectrum.
    speech::dsp::fft fft(6);  // length 64
    std::vector<float> real(64, 0.0f);
    real[0] = 1.0f;

    auto [out_real, out_imag] = fft.forward(real);

    for (int i = 0; i < 64; ++i) {
        ASSERT_NEAR(static_cast<double>(out_real[i]), 1.0, 1e-4);
        ASSERT_NEAR(static_cast<double>(out_imag[i]), 0.0, 1e-4);
    }
}

UTEST(fft, sine_wave_has_energy_at_expected_bin) {
    // A pure sine at k cycles over N samples should show up (almost)
    // entirely in FFT bin k (and its mirror at N-k).
    const int n = 64;
    const int k = 5;
    speech::dsp::fft fft(6);  // length 64
    auto real = make_sine(n, static_cast<float>(k));

    auto [out_real, out_imag] = fft.forward(real);

    float mag_at_k = std::sqrt(out_real[k] * out_real[k] + out_imag[k] * out_imag[k]);
    float total_energy = 0.0f;
    for (int i = 0; i < n; ++i) {
        total_energy += out_real[i] * out_real[i] + out_imag[i] * out_imag[i];
    }
    // Bin k (plus its mirror) should hold the overwhelming majority of the energy.
    ASSERT_TRUE((2.0f * mag_at_k * mag_at_k) / total_energy > 0.95f);
}

UTEST(fft, forward_inverse_round_trip) {
    const int n = 64;
    speech::dsp::fft fft(6);
    auto real = make_sine(n, 3.0f);
    std::vector<float> imag(n, 0.0f);

    auto [freq_real, freq_imag] = fft.forward(real);
    auto [back_real, back_imag] = fft.inverse(freq_real, freq_imag);

    for (int i = 0; i < n; ++i) {
        ASSERT_NEAR(static_cast<double>(back_real[i]), static_cast<double>(real[i]), 1e-3);
        ASSERT_NEAR(static_cast<double>(back_imag[i]), 0.0, 1e-3);
    }
}

UTEST(fft, dct_idct_round_trip) {
    const int n = 64;
    speech::dsp::fft fft(6);
    auto signal = make_sine(n, 4.0f);

    auto coeffs = fft.dct(signal, true);
    auto reconstructed = fft.idct(coeffs, true);

    for (int i = 0; i < n; ++i) {
        ASSERT_NEAR(static_cast<double>(reconstructed[i]), static_cast<double>(signal[i]), 1e-2);
    }
}

UTEST(fft, idct_does_not_mutate_input) {
    // Regression test for third_party/audioflux/ISSUES.md Issue 3: AudioFlux's
    // fftObj_idct() mutates its input array in place. Our wrapper must copy
    // internally so the caller's vector is untouched.
    const int n = 64;
    speech::dsp::fft fft(6);
    auto signal = make_sine(n, 4.0f);
    auto coeffs = fft.dct(signal, true);
    std::vector<float> coeffs_copy = coeffs;

    fft.idct(coeffs, true);

    for (int i = 0; i < n; ++i) {
        ASSERT_EQ(coeffs[i], coeffs_copy[i]);
    }
}

UTEST(fft, wrong_length_input_throws) {
    speech::dsp::fft fft(6);  // length 64
    std::vector<float> wrong_size(32, 0.0f);

    bool threw = false;
    try {
        fft.forward(wrong_size);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    ASSERT_TRUE(threw);
}

UTEST(fft, invalid_radix2_exp_throws) {
    bool threw = false;
    try {
        speech::dsp::fft fft(0);  // out of [1,30] range
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    ASSERT_TRUE(threw);
}

