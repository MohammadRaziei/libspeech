#include "utest.h"

#include <cmath>
#include <stdexcept>
#include <vector>

#include "libspeech/dsp/fft.h"

namespace {
std::vector<float> makeSine(int n, float cyclesOverN) {
    std::vector<float> out(n);
    for (int i = 0; i < n; ++i) {
        out[i] = std::sin(2.0f * static_cast<float>(M_PI) * cyclesOverN * i / n);
    }
    return out;
}
}  // namespace

UTEST(FFT, SizeMatchesRadix2Exp) {
    speech::dsp::FFT fft(8);  // 2^8 = 256
    ASSERT_EQ(fft.size(), 256);
}

UTEST(FFT, ImpulseHasFlatSpectrum) {
    // FFT of a unit impulse (delta function) is a constant (all-ones) spectrum.
    speech::dsp::FFT fft(6);  // length 64
    std::vector<float> real(64, 0.0f);
    real[0] = 1.0f;

    auto [outReal, outImag] = fft.forward(real);

    for (int i = 0; i < 64; ++i) {
        ASSERT_NEAR(static_cast<double>(outReal[i]), 1.0, 1e-4);
        ASSERT_NEAR(static_cast<double>(outImag[i]), 0.0, 1e-4);
    }
}

UTEST(FFT, SineWaveHasEnergyAtExpectedBin) {
    // A pure sine at k cycles over N samples should show up (almost)
    // entirely in FFT bin k (and its mirror at N-k).
    const int n = 64;
    const int k = 5;
    speech::dsp::FFT fft(6);  // length 64
    auto real = makeSine(n, static_cast<float>(k));

    auto [outReal, outImag] = fft.forward(real);

    float magAtK = std::sqrt(outReal[k] * outReal[k] + outImag[k] * outImag[k]);
    float totalEnergy = 0.0f;
    for (int i = 0; i < n; ++i) {
        totalEnergy += outReal[i] * outReal[i] + outImag[i] * outImag[i];
    }
    // Bin k (plus its mirror) should hold the overwhelming majority of the energy.
    ASSERT_TRUE((2.0f * magAtK * magAtK) / totalEnergy > 0.95f);
}

UTEST(FFT, ForwardInverseRoundTrip) {
    const int n = 64;
    speech::dsp::FFT fft(6);
    auto real = makeSine(n, 3.0f);
    std::vector<float> imag(n, 0.0f);

    auto [freqReal, freqImag] = fft.forward(real);
    auto [backReal, backImag] = fft.inverse(freqReal, freqImag);

    for (int i = 0; i < n; ++i) {
        ASSERT_NEAR(static_cast<double>(backReal[i]), static_cast<double>(real[i]), 1e-3);
        ASSERT_NEAR(static_cast<double>(backImag[i]), 0.0, 1e-3);
    }
}

UTEST(FFT, DctIdctRoundTrip) {
    const int n = 64;
    speech::dsp::FFT fft(6);
    auto signal = makeSine(n, 4.0f);

    auto coeffs = fft.dct(signal, true);
    auto reconstructed = fft.idct(coeffs, true);

    for (int i = 0; i < n; ++i) {
        ASSERT_NEAR(static_cast<double>(reconstructed[i]), static_cast<double>(signal[i]), 1e-2);
    }
}

UTEST(FFT, IdctDoesNotMutateInput) {
    // Regression test for audioflux_issues.md Issue 3: AudioFlux's
    // fftObj_idct() mutates its input array in place. Our wrapper must copy
    // internally so the caller's vector is untouched.
    const int n = 64;
    speech::dsp::FFT fft(6);
    auto signal = makeSine(n, 4.0f);
    auto coeffs = fft.dct(signal, true);
    std::vector<float> coeffsCopy = coeffs;

    fft.idct(coeffs, true);

    for (int i = 0; i < n; ++i) {
        ASSERT_EQ(coeffs[i], coeffsCopy[i]);
    }
}

UTEST(FFT, WrongLengthInputThrows) {
    speech::dsp::FFT fft(6);  // length 64
    std::vector<float> wrongSize(32, 0.0f);

    bool threw = false;
    try {
        fft.forward(wrongSize);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    ASSERT_TRUE(threw);
}

UTEST(FFT, InvalidRadix2ExpThrows) {
    bool threw = false;
    try {
        speech::dsp::FFT fft(0);  // out of [1,30] range
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    ASSERT_TRUE(threw);
}

UTEST_MAIN();
