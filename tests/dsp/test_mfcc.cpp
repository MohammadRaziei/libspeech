#include "utest.h"

#include <cmath>
#include <stdexcept>
#include <vector>

#include "libspeech/dsp/mfcc.h"

namespace {
std::vector<float> makeSine(int n, float freqHz, int sampleRate) {
    std::vector<float> out(n);
    for (int i = 0; i < n; ++i) {
        out[i] = std::sin(2.0 * M_PI * freqHz * i / sampleRate);
    }
    return out;
}
}  // namespace

UTEST(MFCC, OutputShapeMatchesParams) {
    speech::dsp::MFCC::Params params;
    params.sampleRate = 16000;
    params.numMelFilters = 26;
    params.numCoefficients = 13;
    params.radix2Exp = 9;  // fftLength = 512

    speech::dsp::MFCC mfcc(params);

    int dataLength = 512 + 128 * 9;  // 10 frames at default slideLength=fftLength/4
    auto signal = makeSine(dataLength, 440.0f, 16000);

    auto coeffs = mfcc.compute(signal);

    ASSERT_EQ(coeffs.size(), static_cast<size_t>(10));
    for (const auto& frame : coeffs) {
        ASSERT_EQ(frame.size(), static_cast<size_t>(13));
    }
}

UTEST(MFCC, OutputIsFiniteForRealSignal) {
    speech::dsp::MFCC::Params params;
    params.sampleRate = 16000;
    speech::dsp::MFCC mfcc(params);

    int dataLength = 1024 + 256 * 9;
    auto signal = makeSine(dataLength, 220.0f, 16000);
    auto coeffs = mfcc.compute(signal);

    for (const auto& frame : coeffs) {
        for (float v : frame) {
            ASSERT_TRUE(std::isfinite(v));
        }
    }
}

UTEST(MFCC, OutputIsFiniteForSilence) {
    // Regression-style test: log(0 + epsilon) must not produce NaN/Inf.
    speech::dsp::MFCC::Params params;
    params.sampleRate = 16000;
    speech::dsp::MFCC mfcc(params);

    int dataLength = 1024 + 256 * 5;
    std::vector<float> silence(dataLength, 0.0f);
    auto coeffs = mfcc.compute(silence);

    for (const auto& frame : coeffs) {
        for (float v : frame) {
            ASSERT_TRUE(std::isfinite(v));
        }
    }
}

UTEST(MFCC, StationarySignalGivesConsistentCoefficientsAcrossFrames) {
    // For a sinusoid at a frequency EXACTLY aligned to an FFT bin (k *
    // sampleRate / fftLength), shifting the analysis window only rotates
    // the phase of that bin's DFT coefficient -- magnitude is provably
    // invariant to the shift. That makes log-mel-energy (and therefore the
    // MFCC output) essentially identical across interior frames, which is a
    // real, backend-agnostic correctness signal (unlike an arbitrary
    // frequency, where windowed-FFT magnitude legitimately varies frame to
    // frame due to spectral leakage -- that's expected DSP behavior, not a
    // bug, so it would make a bad test).
    speech::dsp::MFCC::Params params;
    params.sampleRate = 16000;
    params.radix2Exp = 9;  // fftLength = 512, bin spacing = 16000/512 = 31.25 Hz
    speech::dsp::MFCC mfcc(params);

    const float binAlignedFreq = 14.0f * 16000.0f / 512.0f;  // = 437.5 Hz, exactly bin 14
    int dataLength = 512 + 128 * 19;  // 20 frames
    auto signal = makeSine(dataLength, binAlignedFreq, 16000);
    auto coeffs = mfcc.compute(signal);

    // Compare frame 5 to frame 15 (both well away from edge-taper effects).
    const auto& a = coeffs[5];
    const auto& b = coeffs[15];
    for (size_t i = 0; i < a.size(); ++i) {
        ASSERT_NEAR(static_cast<double>(a[i]), static_cast<double>(b[i]), 0.05);
    }
}

UTEST(MFCC, TooFewMelFiltersThrows) {
    speech::dsp::MFCC::Params params;
    params.sampleRate = 16000;
    params.numMelFilters = 1;

    bool threw = false;
    try {
        speech::dsp::MFCC mfcc(params);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    ASSERT_TRUE(threw);
}

UTEST(MFCC, TooManyCoefficientsThrows) {
    speech::dsp::MFCC::Params params;
    params.sampleRate = 16000;
    params.numMelFilters = 13;
    params.numCoefficients = 20;  // more than numMelFilters

    bool threw = false;
    try {
        speech::dsp::MFCC mfcc(params);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    ASSERT_TRUE(threw);
}

UTEST_MAIN();
