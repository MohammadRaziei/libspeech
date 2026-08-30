#include "utest.h"

#include <cmath>
#include <stdexcept>
#include <vector>

#include "libspeech/dsp/stft.h"

namespace {
std::vector<float> makeSine(int n, float freqHz, int sampleRate) {
    std::vector<float> out(n);
    for (int i = 0; i < n; ++i) {
        out[i] = std::sin(2.0 * M_PI * freqHz * i / sampleRate);
    }
    return out;
}
}  // namespace

UTEST(STFT, FftLengthAndSlideLengthDefaults) {
    speech::dsp::STFT stft(9);  // fftLength = 512
    ASSERT_EQ(stft.fftLength(), 512);
    ASSERT_EQ(stft.slideLength(), 128);  // default: fftLength/4
}

UTEST(STFT, CalTimeLengthMatchesFrameCount) {
    speech::dsp::STFT stft(9);  // fftLength=512, slideLength=128
    // dataLength chosen so (dataLength - fftLength) is an exact multiple of slideLength.
    int dataLength = 512 + 128 * 9;  // 10 frames
    ASSERT_EQ(stft.calTimeLength(dataLength), 10);
}

UTEST(STFT, StftProducesExpectedFrameCountAndShape) {
    speech::dsp::STFT stft(9);
    int dataLength = 512 + 128 * 9;
    auto signal = makeSine(dataLength, 440.0f, 16000);

    auto [real, imag] = stft.stft(signal);

    ASSERT_EQ(real.size(), static_cast<size_t>(10));
    ASSERT_EQ(imag.size(), static_cast<size_t>(10));
    for (const auto& frame : real) {
        ASSERT_EQ(frame.size(), static_cast<size_t>(512));
    }
}

UTEST(STFT, TooShortInputProducesNoFrames) {
    speech::dsp::STFT stft(9);  // fftLength=512
    std::vector<float> shortSignal(100, 0.0f);

    auto [real, imag] = stft.stft(shortSignal);
    ASSERT_EQ(real.size(), static_cast<size_t>(0));
}

UTEST(STFT, IstftReconstructsInteriorOfSignal) {
    // Round-trip test: Hann window + 75% overlap (the defaults) satisfies
    // the constant-overlap-add condition, so istft(stft(x)) should closely
    // match x -- EXCEPT within one fftLength of each edge, where the
    // analysis window necessarily tapers toward zero (a property of
    // windowed STFT in general, not a bug -- see stft.h's istft() docs).
    speech::dsp::STFT stft(9);  // fftLength=512, slideLength=128
    int dataLength = 512 + 128 * 19;  // 20 frames, comfortably long
    auto signal = makeSine(dataLength, 440.0f, 16000);

    auto [real, imag] = stft.stft(signal);
    auto reconstructed = stft.istft(real, imag);

    ASSERT_EQ(reconstructed.size(), static_cast<size_t>(dataLength));

    // Only check the interior, away from edge taper effects.
    int margin = stft.fftLength();
    double maxAbsErr = 0.0;
    for (int i = margin; i < dataLength - margin; ++i) {
        double err = std::fabs(reconstructed[i] - signal[i]);
        maxAbsErr = std::max(maxAbsErr, err);
    }
    ASSERT_TRUE(maxAbsErr < 1e-2);
}

UTEST(STFT, MismatchedFrameLengthThrows) {
    speech::dsp::STFT stft(9);  // fftLength=512
    std::vector<std::vector<float>> badReal = {std::vector<float>(256, 0.0f)};  // wrong length
    std::vector<std::vector<float>> badImag = {std::vector<float>(256, 0.0f)};

    bool threw = false;
    try {
        stft.istft(badReal, badImag);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    ASSERT_TRUE(threw);
}

UTEST(STFT, InvalidRadix2ExpThrows) {
    bool threw = false;
    try {
        speech::dsp::STFT stft(0);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    ASSERT_TRUE(threw);
}

