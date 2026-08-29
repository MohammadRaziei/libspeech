#include "utest.h"

#include <cmath>
#include <vector>

#include "libspeech/dsp/resample.h"

namespace {

// Generates `numSamples` of a sine wave at `freqHz`, sampled at `sampleRate`.
std::vector<float> makeSine(int sampleRate, float freqHz, int numSamples) {
    std::vector<float> out(numSamples);
    for (int i = 0; i < numSamples; ++i) {
        out[i] = std::sin(2.0 * M_PI * freqHz * i / sampleRate);
    }
    return out;
}

}  // namespace

UTEST(Resample, OutputLengthMatchesRateRatio) {
    // 1 second of audio at 44100Hz downsampled to 16000Hz should yield ~16000 samples.
    speech::dsp::Resample resampler(44100, 16000);
    std::vector<float> input = makeSine(44100, 440.0f, 44100);

    std::vector<float> output = resampler.resample(input);

    // Allow a small tolerance for resampler filter latency/edge effects.
    ASSERT_NEAR(static_cast<double>(output.size()), 16000.0, 200.0);
}

UTEST(Resample, UpsamplingProducesMoreSamples) {
    speech::dsp::Resample resampler(8000, 16000);
    std::vector<float> input = makeSine(8000, 440.0f, 8000);

    std::vector<float> output = resampler.resample(input);

    ASSERT_NEAR(static_cast<double>(output.size()), 16000.0, 200.0);
}

// Regression test for a real AudioFlux gotcha: resampleObj_setSamplate()
// silently no-ops when sourceRate == targetRate, leaving its internal ratio
// at a hardcoded default of 0.5 (i.e. it would silently drop half the
// samples). Our wrapper must bypass AudioFlux entirely in this case.
UTEST(Resample, SameRateIsExactIdentity) {
    speech::dsp::Resample resampler(16000, 16000);
    std::vector<float> input = makeSine(16000, 440.0f, 1000);

    std::vector<float> output = resampler.resample(input);

    ASSERT_EQ(output.size(), input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        ASSERT_EQ(output[i], input[i]);
    }
}

UTEST(Resample, EmptyInputReturnsEmptyOutput) {
    speech::dsp::Resample resampler(44100, 16000);
    std::vector<float> input;

    // Contract: empty in -> empty out, no exception. (AudioFlux's underlying
    // C call errors out on zero-length input, so we special-case it.)
    std::vector<float> output = resampler.resample(input);
    ASSERT_EQ(output.size(), static_cast<size_t>(0));
}

UTEST(Resample, PreservesSignalEnergyRoughly)  {
    // A resampled sine wave should retain roughly the same amplitude range,
    // i.e. resampling should not silently zero-out or blow up the signal.
    speech::dsp::Resample resampler(44100, 16000);
    std::vector<float> input = makeSine(44100, 440.0f, 44100);

    std::vector<float> output = resampler.resample(input);

    float maxAbs = 0.0f;
    for (float v : output) {
        maxAbs = std::max(maxAbs, std::fabs(v));
    }
    ASSERT_TRUE(maxAbs > 0.5f);
    ASSERT_TRUE(maxAbs < 1.5f);
}

UTEST_MAIN();
