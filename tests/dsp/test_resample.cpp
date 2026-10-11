#include "utest.h"

#include <cmath>
#include <vector>

#include "libspeech/dsp/resample.h"

namespace {

// Generates `num_samples` of a sine wave at `freq_hz`, sampled at `sample_rate`.
std::vector<float> make_sine(int sample_rate, float freq_hz, int num_samples) {
    std::vector<float> out(num_samples);
    for (int i = 0; i < num_samples; ++i) {
        out[i] = std::sin(2.0 * M_PI * freq_hz * i / sample_rate);
    }
    return out;
}

}  // namespace

UTEST(resample, output_length_matches_rate_ratio) {
    // 1 second of audio at 44100Hz downsampled to 16000Hz should yield ~16000 samples.
    speech::dsp::resample resampler(44100, 16000);
    std::vector<float> input = make_sine(44100, 440.0f, 44100);

    std::vector<float> output = resampler.process(input);

    // Allow a small tolerance for resampler filter latency/edge effects.
    ASSERT_NEAR(static_cast<double>(output.size()), 16000.0, 200.0);
}

UTEST(resample, upsampling_produces_more_samples) {
    speech::dsp::resample resampler(8000, 16000);
    std::vector<float> input = make_sine(8000, 440.0f, 8000);

    std::vector<float> output = resampler.process(input);

    ASSERT_NEAR(static_cast<double>(output.size()), 16000.0, 200.0);
}

// Regression test for a real AudioFlux gotcha: resampleObj_setSamplate()
// silently no-ops when source_rate == target_rate, leaving its internal ratio
// at a hardcoded default of 0.5 (i.e. it would silently drop half the
// samples). Our wrapper must bypass AudioFlux entirely in this case.
UTEST(resample, same_rate_is_exact_identity) {
    speech::dsp::resample resampler(16000, 16000);
    std::vector<float> input = make_sine(16000, 440.0f, 1000);

    std::vector<float> output = resampler.process(input);

    ASSERT_EQ(output.size(), input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        ASSERT_EQ(output[i], input[i]);
    }
}

UTEST(resample, empty_input_returns_empty_output) {
    speech::dsp::resample resampler(44100, 16000);
    std::vector<float> input;

    // Contract: empty in -> empty out, no exception. (AudioFlux's underlying
    // C call errors out on zero-length input, so we special-case it.)
    std::vector<float> output = resampler.process(input);
    ASSERT_EQ(output.size(), static_cast<size_t>(0));
}

UTEST(resample, preserves_signal_energy_roughly)  {
    // A resampled sine wave should retain roughly the same amplitude range,
    // i.e. resampling should not silently zero-out or blow up the signal.
    speech::dsp::resample resampler(44100, 16000);
    std::vector<float> input = make_sine(44100, 440.0f, 44100);

    std::vector<float> output = resampler.process(input);

    float max_abs = 0.0f;
    for (float v : output) {
        max_abs = std::max(max_abs, std::fabs(v));
    }
    ASSERT_TRUE(max_abs > 0.5f);
    ASSERT_TRUE(max_abs < 1.5f);
}

