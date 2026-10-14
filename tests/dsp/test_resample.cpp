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


namespace {

// Level (dB re the 0.5 input amplitude) of the tone that `freq_hz` becomes after resampling,
// estimated by projecting the middle of the output onto sin/cos at the folded frequency.
double resampled_tone_db(speech::dsp::resample_quality quality, double freq_hz) {
    const int sr_in = 44100, sr_out = 16000, n = sr_in * 2;
    std::vector<float> x(n);
    for (int i = 0; i < n; ++i) x[i] = 0.5f * static_cast<float>(std::sin(2.0 * M_PI * freq_hz * i / sr_in + 0.3));
    speech::dsp::resample rs(sr_in, sr_out, quality);
    const std::vector<float> y = rs.process(x);
    const double folded = std::fabs(freq_hz - sr_out * std::round(freq_hz / sr_out));
    double s = 0.0, c = 0.0;
    long count = 0;
    for (size_t i = 2000; i + 2000 < y.size(); ++i, ++count) {
        const double a = 2.0 * M_PI * folded * static_cast<double>(i) / sr_out;
        s += y[i] * std::sin(a);
        c += y[i] * std::cos(a);
    }
    const double amp = 2.0 * std::hypot(s, c) / static_cast<double>(count);
    return 20.0 * std::log10(std::max(amp, 1e-12) / 0.5);
}

}  // namespace

UTEST(resample, every_quality_level_keeps_in_band_amplitude_and_rejects_aliases) {
    using q = speech::dsp::resample_quality;
    struct level {
        q quality;
        double max_alias_db;     // 12 kHz tone (aliases to 4 kHz)
        double max_loss_7k_db;   // passband loss at 7 kHz
    };
    // Measured for 44.1 -> 16 kHz (alias dB / loss at 7 kHz dB): best -153 / 0.00, high -135 / 0.00,
    // medium -124 / 0.07, fast -107 / 2.3. Bounds sit between neighbouring levels so a preset
    // swapped for another one is caught.
    const level levels[] = {
        {q::best, -140.0, 0.05}, {q::high, -125.0, 0.05}, {q::medium, -115.0, 0.3}, {q::fast, -95.0, 3.5}};
    for (const level& l : levels) {
        ASSERT_TRUE(std::fabs(resampled_tone_db(l.quality, 3000.0)) < 0.05);
        ASSERT_TRUE(std::fabs(resampled_tone_db(l.quality, 7000.0)) < l.max_loss_7k_db);
        ASSERT_TRUE(resampled_tone_db(l.quality, 12000.0) < l.max_alias_db);
    }
}

UTEST(resample, default_constructor_is_the_best_level) {
    std::vector<float> x = make_sine(44100, 1000.0f, 44100);
    speech::dsp::resample a(44100, 16000);
    speech::dsp::resample b(44100, 16000, speech::dsp::resample_quality::best);
    const std::vector<float> ya = a.process(x), yb = b.process(x);
    ASSERT_EQ(ya.size(), yb.size());
    for (size_t i = 0; i < ya.size(); ++i) ASSERT_TRUE(ya[i] == yb[i]);
}
