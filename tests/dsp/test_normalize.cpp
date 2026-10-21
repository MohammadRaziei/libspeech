#include "utest.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "libspeech/dsp/normalize.h"

namespace {

using speech::dsp::rms_normalize;
using speech::dsp::rms_normalize_result;
using speech::dsp::scale_to_rms;

constexpr double pi = 3.14159265358979323846;

std::vector<float> sine(std::size_t n, double amplitude, double cycles) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) {
        v[i] = static_cast<float>(amplitude * std::sin(2.0 * pi * cycles * static_cast<double>(i) / static_cast<double>(n)));
    }
    return v;
}

// Independent references: long double, written from the definitions, not sharing code with the implementation.
long double ref_rms(const std::vector<float>& v) {
    long double e = 0.0L;
    for (const float s : v) e += static_cast<long double>(s) * static_cast<long double>(s);
    return std::sqrt(e / static_cast<long double>(v.size()));
}
double ref_dbfs(const std::vector<float>& v) { return static_cast<double>(20.0L * std::log10(ref_rms(v))); }
double ref_peak(const std::vector<float>& v) {
    double p = 0.0;
    for (const float s : v) p = std::fmax(p, std::fabs(static_cast<double>(s)));
    return p;
}
}  // namespace

UTEST(normalize, rms_and_rms_dbfs_of_known_signals) {
    const std::vector<float> full_scale_sine = sine(48000, 1.0, 100);  // whole cycles: RMS = 1/sqrt(2)
    ASSERT_NEAR(speech::dsp::rms(full_scale_sine.data(), full_scale_sine.size()), 1.0 / std::sqrt(2.0), 1e-6);
    ASSERT_NEAR(speech::dsp::rms_dbfs(full_scale_sine.data(), full_scale_sine.size()), -3.0103, 1e-3);
    const std::vector<float> dc(100, 1.0f);  // constant full scale: 0 dBFS
    ASSERT_NEAR(speech::dsp::rms_dbfs(dc.data(), dc.size()), 0.0, 1e-9);
    ASSERT_EQ(speech::dsp::rms(nullptr, 0), 0.0);
    const std::vector<float> silence(100, 0.0f);
    ASSERT_TRUE(std::isinf(speech::dsp::rms_dbfs(silence.data(), silence.size())));
    ASSERT_LT(speech::dsp::rms_dbfs(silence.data(), silence.size()), 0.0);
}

UTEST(normalize, a_quiet_signal_is_raised_to_the_target_level) {
    std::vector<float> x = sine(16000, 0.01, 50);  // about -43 dBFS
    const rms_normalize_result r = rms_normalize(x.data(), x.size(), -20.0);
    ASSERT_NEAR(ref_dbfs(x), -20.0, 1e-3);
    ASSERT_NEAR(r.output_rms_dbfs, -20.0, 1e-3);
    ASSERT_NEAR(r.input_rms_dbfs, -3.0103 - 40.0, 1e-2);
    ASSERT_GT(r.gain, 1.0);
    ASSERT_FALSE(r.peak_limited);
}

UTEST(normalize, a_loud_signal_is_lowered_to_the_target_level) {
    std::vector<float> x = sine(16000, 4.0, 50);  // float audio far beyond +-1
    const rms_normalize_result r = rms_normalize(x.data(), x.size(), -18.0);
    ASSERT_NEAR(ref_dbfs(x), -18.0, 1e-3);
    ASSERT_LT(r.gain, 1.0);
    ASSERT_FALSE(r.peak_limited);
    ASSERT_LE(ref_peak(x), 0.99);
}

UTEST(normalize, every_sample_gets_the_same_gain_so_the_shape_is_unchanged) {
    const std::vector<float> original = sine(4000, 0.05, 7);
    std::vector<float> x = original;
    const rms_normalize_result r = rms_normalize(x.data(), x.size(), -20.0);
    for (std::size_t i = 0; i < x.size(); ++i) {
        ASSERT_NEAR(x[i], static_cast<double>(original[i]) * r.gain, 1e-6);
    }
}

UTEST(normalize, the_peak_ceiling_wins_over_the_rms_target) {
    // Spikes on a quiet floor: reaching -10 dBFS RMS would put the spikes far beyond full scale.
    std::vector<float> x(16000, 0.001f);
    for (std::size_t i = 0; i < x.size(); i += 1000) x[i] = 0.5f;
    const rms_normalize_result r = rms_normalize(x.data(), x.size(), -10.0);
    ASSERT_TRUE(r.peak_limited);
    ASSERT_NEAR(ref_peak(x), 0.99, 1e-6);   // held at the ceiling ...
    ASSERT_LT(ref_dbfs(x), -10.0);          // ... so the level stays below the target
    ASSERT_NEAR(r.output_rms_dbfs, ref_dbfs(x), 1e-3);  // and the result reports what really happened
}

UTEST(normalize, the_peak_ceiling_is_a_parameter) {
    std::vector<float> x = sine(16000, 0.3, 50);
    const rms_normalize_result r = rms_normalize(x.data(), x.size(), -3.0, 0.5);  // -3 dBFS needs peak 0.7
    ASSERT_TRUE(r.peak_limited);
    ASSERT_NEAR(ref_peak(x), 0.5, 1e-6);
}

UTEST(normalize, the_reported_levels_match_the_measured_ones) {
    std::vector<float> x = sine(12345, 0.2, 31);
    const double before = ref_dbfs(x);
    const rms_normalize_result r = rms_normalize(x.data(), x.size(), -25.0);
    ASSERT_NEAR(r.input_rms_dbfs, before, 1e-4);
    ASSERT_NEAR(r.output_rms_dbfs, ref_dbfs(x), 1e-4);
    ASSERT_NEAR(20.0 * std::log10(r.gain), r.output_rms_dbfs - r.input_rms_dbfs, 1e-9);
}

UTEST(normalize, normalizing_twice_changes_nothing_more) {
    std::vector<float> x = sine(16000, 0.07, 9);
    rms_normalize(x.data(), x.size(), -20.0);
    const std::vector<float> once = x;
    const rms_normalize_result r = rms_normalize(x.data(), x.size(), -20.0);
    ASSERT_NEAR(r.gain, 1.0, 1e-5);
    for (std::size_t i = 0; i < x.size(); ++i) ASSERT_NEAR(x[i], once[i], 1e-6);
}

UTEST(normalize, silence_is_left_alone) {
    std::vector<float> x(1000, 0.0f);
    const rms_normalize_result r = rms_normalize(x.data(), x.size(), -20.0);
    ASSERT_EQ(r.gain, 1.0);
    ASSERT_FALSE(r.peak_limited);
    ASSERT_TRUE(std::isinf(r.input_rms_dbfs));
    for (const float s : x) ASSERT_EQ(s, 0.0f);
}

UTEST(normalize, a_non_finite_sample_leaves_the_signal_alone) {
    for (const float bad : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()}) {
        std::vector<float> x = sine(100, 0.1, 3);
        x[50] = bad;
        const std::vector<float> before = x;
        const rms_normalize_result r = rms_normalize(x.data(), x.size(), -20.0);
        ASSERT_EQ(r.gain, 1.0);
        for (std::size_t i = 0; i < x.size(); ++i) {
            if (std::isnan(before[i])) {
                ASSERT_TRUE(std::isnan(x[i]));
            } else {
                ASSERT_EQ(x[i], before[i]);
            }
        }
    }
}

UTEST(normalize, an_empty_signal_is_fine) {
    const rms_normalize_result r = rms_normalize(nullptr, 0, -20.0);
    ASSERT_EQ(r.gain, 1.0);
}

UTEST(normalize, bad_arguments_throw) {
    std::vector<float> x(10, 0.1f);
    int threw = 0;
    const auto expect_throw = [&](auto&& call) {
        try {
            call();
        } catch (const std::invalid_argument&) {
            ++threw;
        }
    };
    expect_throw([&] { rms_normalize(nullptr, 10, -20.0); });
    expect_throw([&] { rms_normalize(x.data(), x.size(), std::numeric_limits<double>::quiet_NaN()); });
    expect_throw([&] { rms_normalize(x.data(), x.size(), -std::numeric_limits<double>::infinity()); });
    expect_throw([&] { rms_normalize(x.data(), x.size(), -20.0, 0.0); });
    expect_throw([&] { rms_normalize(x.data(), x.size(), -20.0, -1.0); });
    expect_throw([&] { rms_normalize(x.data(), x.size(), -20.0, std::numeric_limits<double>::infinity()); });
    expect_throw([&] { scale_to_rms(x.data(), x.size(), 0.0); });
    expect_throw([&] { scale_to_rms(x.data(), x.size(), -0.1); });
    ASSERT_EQ(threw, 8);
    for (const float s : x) ASSERT_EQ(s, 0.1f);  // a rejected call must not touch the data
}

UTEST(normalize, one_call_on_a_planar_block_keeps_the_balance_between_channels) {
    // channel 0 is 10x louder than channel 1; a joint gain keeps that ratio, per-channel gains would not.
    const std::vector<float> a = sine(8000, 0.2, 20), b = sine(8000, 0.02, 33);
    std::vector<float> block(a);
    block.insert(block.end(), b.begin(), b.end());
    rms_normalize(block.data(), block.size(), -20.0);
    const std::vector<float> ca(block.begin(), block.begin() + 8000), cb(block.begin() + 8000, block.end());
    ASSERT_NEAR(static_cast<double>(ref_rms(ca) / ref_rms(cb)), 10.0, 1e-3);
    ASSERT_NEAR(ref_dbfs(block), -20.0, 1e-3);
}

UTEST(normalize, scale_to_rms_takes_a_linear_target) {
    std::vector<float> x = sine(16000, 0.5, 40);
    scale_to_rms(x.data(), x.size(), 0.1);
    ASSERT_NEAR(static_cast<double>(ref_rms(x)), 0.1, 1e-6);
}
