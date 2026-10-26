#include "utest.h"

#include <algorithm>
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

// --- speech-weighted level --------------------------------------------------------------------------------------

namespace {
using speech::dsp::rms_normalize_weighted;
using speech::dsp::speech_gate;
using speech::dsp::speech_weights;
using speech::dsp::weighted_rms;

// Long double reference of sqrt(sum w_k e_k / sum w_k len_k), written from the definition.
double ref_weighted_rms(const std::vector<float>& x, const std::vector<float>& w, std::size_t window) {
    long double e = 0.0L, c = 0.0L;
    for (std::size_t k = 0; k < w.size(); ++k) {
        for (std::size_t i = k * window; i < std::min(x.size(), (k + 1) * window); ++i) {
            e += static_cast<long double>(w[k]) * static_cast<long double>(x[i]) * static_cast<long double>(x[i]);
            c += static_cast<long double>(w[k]);
        }
    }
    return c > 0.0L ? static_cast<double>(std::sqrt(e / c)) : 0.0;
}
}  // namespace

UTEST(normalize, speech_weights_hard_is_a_step_at_the_threshold) {
    const std::vector<float> p = {0.0f, 0.2f, 0.4999f, 0.5f, 0.51f, 1.0f};
    std::vector<float> w(p.size());
    speech_weights(p.data(), p.size(), speech_gate::hard, 0.5, w.data());
    const std::vector<float> expect = {0, 0, 0, 1, 1, 1};
    for (std::size_t i = 0; i < p.size(); ++i) ASSERT_EQ(w[i], expect[i]);
}

UTEST(normalize, speech_weights_soft_is_a_ramp_from_the_threshold_to_one) {
    const std::vector<float> p = {0.0f, 0.4f, 0.5f, 0.6f, 0.75f, 1.0f};
    std::vector<float> w(p.size());
    speech_weights(p.data(), p.size(), speech_gate::soft, 0.5, w.data());
    const double expect[] = {0.0, 0.0, 0.0, 0.2, 0.5, 1.0};  // (p - 0.5) / 0.5, clamped
    for (std::size_t i = 0; i < p.size(); ++i) ASSERT_NEAR(static_cast<double>(w[i]), expect[i], 1e-6);
    speech_weights(p.data(), p.size(), speech_gate::soft, 0.2, w.data());  // (p - 0.2) / 0.8
    ASSERT_NEAR(static_cast<double>(w[3]), 0.5, 1e-6);
}

UTEST(normalize, speech_weights_bad_arguments_throw) {
    std::vector<float> p = {0.3f, 0.7f}, w(2);
    int threw = 0;
    auto expect_throw = [&](auto&& call) {
        try {
            call();
        } catch (const std::invalid_argument&) {
            ++threw;
        }
    };
    expect_throw([&] { speech_weights(p.data(), 2, speech_gate::hard, 0.0, w.data()); });
    expect_throw([&] { speech_weights(p.data(), 2, speech_gate::hard, 1.5, w.data()); });
    expect_throw([&] { speech_weights(p.data(), 2, speech_gate::soft, 1.0, w.data()); });  // ramp needs threshold < 1
    expect_throw([&] { speech_weights(p.data(), 2, speech_gate::hard, std::nan(""), w.data()); });
    const std::vector<float> bad = {0.3f, 1.2f};
    expect_throw([&] { speech_weights(bad.data(), 2, speech_gate::hard, 0.5, w.data()); });
    const std::vector<float> nan = {0.3f, std::nanf("")};
    expect_throw([&] { speech_weights(nan.data(), 2, speech_gate::soft, 0.5, w.data()); });
    expect_throw([&] { speech_weights(nullptr, 2, speech_gate::hard, 0.5, w.data()); });
    ASSERT_EQ(threw, 7);
    speech_weights(p.data(), 2, speech_gate::hard, 1.0, w.data());  // hard may use threshold 1 (only p == 1 counts)
    ASSERT_EQ(w[0] + w[1], 0.0f);
}

UTEST(normalize, weighted_rms_matches_the_definition_with_a_partial_last_window) {
    // 5 windows of 100 samples, the last one cut at 450 samples; different levels and weights per window.
    std::vector<float> x;
    const double amp[] = {0.01, 0.4, 0.02, 0.3, 0.05};
    for (std::size_t k = 0; k < 5; ++k) {
        const std::size_t len = (k == 4) ? 50 : 100;
        for (std::size_t i = 0; i < len; ++i) x.push_back(static_cast<float>(amp[k] * (i % 2 ? 1.0 : -1.0)));
    }
    const std::vector<float> w = {0.0f, 1.0f, 0.25f, 0.5f, 1.0f};
    ASSERT_NEAR(weighted_rms(x.data(), x.size(), w.data(), w.size(), 100), ref_weighted_rms(x, w, 100), 1e-9);
    // fewer weights than windows: samples after the last window do not count
    const std::vector<float> two = {0.0f, 1.0f};
    ASSERT_NEAR(weighted_rms(x.data(), x.size(), two.data(), two.size(), 100), 0.4, 1e-6);
    ASSERT_EQ(weighted_rms(x.data(), x.size(), nullptr, 0, 100), 0.0);
}

UTEST(normalize, weighted_normalize_hits_the_target_on_the_speech_part_only) {
    // 1 s of "speech" (a sine at 0.1) and 3 s of near-silence (0.001): the plain RMS is dominated by the silence.
    const std::size_t window = 512, windows = 125;
    std::vector<float> x(window * windows, 0.0f), w(windows, 0.0f);
    for (std::size_t i = 0; i < x.size(); ++i) {
        const bool speech = i < window * 31;
        x[i] = static_cast<float>((speech ? 0.1 : 0.001) * std::sin(2.0 * pi * 200.0 * static_cast<double>(i) / 16000.0));
    }
    for (std::size_t k = 0; k < 31; ++k) w[k] = 1.0f;

    std::vector<float> plain(x);
    rms_normalize(plain.data(), plain.size(), -20.0);
    std::vector<float> gated(x);
    const rms_normalize_result r = rms_normalize_weighted(gated.data(), gated.size(), w.data(), w.size(), window, -20.0);

    const std::vector<float> speech_part(gated.begin(), gated.begin() + static_cast<long>(window * 31));
    ASSERT_NEAR(ref_dbfs(speech_part), -20.0, 1e-3);          // the speech reads the target
    ASSERT_NEAR(r.output_rms_dbfs, -20.0, 1e-3);
    const std::vector<float> plain_speech(plain.begin(), plain.begin() + static_cast<long>(window * 31));
    ASSERT_GT(ref_dbfs(plain_speech), -20.0 + 6.0);           // the plain gain made the speech much too loud
    ASSERT_FALSE(r.peak_limited);
    const double ratio = static_cast<double>(gated[100]) / static_cast<double>(x[100]);
    ASSERT_NEAR(static_cast<double>(gated[x.size() - 7]) / static_cast<double>(x[x.size() - 7]), ratio, 1e-4 * ratio);  // one gain
}

UTEST(normalize, weighted_normalize_with_no_weight_or_silence_leaves_the_signal_alone) {
    std::vector<float> x = sine(2048, 0.1, 10);
    const std::vector<float> before = x;
    const std::vector<float> none(4, 0.0f);
    const rms_normalize_result r = rms_normalize_weighted(x.data(), x.size(), none.data(), none.size(), 512);
    ASSERT_EQ(r.gain, 1.0);
    ASSERT_TRUE(std::isinf(r.input_rms_dbfs) && r.input_rms_dbfs < 0.0);
    for (std::size_t i = 0; i < x.size(); ++i) ASSERT_EQ(x[i], before[i]);

    std::vector<float> nan_x = before;
    nan_x[3] = std::nanf("");
    const std::vector<float> all(4, 1.0f);
    ASSERT_EQ(rms_normalize_weighted(nan_x.data(), nan_x.size(), all.data(), all.size(), 512).gain, 1.0);
}

UTEST(normalize, weighted_normalize_peak_ceiling_still_wins) {
    std::vector<float> x(4096, 0.001f);
    x[2000] = 0.9f;  // a peak in a window with weight 0: it must still limit the gain
    std::vector<float> w(8, 0.0f);
    w[0] = 1.0f;
    const rms_normalize_result r = rms_normalize_weighted(x.data(), x.size(), w.data(), w.size(), 512, -20.0, 0.99);
    ASSERT_TRUE(r.peak_limited);
    ASSERT_NEAR(static_cast<double>(x[2000]), 0.99, 1e-6);
}

UTEST(normalize, weighted_bad_arguments_throw) {
    std::vector<float> x = sine(1024, 0.1, 5);
    const std::vector<float> w = {1.0f, 1.0f};
    int threw = 0;
    auto expect_throw = [&](auto&& call) {
        try {
            call();
        } catch (const std::invalid_argument&) {
            ++threw;
        }
    };
    expect_throw([&] { rms_normalize_weighted(x.data(), x.size(), w.data(), 2, 0); });                       // window 0
    const std::vector<float> bad = {1.0f, 2.0f};
    expect_throw([&] { rms_normalize_weighted(x.data(), x.size(), bad.data(), 2, 512); });                   // weight > 1
    expect_throw([&] { rms_normalize_weighted(x.data(), x.size(), nullptr, 2, 512); });                      // null weights
    expect_throw([&] { rms_normalize_weighted(nullptr, 5, w.data(), 2, 512); });                             // null data
    expect_throw([&] { rms_normalize_weighted(x.data(), x.size(), w.data(), 2, 512, std::nan("")); });       // bad target
    expect_throw([&] { rms_normalize_weighted(x.data(), x.size(), w.data(), 2, 512, -20.0, 0.0); });         // bad ceiling
    ASSERT_EQ(threw, 6);
    const std::vector<float> same = sine(1024, 0.1, 5);
    for (std::size_t i = 0; i < x.size(); ++i) ASSERT_EQ(x[i], same[i]);  // a rejected call must not touch the data
}
