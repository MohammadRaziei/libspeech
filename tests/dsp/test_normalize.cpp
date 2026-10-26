#include "utest.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
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

// --- robust level (clip_percentile), soft limiter, weighted CMVN ---------------------------------------------------

namespace {
using speech::dsp::limit_peaks;
using speech::dsp::weighted_cmvn;

// Reference of the clipped weighted RMS, written from the definition with long double and a plain sort.
double ref_clipped_rms(const std::vector<float>& x, const std::vector<float>& w, std::size_t window, double p) {
    struct item {
        long double ms, mass;
    };
    std::vector<item> items;
    long double total = 0.0L;
    for (std::size_t k = 0; k < w.size(); ++k) {
        const std::size_t b = k * window, e = std::min(x.size(), (k + 1) * window);
        if (b >= x.size() || w[k] == 0.0f) continue;
        long double s = 0.0L;
        for (std::size_t i = b; i < e; ++i) s += static_cast<long double>(x[i]) * static_cast<long double>(x[i]);
        const long double len = static_cast<long double>(e - b);
        items.push_back({s / len, static_cast<long double>(w[k]) * len});
        total += static_cast<long double>(w[k]) * len;
    }
    if (items.empty()) return 0.0;
    std::sort(items.begin(), items.end(), [](const item& a, const item& b) { return a.ms < b.ms; });
    long double acc = 0.0L, limit = items.back().ms;
    for (const item& it : items) {
        acc += it.mass;
        if (acc >= total * static_cast<long double>(p) / 100.0L) {
            limit = it.ms;
            break;
        }
    }
    long double energy = 0.0L;
    for (const item& it : items) energy += it.mass * std::min(it.ms, limit);
    return static_cast<double>(std::sqrt(energy / total));
}
}  // namespace

UTEST(normalize, clip_percentile_clips_the_loud_windows_to_the_percentile) {
    // 9 windows at amplitude 0.1 and one at 1.0, weight 1: the mean square of the loud one is clipped from 1 to 0.01.
    std::vector<float> x(1000, 0.1f);
    for (std::size_t i = 900; i < 1000; ++i) x[i] = 1.0f;
    const std::vector<float> w(10, 1.0f);
    ASSERT_NEAR(weighted_rms(x.data(), x.size(), w.data(), w.size(), 100), std::sqrt(0.109), 1e-6);        // off (100)
    ASSERT_NEAR(weighted_rms(x.data(), x.size(), w.data(), w.size(), 100, 90.0), 0.1, 1e-6);              // 90th: clipped
    ASSERT_NEAR(weighted_rms(x.data(), x.size(), w.data(), w.size(), 100, 95.0), std::sqrt(0.109), 1e-6);  // 95th: the loud one is the 95th
}

UTEST(normalize, clip_percentile_matches_the_reference_on_random_windows_with_weights) {
    std::mt19937 rng(7);
    std::normal_distribution<float> noise(0.0f, 1.0f);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    const std::size_t window = 64, windows = 80;
    std::vector<float> x(window * windows - 17), w(windows);  // the last window is partial
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = 0.05f * noise(rng) * (1.0f + 5.0f * uni(rng) * uni(rng));
    for (float& v : w) v = uni(rng) < 0.25f ? 0.0f : uni(rng);
    for (double p : {30.0, 50.0, 80.0, 95.0, 100.0}) {
        ASSERT_NEAR(weighted_rms(x.data(), x.size(), w.data(), w.size(), window, p), ref_clipped_rms(x, w, window, p), 1e-9);
    }
    ASSERT_LT(weighted_rms(x.data(), x.size(), w.data(), w.size(), window, 70.0),
              weighted_rms(x.data(), x.size(), w.data(), w.size(), window, 100.0));  // clipping never raises the level
}

UTEST(normalize, clip_percentile_bad_values_throw) {
    std::vector<float> x(256, 0.1f);
    const std::vector<float> w = {1.0f, 1.0f};
    int threw = 0;
    for (double p : {0.0, -5.0, 100.5, std::nan("")}) {
        try {
            weighted_rms(x.data(), x.size(), w.data(), 2, 128, p);
        } catch (const std::invalid_argument&) {
            ++threw;
        }
    }
    ASSERT_EQ(threw, 4);
}

UTEST(normalize, limit_peaks_keeps_everything_under_the_ceiling) {
    std::mt19937 rng(3);
    std::normal_distribution<float> noise(0.0f, 0.2f);
    std::vector<float> x(20000);
    for (float& v : x) v = noise(rng);
    x[100] = 1.5f;
    x[5000] = -3.0f;
    x[5001] = 2.0f;
    x[19999] = 1.2f;  // last sample
    x[0] = -1.1f;     // first sample
    ASSERT_TRUE(limit_peaks(x.data(), x.size(), 0.99, 64));
    float peak = 0.0f;
    for (float v : x) peak = std::max(peak, std::fabs(v));
    ASSERT_LE(static_cast<double>(peak), 0.99);
    ASSERT_NEAR(static_cast<double>(std::fabs(x[5000])), 0.99, 1e-5);  // the loudest peak lands on the ceiling
}

UTEST(normalize, limit_peaks_leaves_a_quiet_signal_alone_and_only_changes_the_neighbourhood_of_a_peak) {
    std::vector<float> quiet = sine(4096, 0.5, 20);
    const std::vector<float> before = quiet;
    ASSERT_FALSE(limit_peaks(quiet.data(), quiet.size(), 0.99, 64));
    for (std::size_t i = 0; i < quiet.size(); ++i) ASSERT_EQ(quiet[i], before[i]);

    const std::size_t L = 40, at = 5000;
    std::vector<float> x = sine(10000, 0.2, 400);
    const std::vector<float> orig = x;
    x[at] = 2.0f;
    ASSERT_TRUE(limit_peaks(x.data(), x.size(), 0.99, L));
    for (std::size_t i = 0; i < x.size(); ++i) {
        if (i + L <= at || i >= at + L) {  // the gain is exactly 1 beyond L - 1 samples from the peak
            ASSERT_EQ(x[i], orig[i]);
        }
    }
    // look-ahead: the gain already falls before the peak, never rises above 1, and the shape does not flip sign
    ASSERT_LT(std::fabs(x[at - 5]), std::fabs(orig[at - 5]));
    for (std::size_t i = at - L; i < at + L; ++i) ASSERT_LE(std::fabs(static_cast<double>(x[i])), std::fabs(static_cast<double>(orig[i])) + 2.0 * 1e-7 + (i == at ? 2.0 : 0.0));
}

UTEST(normalize, limit_peaks_lookahead_zero_is_one_instant_gain_and_edge_cases) {
    std::vector<float> x(100, 0.1f);
    x[50] = 3.0f;
    ASSERT_TRUE(limit_peaks(x.data(), x.size(), 0.99, 0));
    ASSERT_NEAR(static_cast<double>(x[50]), 0.99, 1e-6);
    ASSERT_EQ(x[49], 0.1f);
    ASSERT_EQ(x[51], 0.1f);

    std::vector<float> tiny = {5.0f};  // lookahead longer than the signal
    ASSERT_TRUE(limit_peaks(tiny.data(), 1, 0.99, 1000));
    ASSERT_LE(static_cast<double>(tiny[0]), 0.99);
    ASSERT_FALSE(limit_peaks(nullptr, 0, 0.99, 8));

    int threw = 0;
    auto expect_throw = [&](auto&& call) {
        try {
            call();
        } catch (const std::invalid_argument&) {
            ++threw;
        }
    };
    std::vector<float> y = {0.1f, std::nanf(""), 0.2f};
    expect_throw([&] { limit_peaks(y.data(), 3, 0.99, 8); });
    expect_throw([&] { limit_peaks(y.data(), 1, 0.0, 8); });
    expect_throw([&] { limit_peaks(y.data(), 1, std::nan(""), 8); });
    expect_throw([&] { limit_peaks(nullptr, 4, 0.99, 8); });
    ASSERT_EQ(threw, 4);
}

UTEST(normalize, limiter_keeps_the_target_level_where_the_hard_ceiling_would_lower_it) {
    // quiet "speech" (sine 0.05) with one 0.9 spike: reaching -20 dBFS needs a gain that pushes the spike far over 0.99.
    const std::size_t window = 512, windows = 60, spike = 20000;
    std::vector<float> base = sine(window * windows, 0.05, 150);
    base[spike] = 0.9f;
    const std::vector<float> w(windows, 1.0f);

    std::vector<float> hard(base);
    const rms_normalize_result rh = rms_normalize_weighted(hard.data(), hard.size(), w.data(), w.size(), window, -20.0, 0.99);
    ASSERT_TRUE(rh.peak_limited);
    const std::vector<float> hard_part(hard.begin() + 2000, hard.begin() + 8000);
    ASSERT_LT(ref_dbfs(hard_part), -20.0 - 6.0);  // the one spike made the whole signal much quieter than the target

    std::vector<float> soft(base);
    const rms_normalize_result rs = rms_normalize_weighted(soft.data(), soft.size(), w.data(), w.size(), window, -20.0, 0.99, 100.0, 80);
    ASSERT_TRUE(rs.peak_limited);
    const std::vector<float> soft_part(soft.begin() + 2000, soft.begin() + 8000);
    ASSERT_NEAR(ref_dbfs(soft_part), -20.0, 0.2);  // away from the spike the speech reaches the target (the spike itself adds ~0.1 dB)
    float peak = 0.0f;
    for (float v : soft) peak = std::max(peak, std::fabs(v));
    ASSERT_LE(static_cast<double>(peak), 0.99);
}

UTEST(normalize, clip_percentile_in_the_normalizer_raises_the_gain_when_a_few_windows_are_loud) {
    std::vector<float> x = sine(512 * 40, 0.05, 100);
    for (std::size_t i = 512 * 3; i < 512 * 4; ++i) x[i] *= 8.0f;  // one loud window
    const std::vector<float> w(40, 1.0f);
    std::vector<float> a(x), b(x);
    const rms_normalize_result plain = rms_normalize_weighted(a.data(), a.size(), w.data(), w.size(), 512, -20.0, 10.0);
    const rms_normalize_result clipped = rms_normalize_weighted(b.data(), b.size(), w.data(), w.size(), 512, -20.0, 10.0, 90.0);
    ASSERT_GT(clipped.gain, plain.gain * 1.1);
    ASSERT_NEAR(plain.output_rms_dbfs, -20.0, 1e-6);  // plain: the whole-signal level is the target
}

UTEST(normalize, weighted_cmvn_statistics_come_from_the_speech_frames_only) {
    const std::size_t frames = 300, dim = 6;
    std::mt19937 rng(11);
    std::normal_distribution<float> noise(0.0f, 1.0f);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    std::vector<float> f(frames * dim), w(frames);
    for (std::size_t t = 0; t < frames; ++t) {
        w[t] = (t < 100) ? 0.0f : (uni(rng) < 0.3f ? 1.0f : uni(rng));
        for (std::size_t d = 0; d < dim; ++d) f[t * dim + d] = static_cast<float>(d + 1) * noise(rng) + 5.0f * static_cast<float>(d) + (t < 100 ? 40.0f : 0.0f);
    }
    std::vector<float> out(f);
    weighted_cmvn(out.data(), frames, dim, w.data());
    for (std::size_t d = 0; d < dim; ++d) {
        long double tw = 0.0L, m = 0.0L, v = 0.0L;
        for (std::size_t t = 0; t < frames; ++t) {
            tw += w[t];
            m += static_cast<long double>(w[t]) * out[t * dim + d];
        }
        m /= tw;
        for (std::size_t t = 0; t < frames; ++t) {
            const long double c = static_cast<long double>(out[t * dim + d]) - m;
            v += static_cast<long double>(w[t]) * c * c;
        }
        ASSERT_NEAR(static_cast<double>(m), 0.0, 1e-4);
        ASSERT_NEAR(static_cast<double>(v / tw), 1.0, 1e-4);
    }
    // the weight-0 frames were shifted too (they were offset by 40, so they are far from 0 afterwards)
    ASSERT_GT(std::fabs(out[0]), 5.0f);

    std::vector<float> mean_only(f);
    weighted_cmvn(mean_only.data(), frames, dim, w.data(), false);
    for (std::size_t i = 0; i < f.size(); ++i) {
        const std::size_t d = i % dim;
        // mean-only keeps the spread: out - (f - mean) is the same constant for every frame of a dimension
        ASSERT_NEAR(static_cast<double>(mean_only[i] - mean_only[d] - (f[i] - f[d])), 0.0, 1e-4);
    }
}

UTEST(normalize, weighted_cmvn_no_weight_empty_and_bad_arguments) {
    std::vector<float> f = {1, 2, 3, 4, 5, 6};
    const std::vector<float> before = f, none(3, 0.0f), all(3, 1.0f);
    weighted_cmvn(f.data(), 3, 2, none.data());
    for (std::size_t i = 0; i < f.size(); ++i) ASSERT_EQ(f[i], before[i]);
    weighted_cmvn(nullptr, 0, 4, nullptr);  // nothing to do

    int threw = 0;
    auto expect_throw = [&](auto&& call) {
        try {
            call();
        } catch (const std::invalid_argument&) {
            ++threw;
        }
    };
    expect_throw([&] { weighted_cmvn(f.data(), 3, 0, all.data()); });
    expect_throw([&] { weighted_cmvn(f.data(), 3, 2, all.data(), true, -1.0); });
    expect_throw([&] { weighted_cmvn(f.data(), 3, 2, nullptr); });
    const std::vector<float> bad = {1.0f, 2.0f, 0.0f};
    expect_throw([&] { weighted_cmvn(f.data(), 3, 2, bad.data()); });
    ASSERT_EQ(threw, 4);
    for (std::size_t i = 0; i < f.size(); ++i) ASSERT_EQ(f[i], before[i]);  // a rejected call must not touch the data
}

UTEST(normalize, limit_peaks_gain_changes_smoothly_and_recovers_after_the_peak) {
    // A constant 0.2 signal makes the output sample / 0.2 the gain itself. One spike of 2.0 needs a gain of 0.495.
    const std::size_t L = 50, at = 3000;
    std::vector<float> x(6000, 0.2f);
    x[at] = 2.0f;
    ASSERT_TRUE(limit_peaks(x.data(), x.size(), 0.99, L));
    std::vector<double> gain(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) gain[i] = static_cast<double>(x[i]) / (i == at ? 2.0 : static_cast<double>(0.2f));
    double min_gain = 1.0, max_step = 0.0;
    for (std::size_t i = 0; i < gain.size(); ++i) {
        min_gain = std::min(min_gain, gain[i]);
        if (i > 0) max_step = std::max(max_step, std::fabs(gain[i] - gain[i - 1]));
    }
    ASSERT_NEAR(min_gain, 0.99 / 2.0, 1e-6);
    ASSERT_LE(max_step, (1.0 - min_gain) / static_cast<double>(L) + 1e-6);  // a ramp of L samples, never a jump
    ASSERT_LT(gain[at - L + 1], 1.0 - 1e-6);                               // it starts L - 1 samples before the peak
    ASSERT_LT(gain[at + L - 1], 1.0 - 1e-6);                               // and is still releasing L - 1 samples after it
    ASSERT_NEAR(gain[at + L], 1.0, 1e-9);
    ASSERT_NEAR(gain[at - L], 1.0, 1e-9);
}
