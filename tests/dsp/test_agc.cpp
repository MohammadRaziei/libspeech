#include "utest.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "libspeech/dsp/agc.h"

namespace {
using speech::dsp::speech_agc;
using speech::dsp::speech_agc_params;

constexpr double pi = 3.14159265358979323846;
constexpr double fs = 16000.0;
constexpr std::size_t window = 512;

std::vector<float> tone(std::size_t n, double amplitude, double freq = 200.0) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = static_cast<float>(amplitude * std::sin(2.0 * pi * freq * static_cast<double>(i) / fs));
    return v;
}

std::vector<float> concat(const std::vector<float>& a, const std::vector<float>& b) {
    std::vector<float> v(a);
    v.insert(v.end(), b.begin(), b.end());
    return v;
}

double dbfs(const std::vector<float>& v, std::size_t begin, std::size_t end) {
    long double e = 0.0L;
    for (std::size_t i = begin; i < end; ++i) e += static_cast<long double>(v[i]) * static_cast<long double>(v[i]);
    return static_cast<double>(10.0L * std::log10(e / static_cast<long double>(end - begin)));
}

// Fast time constants so that a test does not need minutes of audio.
speech_agc_params fast() {
    speech_agc_params p;
    p.level_ms = 500.0;
    p.attack_ms = 20.0;
    p.release_ms = 100.0;
    p.warmup_ms = 100.0;
    return p;
}

// Weights for n samples: one per window, all the same.
std::vector<float> constant_weights(std::size_t n, float w) { return std::vector<float>((n + window - 1) / window, w); }
}  // namespace

UTEST(speech_agc, steady_speech_is_brought_to_the_target) {
    speech_agc agc(fs, fast());
    std::vector<float> x = tone(160000, 0.05);  // 10 s at -29 dBFS
    const std::vector<float> w = constant_weights(x.size(), 1.0f);
    agc.process(x.data(), x.size(), w.data(), w.size(), window);
    ASSERT_NEAR(dbfs(x, 128000, 160000), -20.0, 0.3);
    ASSERT_NEAR(agc.level_dbfs(), 10.0 * std::log10(0.05 * 0.05 / 2.0), 0.05);
    ASSERT_NEAR(agc.gain(), std::pow(10.0, (-20.0 - agc.level_dbfs()) / 20.0), 0.01 * agc.gain());
}

UTEST(speech_agc, follows_a_change_of_the_speech_level_in_both_directions) {
    speech_agc agc(fs, fast());
    std::vector<float> x = concat(tone(96000, 0.02), tone(96000, 0.2));  // 6 s quiet, then 6 s 20 dB louder
    const std::vector<float> w = constant_weights(x.size(), 1.0f);
    agc.process(x.data(), x.size(), w.data(), w.size(), window);
    ASSERT_NEAR(dbfs(x, 80000, 96000), -20.0, 0.5);
    ASSERT_NEAR(dbfs(x, 176000, 192000), -20.0, 0.5);
}

UTEST(speech_agc, level_and_target_gain_stay_frozen_while_the_weight_is_zero) {
    speech_agc agc(fs, fast());
    std::vector<float> speech = tone(48000, 0.05);
    const std::vector<float> w1 = constant_weights(speech.size(), 1.0f);
    agc.process(speech.data(), speech.size(), w1.data(), w1.size(), window);
    const double gain_before = agc.gain(), level_before = agc.level_dbfs();

    std::vector<float> noise = tone(48000, 0.002, 1234.0);  // a pause with a noise floor, weight 0
    const std::vector<float> original(noise);
    const std::vector<float> w0 = constant_weights(noise.size(), 0.0f);
    agc.process(noise.data(), noise.size(), w0.data(), w0.size(), window);
    ASSERT_EQ(agc.level_dbfs(), level_before);
    ASSERT_NEAR(agc.gain(), gain_before, 2e-3 * gain_before);  // only the ripple of the level estimate settles; the noise (-57 dBFS, +24 dB if followed) did not pull it up
    for (std::size_t i = 0; i < noise.size(); ++i) {
        ASSERT_NEAR(static_cast<double>(noise[i]), static_cast<double>(original[i]) * gain_before, 3e-3 * 0.002 * gain_before + 1e-9);
    }
}

UTEST(speech_agc, no_speech_at_all_and_the_warm_up_leave_the_signal_alone) {
    speech_agc agc(fs, fast());
    std::vector<float> x = tone(16000, 0.05);
    const std::vector<float> original(x);
    const std::vector<float> w0 = constant_weights(x.size(), 0.0f);
    agc.process(x.data(), x.size(), w0.data(), w0.size(), window);
    ASSERT_EQ(agc.gain(), 1.0);
    ASSERT_TRUE(std::isinf(agc.level_dbfs()) && agc.level_dbfs() < 0.0);
    for (std::size_t i = 0; i < x.size(); ++i) ASSERT_EQ(x[i], original[i]);

    // speech starts: the first 100 ms (the warm-up) are not touched, the gain moves afterwards
    speech_agc warm(fs, fast());
    std::vector<float> y = tone(16000, 0.05);
    const std::vector<float> w1 = constant_weights(y.size(), 1.0f);
    warm.process(y.data(), y.size(), w1.data(), w1.size(), window);
    for (std::size_t i = 0; i < 1500; ++i) ASSERT_EQ(y[i], original[i]);
    ASSERT_GT(warm.gain(), 1.5);
}

UTEST(speech_agc, no_output_sample_is_above_the_ceiling) {
    speech_agc agc(fs, fast());
    std::vector<float> x = tone(64000, 0.05);
    x[30000] = 5.0f;
    x[30001] = -4.0f;
    const std::vector<float> w = constant_weights(x.size(), 1.0f);
    agc.process(x.data(), x.size(), w.data(), w.size(), window);
    float peak = 0.0f;
    for (float v : x) peak = std::max(peak, std::fabs(v));
    ASSERT_LE(static_cast<double>(peak), 0.99);
    ASSERT_NEAR(static_cast<double>(x[30000]), 0.99, 1e-5);
    ASSERT_NEAR(static_cast<double>(x[30001]), -0.99, 1e-5);
}

UTEST(speech_agc, the_gain_stays_inside_its_limits) {
    speech_agc_params p = fast();
    speech_agc up(fs, p);
    std::vector<float> quiet = tone(160000, 1e-4);  // needs +60 dB, the limit is +24
    const std::vector<float> w = constant_weights(quiet.size(), 1.0f);
    up.process(quiet.data(), quiet.size(), w.data(), w.size(), window);
    ASSERT_NEAR(up.gain(), std::pow(10.0, 24.0 / 20.0), 1e-3 * up.gain());

    p.target_dbfs = -40.0;
    speech_agc down(fs, p);
    std::vector<float> loud = tone(160000, 0.9);  // needs -36 dB, the limit is -24
    down.process(loud.data(), loud.size(), w.data(), w.size(), window);
    ASSERT_NEAR(down.gain(), std::pow(10.0, -24.0 / 20.0), 1e-3 * down.gain());
}

UTEST(speech_agc, soft_weights_give_the_same_level_estimate_as_full_weights) {
    speech_agc full(fs, fast()), half(fs, fast());
    std::vector<float> a = tone(48000, 0.05), b = a;
    const std::vector<float> w1 = constant_weights(a.size(), 1.0f), w5 = constant_weights(b.size(), 0.5f);
    full.process(a.data(), a.size(), w1.data(), w1.size(), window);
    half.process(b.data(), b.size(), w5.data(), w5.size(), window);
    ASSERT_NEAR(full.level_dbfs(), half.level_dbfs(), 0.05);
}

UTEST(speech_agc, any_chunking_on_window_boundaries_gives_the_identical_result) {
    std::mt19937 rng(5);
    std::normal_distribution<float> noise(0.0f, 1.0f);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    const std::size_t n = window * 150;
    std::vector<float> x(n);
    std::vector<float> w(n / window);
    for (std::size_t i = 0; i < n; ++i) x[i] = 0.05f * noise(rng) * (1.0f + 3.0f * static_cast<float>(i) / static_cast<float>(n));
    for (float& v : w) v = uni(rng) < 0.3f ? 0.0f : uni(rng);

    speech_agc whole(fs, fast());
    std::vector<float> expected(x);
    whole.process(expected.data(), expected.size(), w.data(), w.size(), window);

    for (std::size_t windows_per_chunk : {1u, 3u, 7u, 64u}) {
        speech_agc chunked(fs, fast());
        std::vector<float> y(x);
        for (std::size_t k = 0; k < w.size(); k += windows_per_chunk) {
            const std::size_t count = std::min(windows_per_chunk, w.size() - k);
            chunked.process(y.data() + k * window, count * window, w.data() + k, count, window);
        }
        for (std::size_t i = 0; i < n; ++i) ASSERT_EQ(y[i], expected[i]);
        ASSERT_EQ(chunked.gain(), whole.gain());
    }
}

UTEST(speech_agc, reset_gives_a_new_instance) {
    speech_agc used(fs, fast()), fresh(fs, fast());
    std::vector<float> junk = tone(48000, 0.3);
    const std::vector<float> w = constant_weights(junk.size(), 1.0f);
    used.process(junk.data(), junk.size(), w.data(), w.size(), window);
    used.reset();
    ASSERT_EQ(used.gain(), 1.0);
    ASSERT_TRUE(std::isinf(used.level_dbfs()));

    std::vector<float> a = tone(32000, 0.04, 300.0), b = a;
    const std::vector<float> w2 = constant_weights(a.size(), 1.0f);
    used.process(a.data(), a.size(), w2.data(), w2.size(), window);
    fresh.process(b.data(), b.size(), w2.data(), w2.size(), window);
    for (std::size_t i = 0; i < a.size(); ++i) ASSERT_EQ(a[i], b[i]);
}

UTEST(speech_agc, bad_arguments_throw_and_a_rejected_call_changes_nothing) {
    int threw = 0;
    auto expect_throw = [&](auto&& call) {
        try {
            call();
        } catch (const std::invalid_argument&) {
            ++threw;
        }
    };
    expect_throw([] { speech_agc(0.0); });
    expect_throw([] { speech_agc(std::nan("")); });
    expect_throw([] {
        speech_agc_params p;
        p.level_ms = 0.0;
        speech_agc(fs, p);
    });
    expect_throw([] {
        speech_agc_params p;
        p.attack_ms = -1.0;
        speech_agc(fs, p);
    });
    expect_throw([] {
        speech_agc_params p;
        p.max_gain_db = -30.0;  // below min_gain_db
        speech_agc(fs, p);
    });
    expect_throw([] {
        speech_agc_params p;
        p.peak_ceiling = 0.0;
        speech_agc(fs, p);
    });
    expect_throw([] {
        speech_agc_params p;
        p.level_ms = 0.01;  // shorter than one sample at 16 kHz
        speech_agc(fs, p);
    });
    ASSERT_EQ(threw, 7);

    speech_agc agc(fs, fast());
    std::vector<float> x = tone(2048, 0.05);
    const std::vector<float> before(x);
    const std::vector<float> w(4, 1.0f), bad_w = {1.0f, 2.0f, 1.0f, 1.0f};
    int rejected = 0;
    auto expect_rejected = [&](auto&& call) {
        try {
            call();
        } catch (const std::invalid_argument&) {
            ++rejected;
        }
    };
    expect_rejected([&] { agc.process(x.data(), x.size(), w.data(), w.size(), 0); });
    expect_rejected([&] { agc.process(x.data(), x.size(), bad_w.data(), bad_w.size(), window); });
    expect_rejected([&] { agc.process(x.data(), x.size(), nullptr, 4, window); });
    expect_rejected([&] { agc.process(nullptr, 5, w.data(), w.size(), window); });
    std::vector<float> with_nan(x);
    with_nan[2000] = std::nanf("");
    expect_rejected([&] { agc.process(with_nan.data(), with_nan.size(), w.data(), w.size(), window); });
    ASSERT_EQ(rejected, 5);
    ASSERT_EQ(agc.gain(), 1.0);  // nothing was processed
    ASSERT_TRUE(std::isinf(agc.level_dbfs()));
    for (std::size_t i = 0; i < x.size(); ++i) ASSERT_EQ(x[i], before[i]);
}

UTEST(speech_agc, the_level_estimate_is_the_weighted_mean_of_the_speech_windows) {
    // DC windows make the mean square exact: 0.1 with weight 1 and 0.4 with weight 0.25, alternating, 20 windows. The weighted
    // mean square is (0.01 * 1 + 0.16 * 0.25) / 1.25 = 0.04 (-13.98 dBFS); the total weight (7680) is below level_ms, so the
    // estimate is still the exact running weighted mean.
    speech_agc agc(fs, fast());
    std::vector<float> x(window * 20);
    std::vector<float> w(20);
    for (std::size_t k = 0; k < 20; ++k) {
        w[k] = (k % 2 == 0) ? 1.0f : 0.25f;
        for (std::size_t i = 0; i < window; ++i) x[k * window + i] = (k % 2 == 0) ? 0.1f : 0.4f;
    }
    agc.process(x.data(), x.size(), w.data(), w.size(), window);
    ASSERT_NEAR(agc.level_dbfs(), 10.0 * std::log10(0.04), 0.01);
}

UTEST(speech_agc, the_gain_falls_with_the_attack_time_and_rises_with_the_release_time) {
    speech_agc_params p;
    p.level_ms = 10.0;   // the level follows quickly, so the gain dynamics are the ones of the smoothing
    p.attack_ms = 5.0;
    p.release_ms = 2000.0;
    p.warmup_ms = 0.0;

    speech_agc agc(fs, p);
    std::vector<float> loud = tone(32000, 0.2);  // 2 s: the gain settles near 0.707 (rms 0.1414 -> -20 dBFS)
    const std::vector<float> wloud = constant_weights(loud.size(), 1.0f);
    agc.process(loud.data(), loud.size(), wloud.data(), wloud.size(), window);
    ASSERT_NEAR(agc.gain(), 0.707, 0.02);

    // speech gets 20 dB quieter: the gain has to rise to ~7, slowly (2 s time constant): after 100 ms it is far from there
    std::vector<float> quiet = tone(1600, 0.02);
    const std::vector<float> wquiet = constant_weights(quiet.size(), 1.0f);
    agc.process(quiet.data(), quiet.size(), wquiet.data(), wquiet.size(), window);
    ASSERT_LT(agc.gain(), 1.5);

    // let it settle high, then the speech gets 20 dB louder: the gain must fall to ~0.7 within about 100 ms
    std::vector<float> settle = tone(96000, 0.02);
    const std::vector<float> ws = constant_weights(settle.size(), 1.0f);
    agc.process(settle.data(), settle.size(), ws.data(), ws.size(), window);
    ASSERT_GT(agc.gain(), 6.0);
    std::vector<float> loud_again = tone(1600, 0.2);
    const std::vector<float> wl = constant_weights(loud_again.size(), 1.0f);
    agc.process(loud_again.data(), loud_again.size(), wl.data(), wl.size(), window);
    ASSERT_LT(agc.gain(), 0.8);
}
