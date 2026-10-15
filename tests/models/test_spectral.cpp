#include "utest.h"

#include <cmath>
#include <vector>

#include "spectral.h"

using speech::models::detail::real_fft;
using speech::models::detail::reflect_index;
using speech::models::detail::sqrt_hann_window;
using speech::models::detail::stft_apply_istft;

namespace {
std::vector<float> test_signal(std::size_t n) {
    std::vector<float> x(n);
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = 0.5f * std::sin(0.05f * static_cast<float>(i)) + 0.3f * std::cos(0.31f * static_cast<float>(i) + 1.0f);
    }
    return x;
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float worst = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) worst = std::max(worst, std::fabs(a[i] - b[i]));
    return worst;
}
}  // namespace

UTEST(spectral, rfft_matches_a_direct_dft_for_any_size) {
    // powers of two, mixed radix (320 and 960 are the DPDFNet sizes), odd, and a prime
    for (const std::size_t n : {std::size_t{1}, std::size_t{2}, std::size_t{3}, std::size_t{64}, std::size_t{97},
                                std::size_t{320}, std::size_t{960}}) {
        const real_fft fft(n);
        const std::vector<float> x = test_signal(n);
        std::vector<float> re(fft.bins()), im(fft.bins());
        fft.forward(x.data(), re.data(), im.data());
        const double pi = std::acos(-1.0);
        for (std::size_t k = 0; k < fft.bins(); ++k) {
            double sr = 0.0, si = 0.0;
            for (std::size_t t = 0; t < n; ++t) {
                sr += x[t] * std::cos(2.0 * pi * double(k * t) / double(n));
                si -= x[t] * std::sin(2.0 * pi * double(k * t) / double(n));
            }
            ASSERT_NEAR(re[k], sr, 1e-3);
            ASSERT_NEAR(im[k], si, 1e-3);
        }
    }
}

UTEST(spectral, rfft_inverse_round_trips) {
    for (const std::size_t n : {std::size_t{2}, std::size_t{3}, std::size_t{97}, std::size_t{320}, std::size_t{512},
                                std::size_t{960}}) {
        const real_fft fft(n);
        const std::vector<float> x = test_signal(n);
        std::vector<float> re(fft.bins()), im(fft.bins()), y(n);
        fft.forward(x.data(), re.data(), im.data());
        fft.inverse(re.data(), im.data(), y.data());
        ASSERT_LT(max_abs_diff(x, y), 1e-5f);
    }
}

UTEST(spectral, rfft_rejects_size_zero) {
    bool threw = false;
    try {
        real_fft fft(0);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    ASSERT_TRUE(threw);
}

UTEST(spectral, reflect_index_mirrors_without_repeating_the_edge_sample) {
    // torch pad_mode="reflect" on [a b c d]: ... c b | a b c d | c b ...
    ASSERT_EQ(reflect_index(-1, 4), 1u);
    ASSERT_EQ(reflect_index(-2, 4), 2u);
    ASSERT_EQ(reflect_index(4, 4), 2u);
    ASSERT_EQ(reflect_index(5, 4), 1u);
    ASSERT_EQ(reflect_index(-3, 1), 0u);  // a single sample is its own mirror
}

UTEST(spectral, stft_apply_istft_with_identity_reproduces_the_signal) {
    const real_fft fft(512);
    const std::vector<float> window = sqrt_hann_window(512);
    // lengths: shorter than a hop, exactly a hop, a frame, not a multiple of the hop, longer
    for (const std::size_t n : {std::size_t{1}, std::size_t{100}, std::size_t{256}, std::size_t{512},
                                std::size_t{1000}, std::size_t{4096}, std::size_t{16001}}) {
        const std::vector<float> x = test_signal(n);
        const std::vector<float> y = stft_apply_istft(x.data(), n, fft, window, 256, [](float*, float*) {});
        ASSERT_EQ(y.size(), n);
        ASSERT_LT(max_abs_diff(x, y), 1e-4f);
    }
}

UTEST(spectral, stft_apply_istft_applies_the_per_frame_callback) {
    const real_fft fft(512);
    const std::vector<float> window = sqrt_hann_window(512);
    const std::vector<float> x = test_signal(4000);
    // Zeroing every spectrum must silence the output (proves the callback result is what is synthesised).
    const std::vector<float> y = stft_apply_istft(x.data(), x.size(), fft, window, 256, [&](float* re, float* im) {
        std::fill(re, re + fft.bins(), 0.0f);
        std::fill(im, im + fft.bins(), 0.0f);
    });
    for (const float v : y) ASSERT_NEAR(v, 0.0f, 1e-7);
    // Halving every spectrum halves the output (linearity).
    const std::vector<float> half = stft_apply_istft(x.data(), x.size(), fft, window, 256, [&](float* re, float* im) {
        for (std::size_t k = 0; k < fft.bins(); ++k) {
            re[k] *= 0.5f;
            im[k] *= 0.5f;
        }
    });
    for (std::size_t i = 0; i < x.size(); ++i) ASSERT_NEAR(half[i], 0.5f * x[i], 1e-4);
}

UTEST(spectral, vorbis_window_is_power_complementary_at_50_percent_overlap) {
    // the property the overlap-add reconstruction relies on: w[i]^2 + w[i + n/2]^2 == 1
    const auto w = speech::models::detail::vorbis_window(320);
    for (std::size_t i = 0; i < 160; ++i) ASSERT_NEAR(w[i] * w[i] + w[i + 160] * w[i + 160], 1.0f, 1e-5);
}

UTEST(spectral, stft_apply_istft_identity_with_the_dpdfnet_geometry) {
    const real_fft fft(320);
    const std::vector<float> window = speech::models::detail::vorbis_window(320);
    for (const std::size_t n : {std::size_t{1}, std::size_t{159}, std::size_t{160}, std::size_t{1000}, std::size_t{16001}}) {
        const std::vector<float> x = test_signal(n);
        const std::vector<float> y = stft_apply_istft(x.data(), n, fft, window, 160, [](float*, float*) {});
        ASSERT_EQ(y.size(), n);
        ASSERT_LT(max_abs_diff(x, y), 1e-4f);
    }
}

UTEST(spectral, shift_realigns_a_model_that_answers_late_and_keeps_the_tail) {
    // A "model" that returns, for frame t, the spectrum of frame t - 4 (what a causal network with a
    // 4-hop look-ahead effectively does). With shift = 4 hops the output must line up with the input
    // again -- including the last 4 hops, which only exist because of the extra flush frames.
    const std::size_t hop = 160, delay = 4;
    const real_fft fft(320);
    const std::vector<float> window = speech::models::detail::vorbis_window(320);
    const std::size_t n = 5000;
    const std::vector<float> x = test_signal(n);
    std::vector<std::vector<float>> past_re, past_im;
    const std::vector<float> y = stft_apply_istft(
        x.data(), n, fft, window, hop,
        [&](float* re, float* im) {
            past_re.emplace_back(re, re + fft.bins());
            past_im.emplace_back(im, im + fft.bins());
            if (past_re.size() > delay) {
                const auto& r = past_re[past_re.size() - 1 - delay];
                const auto& i = past_im[past_im.size() - 1 - delay];
                std::copy(r.begin(), r.end(), re);
                std::copy(i.begin(), i.end(), im);
            } else {
                std::fill(re, re + fft.bins(), 0.0f);
                std::fill(im, im + fft.bins(), 0.0f);
            }
        },
        delay * hop);
    ASSERT_EQ(y.size(), n);
    // Skip the first hops (the model had nothing to say yet); the rest -- tail included -- matches.
    for (std::size_t i = 2 * hop; i < n; ++i) ASSERT_NEAR(y[i], x[i], 2e-4);
}
