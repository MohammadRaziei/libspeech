#include "utest.h"

#include <cmath>
#include <stdexcept>
#include <vector>

#include "libspeech/dsp/window.h"

using speech::dsp::window_type;

UTEST(window, hann_has_correct_length) {
    auto w = speech::dsp::window::generate(window_type::hann, 512);
    ASSERT_EQ(w.size(), static_cast<size_t>(512));
}

UTEST(window, hann_starts_near_zero) {
    // Periodic Hann window: w[0] should be 0 (or very close to it).
    auto w = speech::dsp::window::generate(window_type::hann, 512);
    ASSERT_TRUE(w[0] >= 0.0f && w[0] < 0.01f);
}

UTEST(window, hann_peaks_near_middle) {
    auto w = speech::dsp::window::generate(window_type::hann, 512);
    float max_val = 0.0f;
    size_t max_idx = 0;
    for (size_t i = 0; i < w.size(); ++i) {
        if (w[i] > max_val) {
            max_val = w[i];
            max_idx = i;
        }
    }
    // Peak should be close to 1.0 and near the center of the window.
    ASSERT_TRUE(max_val > 0.99f);
    ASSERT_TRUE(max_idx > 240 && max_idx < 272);
}

UTEST(window, rect_is_all_ones) {
    auto w = speech::dsp::window::generate(window_type::rect, 100);
    ASSERT_EQ(w.size(), static_cast<size_t>(100));
    for (float v : w) {
        ASSERT_EQ(v, 1.0f);
    }
}

UTEST(window, hamming_differs_from_hann) {
    // Sanity check that different window types actually produce different
    // coefficients (i.e. the type parameter is really being respected).
    auto hann = speech::dsp::window::generate(window_type::hann, 64);
    auto hamm = speech::dsp::window::generate(window_type::hamming, 64);
    bool any_different = false;
    for (size_t i = 0; i < hann.size(); ++i) {
        if (std::fabs(hann[i] - hamm[i]) > 1e-4f) {
            any_different = true;
            break;
        }
    }
    ASSERT_TRUE(any_different);
}

UTEST(window, zero_or_negative_length_throws) {
    bool threw = false;
    try {
        speech::dsp::window::generate(window_type::hann, 0);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    ASSERT_TRUE(threw);
}

