#include "utest.h"

#include <cmath>
#include <stdexcept>
#include <vector>

#include "libspeech/dsp/window.h"

UTEST(Window, HannHasCorrectLength) {
    auto w = speech::dsp::window::generate(Window_Hann, 512);
    ASSERT_EQ(w.size(), static_cast<size_t>(512));
}

UTEST(Window, HannStartsNearZero) {
    // Periodic Hann window: w[0] should be 0 (or very close to it).
    auto w = speech::dsp::window::generate(Window_Hann, 512);
    ASSERT_TRUE(w[0] >= 0.0f && w[0] < 0.01f);
}

UTEST(Window, HannPeaksNearMiddle) {
    auto w = speech::dsp::window::generate(Window_Hann, 512);
    float maxVal = 0.0f;
    size_t maxIdx = 0;
    for (size_t i = 0; i < w.size(); ++i) {
        if (w[i] > maxVal) {
            maxVal = w[i];
            maxIdx = i;
        }
    }
    // Peak should be close to 1.0 and near the center of the window.
    ASSERT_TRUE(maxVal > 0.99f);
    ASSERT_TRUE(maxIdx > 240 && maxIdx < 272);
}

UTEST(Window, RectIsAllOnes) {
    auto w = speech::dsp::window::generate(Window_Rect, 100);
    ASSERT_EQ(w.size(), static_cast<size_t>(100));
    for (float v : w) {
        ASSERT_EQ(v, 1.0f);
    }
}

UTEST(Window, HammingDiffersFromHann) {
    // Sanity check that different window types actually produce different
    // coefficients (i.e. the type parameter is really being respected).
    auto hann = speech::dsp::window::generate(Window_Hann, 64);
    auto hamm = speech::dsp::window::generate(Window_Hamm, 64);
    bool anyDifferent = false;
    for (size_t i = 0; i < hann.size(); ++i) {
        if (std::fabs(hann[i] - hamm[i]) > 1e-4f) {
            anyDifferent = true;
            break;
        }
    }
    ASSERT_TRUE(anyDifferent);
}

UTEST(Window, ZeroOrNegativeLengthThrows) {
    bool threw = false;
    try {
        speech::dsp::window::generate(Window_Hann, 0);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    ASSERT_TRUE(threw);
}

UTEST_MAIN();
