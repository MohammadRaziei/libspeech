#include "utest.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "libspeech/dsp/delta.h"

namespace {

using speech::dsp::delta_edge;

// Deterministic pseudo-random matrix (frame-major), no dependence on <random> implementation details.
std::vector<float> make_matrix(int frames, int features) {
    std::vector<float> m(static_cast<size_t>(frames) * features);
    unsigned state = 12345u;
    for (auto& v : m) {
        state = state * 1664525u + 1013904223u;
        v = (static_cast<float>(state >> 8) / 16777216.0f) * 2.0f - 1.0f;
    }
    return m;
}

// Independent double-precision reference written straight from the definition (clamped indices), not sharing
// code or loop structure with the implementation.
double ref_replicate(const std::vector<float>& c, int T, int F, int N, int t, int f) {
    double num = 0.0, den = 0.0;
    for (int n = 1; n <= N; ++n) {
        const int hi = std::min(t + n, T - 1);
        const int lo = std::max(t - n, 0);
        num += n * (static_cast<double>(c[hi * F + f]) - c[lo * F + f]);
        den += 2.0 * n * n;
    }
    return num / den;
}

// Least-squares slope of the line through `width` consecutive points starting at frame `first`
// (normal equations: slope = sum((x-xm)(y-ym)) / sum((x-xm)^2)).
double ref_line_slope(const std::vector<float>& c, int F, int first, int width, int f) {
    double xm = 0.0, ym = 0.0;
    for (int i = 0; i < width; ++i) {
        xm += i;
        ym += c[(first + i) * F + f];
    }
    xm /= width;
    ym /= width;
    double sxy = 0.0, sxx = 0.0;
    for (int i = 0; i < width; ++i) {
        sxy += (i - xm) * (c[(first + i) * F + f] - ym);
        sxx += (i - xm) * (i - xm);
    }
    return sxy / sxx;
}

}  // namespace

UTEST(delta, replicate_matches_definition_for_several_windows) {
    const int T = 37, F = 5;
    auto c = make_matrix(T, F);
    for (int N : {1, 2, 4, 7}) {
        auto d = speech::dsp::delta(c.data(), T, F, N, delta_edge::replicate);
        ASSERT_EQ(d.num_frames, T);
        ASSERT_EQ(d.num_coefficients, F);
        double worst = 0.0;
        for (int t = 0; t < T; ++t)
            for (int f = 0; f < F; ++f)
                worst = std::max(worst, std::fabs(d.data[t * F + f] - ref_replicate(c, T, F, N, t, f)));
        ASSERT_LT(worst, 1e-5);
    }
}

UTEST(delta, interp_matches_line_fit_reference_everywhere) {
    const int T = 40, F = 4;
    auto c = make_matrix(T, F);
    for (int N : {1, 2, 4}) {
        const int width = 2 * N + 1;
        auto d = speech::dsp::delta(c.data(), T, F, N, delta_edge::interp);
        double worst = 0.0;
        for (int t = 0; t < T; ++t) {
            for (int f = 0; f < F; ++f) {
                double want;
                if (t < N) {
                    want = ref_line_slope(c, F, 0, width, f);
                } else if (t >= T - N) {
                    want = ref_line_slope(c, F, T - width, width, f);
                } else {
                    want = ref_line_slope(c, F, t - N, width, f);
                }
                worst = std::max(worst, std::fabs(d.data[t * F + f] - want));
            }
        }
        ASSERT_LT(worst, 1e-5);
    }
}

// A linear ramp has a constant slope: interp gets it exactly at every frame, replicate only away from the edges
// (and shrinks it at the edges, which is the documented difference between the two modes).
UTEST(delta, linear_ramp_has_constant_delta) {
    const int T = 30, F = 2, N = 3;
    std::vector<float> c(T * F);
    for (int t = 0; t < T; ++t) {
        c[t * F] = 0.5f * t;
        c[t * F + 1] = -2.0f * t + 1.0f;
    }
    auto interp = speech::dsp::delta(c.data(), T, F, N, delta_edge::interp);
    auto repl = speech::dsp::delta(c.data(), T, F, N, delta_edge::replicate);
    for (int t = 0; t < T; ++t) {
        ASSERT_NEAR(interp.data[t * F], 0.5f, 1e-5);
        ASSERT_NEAR(interp.data[t * F + 1], -2.0f, 1e-5);
    }
    for (int t = N; t < T - N; ++t) {
        ASSERT_NEAR(repl.data[t * F], 0.5f, 1e-5);
        ASSERT_NEAR(repl.data[t * F + 1], -2.0f, 1e-5);
    }
    ASSERT_LT(std::fabs(repl.data[0]), 0.5f);  // edge frame is damped under replicate padding
}

UTEST(delta, constant_input_gives_zero) {
    const int T = 20, F = 3;
    std::vector<float> c(T * F, 3.25f);
    for (auto mode : {delta_edge::replicate, delta_edge::interp}) {
        auto d = speech::dsp::delta(c.data(), T, F, 2, mode);
        for (float v : d.data) ASSERT_NEAR(v, 0.0f, 1e-6);
    }
}

UTEST(delta, delta_of_delta_is_second_order) {
    // x[t] = t^2 -> second derivative 2 (interior, window large enough for the two passes).
    const int T = 40, F = 1, N = 2;
    std::vector<float> c(T);
    for (int t = 0; t < T; ++t) c[t] = static_cast<float>(t) * t;
    auto d1 = speech::dsp::delta(c.data(), T, F, N, delta_edge::replicate);
    auto d2 = speech::dsp::delta(d1.data.data(), T, F, N, delta_edge::replicate);
    for (int t = 2 * N; t < T - 2 * N; ++t) ASSERT_NEAR(d2.data[t], 2.0f, 1e-3);
}

UTEST(delta, short_inputs_and_bad_arguments) {
    std::vector<float> c(4 * 2, 1.0f);
    // replicate works on any non-empty input, even a single frame (delta 0).
    auto one = speech::dsp::delta(c.data(), 1, 2, 2, delta_edge::replicate);
    ASSERT_EQ(one.data.size(), static_cast<size_t>(2));
    ASSERT_NEAR(one.data[0], 0.0f, 1e-7);
    auto none = speech::dsp::delta(c.data(), 0, 2, 2, delta_edge::replicate);
    ASSERT_EQ(none.num_frames, 0);

    bool threw = false;
    try { speech::dsp::delta(c.data(), 4, 2, 2, delta_edge::interp); } catch (const std::invalid_argument&) { threw = true; }
    ASSERT_TRUE(threw);  // interp needs 2*2+1 = 5 frames
    threw = false;
    try { speech::dsp::delta(c.data(), 4, 2, 0, delta_edge::replicate); } catch (const std::invalid_argument&) { threw = true; }
    ASSERT_TRUE(threw);
    threw = false;
    try { speech::dsp::delta(c.data(), 4, 0, 2, delta_edge::replicate); } catch (const std::invalid_argument&) { threw = true; }
    ASSERT_TRUE(threw);
}
