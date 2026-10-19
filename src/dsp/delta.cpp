#include "libspeech/dsp/delta.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace speech::dsp {

void delta_into(const float* in, int num_frames, int num_features, int half_window, delta_edge edge, float* out) {
    if (half_window < 1) throw std::invalid_argument("half_window must be >= 1.");
    if (num_features < 1) throw std::invalid_argument("num_features must be >= 1.");
    if (num_frames < 0) throw std::invalid_argument("num_frames must be >= 0.");
    const int width = 2 * half_window + 1;
    if (edge == delta_edge::interp && num_frames > 0 && num_frames < width) {
        throw std::invalid_argument("delta_edge::interp needs at least 2*half_window + 1 frames.");
    }
    if (num_frames == 0) return;

    const std::size_t nf = static_cast<std::size_t>(num_features);
    const auto row = [&](int t) { return in + static_cast<std::size_t>(t) * nf; };

    // 2 * sum_{n=1..N} n^2 = N(N+1)(2N+1)/3
    const float inv_denominator =
        3.0f / (static_cast<float>(half_window) * (half_window + 1.0f) * (2.0f * half_window + 1.0f));

    // slope of the line fitted to the `width` rows starting at `first`; the same value is used for every edge frame.
    const auto fit_slope = [&](int first, float* dst) {
        std::fill(dst, dst + nf, 0.0f);
        for (int k = 1; k <= half_window; ++k) {
            const float* hi = row(first + half_window + k);
            const float* lo = row(first + half_window - k);
            const float w = static_cast<float>(k);
            for (std::size_t f = 0; f < nf; ++f) dst[f] += w * (hi[f] - lo[f]);
        }
        for (std::size_t f = 0; f < nf; ++f) dst[f] *= inv_denominator;
    };

    int begin = 0;
    int end = num_frames;
    if (edge == delta_edge::interp) {
        // Frames [0, N) and [T-N, T) all get the slope of the first / last window (T >= width is checked above).
        float* first_dst = out;
        fit_slope(0, first_dst);
        for (int t = 1; t < half_window; ++t) std::copy(first_dst, first_dst + nf, out + static_cast<std::size_t>(t) * nf);
        float* last_dst = out + static_cast<std::size_t>(num_frames - half_window) * nf;
        fit_slope(num_frames - width, last_dst);
        for (int t = num_frames - half_window + 1; t < num_frames; ++t) {
            std::copy(last_dst, last_dst + nf, out + static_cast<std::size_t>(t) * nf);
        }
        begin = half_window;
        end = num_frames - half_window;
    }

    for (int t = begin; t < end; ++t) {
        float* dst = out + static_cast<std::size_t>(t) * nf;
        std::fill(dst, dst + nf, 0.0f);
        for (int k = 1; k <= half_window; ++k) {
            const float* hi = row(std::min(t + k, num_frames - 1));
            const float* lo = row(std::max(t - k, 0));
            const float w = static_cast<float>(k);
            for (std::size_t f = 0; f < nf; ++f) dst[f] += w * (hi[f] - lo[f]);
        }
        for (std::size_t f = 0; f < nf; ++f) dst[f] *= inv_denominator;
    }
}

mfcc_matrix delta(const float* in, int num_frames, int num_features, int half_window, delta_edge edge) {
    mfcc_matrix out;
    out.num_frames = num_frames;
    out.num_coefficients = num_features;
    out.data.resize(static_cast<std::size_t>(std::max(num_frames, 0)) * static_cast<std::size_t>(std::max(num_features, 0)));
    delta_into(in, num_frames, num_features, half_window, edge, out.data.data());
    return out;
}

}  // namespace speech::dsp
