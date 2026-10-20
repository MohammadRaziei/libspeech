// Internal (not installed): puts a denoiser's output at the level of its input.
#ifndef LIBSPEECH_LEVEL_MATCH_H
#define LIBSPEECH_LEVEL_MATCH_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace speech::models::detail {

/**
 * Scales `signal` so that its RMS equals the RMS of `reference`, then lowers the gain if that would push its
 * peak above max(0.99, peak of `reference`).
 *
 * For models trained with a scale-invariant loss (SI-SNR, e.g. SpeechBrain's SepFormer) the absolute level of the
 * output is arbitrary -- in practice 15-18x louder than the input, with samples far beyond +-1, which clips as soon
 * as it is saved as PCM. Leaves `signal` unchanged when either side is silent or not finite.
 */
inline void match_level(const float* reference, std::size_t reference_size, std::vector<float>& signal) noexcept {
    if (reference_size == 0 || signal.empty()) return;
    double reference_energy = 0.0, reference_peak = 0.0, signal_energy = 0.0, signal_peak = 0.0;
    for (std::size_t i = 0; i < reference_size; ++i) {
        const double v = reference[i];
        reference_energy += v * v;
        reference_peak = std::max(reference_peak, std::fabs(v));
    }
    for (const float s : signal) {
        const double v = s;
        signal_energy += v * v;
        signal_peak = std::max(signal_peak, std::fabs(v));
    }
    if (!std::isfinite(reference_energy) || !std::isfinite(signal_energy) || !(reference_energy > 0.0) ||
        !(signal_energy > 0.0)) {
        return;
    }
    double gain = std::sqrt(reference_energy / static_cast<double>(reference_size)) /
                  std::sqrt(signal_energy / static_cast<double>(signal.size()));
    const double ceiling = std::max(0.99, reference_peak);
    if (signal_peak * gain > ceiling) gain = ceiling / signal_peak;
    for (float& s : signal) s = static_cast<float>(static_cast<double>(s) * gain);
}

}  // namespace speech::models::detail

#endif  // LIBSPEECH_LEVEL_MATCH_H
