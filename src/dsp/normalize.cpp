#include "libspeech/dsp/normalize.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace speech::dsp {

namespace {
double to_dbfs(double linear) {
    return linear > 0.0 ? 20.0 * std::log10(linear) : -std::numeric_limits<double>::infinity();
}
}  // namespace

double rms(const float* x, std::size_t n) {
    if (n == 0) return 0.0;
    if (x == nullptr) throw std::invalid_argument("rms: null data with a non-zero size");
    double energy = 0.0;
    for (std::size_t i = 0; i < n; ++i) energy += static_cast<double>(x[i]) * static_cast<double>(x[i]);
    return std::sqrt(energy / static_cast<double>(n));
}

double rms_dbfs(const float* x, std::size_t n) { return to_dbfs(rms(x, n)); }

rms_normalize_result scale_to_rms(float* x, std::size_t n, double target_rms, double peak_ceiling) {
    if (!std::isfinite(target_rms) || !(target_rms > 0.0)) {
        throw std::invalid_argument("scale_to_rms: target_rms must be a finite number > 0");
    }
    if (!std::isfinite(peak_ceiling) || !(peak_ceiling > 0.0)) {
        throw std::invalid_argument("scale_to_rms: peak_ceiling must be a finite number > 0");
    }
    if (n > 0 && x == nullptr) throw std::invalid_argument("scale_to_rms: null data with a non-zero size");

    rms_normalize_result result;
    double energy = 0.0, peak = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double v = x[i];
        energy += v * v;
        peak = std::max(peak, std::fabs(v));
    }
    // A NaN/inf sample makes the energy non-finite; silence has nothing to scale. Both stay as they are.
    if (n == 0 || !std::isfinite(energy) || !(energy > 0.0)) {
        result.input_rms_dbfs = result.output_rms_dbfs = to_dbfs(n == 0 ? 0.0 : std::sqrt(energy / static_cast<double>(n)));
        return result;
    }

    const double input_rms = std::sqrt(energy / static_cast<double>(n));
    double gain = target_rms / input_rms;
    if (peak * gain > peak_ceiling) {
        gain = peak_ceiling / peak;
        result.peak_limited = true;
    }
    for (std::size_t i = 0; i < n; ++i) x[i] = static_cast<float>(static_cast<double>(x[i]) * gain);

    result.gain = gain;
    result.input_rms_dbfs = to_dbfs(input_rms);
    result.output_rms_dbfs = to_dbfs(input_rms * gain);
    return result;
}

rms_normalize_result rms_normalize(float* x, std::size_t n, double target_dbfs, double peak_ceiling) {
    if (!std::isfinite(target_dbfs)) throw std::invalid_argument("rms_normalize: target_dbfs must be finite");
    return scale_to_rms(x, n, std::pow(10.0, target_dbfs / 20.0), peak_ceiling);
}

void speech_weights(const float* prob, std::size_t n, speech_gate mode, double threshold, float* out) {
    if (!std::isfinite(threshold) || !(threshold > 0.0) || threshold > 1.0 ||
        (mode == speech_gate::soft && !(threshold < 1.0))) {
        throw std::invalid_argument(mode == speech_gate::soft
                                        ? "speech_weights: soft threshold must satisfy 0 < threshold < 1"
                                        : "speech_weights: hard threshold must satisfy 0 < threshold <= 1");
    }
    if (n > 0 && (prob == nullptr || out == nullptr)) throw std::invalid_argument("speech_weights: null data with a non-zero size");
    for (std::size_t i = 0; i < n; ++i) {
        const double p = prob[i];
        if (!(p >= 0.0 && p <= 1.0)) throw std::invalid_argument("speech_weights: a probability is outside [0, 1]");
    }
    for (std::size_t i = 0; i < n; ++i) {
        const double p = prob[i];
        double w;
        if (mode == speech_gate::hard) {
            w = p >= threshold ? 1.0 : 0.0;
        } else {
            w = std::min(1.0, std::max(0.0, (p - threshold) / (1.0 - threshold)));
        }
        out[i] = static_cast<float>(w);
    }
}

namespace {
// Sums w_k * energy_k and w_k * len_k; validates the arguments.
void weighted_sums(const float* x, std::size_t n, const float* weights, std::size_t n_weights, std::size_t window,
                   double& energy, double& count) {
    if (window == 0) throw std::invalid_argument("weighted_rms: window must be > 0");
    if (n > 0 && x == nullptr) throw std::invalid_argument("weighted_rms: null data with a non-zero size");
    if (n_weights > 0 && weights == nullptr) throw std::invalid_argument("weighted_rms: null weights with a non-zero size");
    for (std::size_t k = 0; k < n_weights; ++k) {
        if (!(weights[k] >= 0.0f && weights[k] <= 1.0f)) throw std::invalid_argument("weighted_rms: a weight is outside [0, 1]");
    }
    energy = 0.0;
    count = 0.0;
    for (std::size_t k = 0; k < n_weights; ++k) {
        const std::size_t begin = k * window;
        if (begin >= n) break;
        const std::size_t end = std::min(n, begin + window);
        const double w = weights[k];
        if (w == 0.0) continue;
        double e = 0.0;
        for (std::size_t i = begin; i < end; ++i) e += static_cast<double>(x[i]) * static_cast<double>(x[i]);
        energy += w * e;
        count += w * static_cast<double>(end - begin);
    }
}
}  // namespace

double weighted_rms(const float* x, std::size_t n, const float* weights, std::size_t n_weights, std::size_t window) {
    double energy, count;
    weighted_sums(x, n, weights, n_weights, window, energy, count);
    return count > 0.0 ? std::sqrt(energy / count) : 0.0;
}

rms_normalize_result rms_normalize_weighted(float* x, std::size_t n, const float* weights, std::size_t n_weights,
                                            std::size_t window, double target_dbfs, double peak_ceiling) {
    if (!std::isfinite(target_dbfs)) throw std::invalid_argument("rms_normalize_weighted: target_dbfs must be finite");
    if (!std::isfinite(peak_ceiling) || !(peak_ceiling > 0.0)) {
        throw std::invalid_argument("rms_normalize_weighted: peak_ceiling must be a finite number > 0");
    }
    double energy, count;
    weighted_sums(x, n, weights, n_weights, window, energy, count);

    rms_normalize_result result;
    const bool usable = count > 0.0 && std::isfinite(energy) && energy > 0.0;
    double peak = 0.0;
    bool finite = true;
    for (std::size_t i = 0; i < n; ++i) {
        const double v = std::fabs(static_cast<double>(x[i]));
        if (!std::isfinite(v)) finite = false;
        peak = std::max(peak, v);
    }
    if (!usable || !finite) {
        result.input_rms_dbfs = result.output_rms_dbfs = to_dbfs(count > 0.0 && std::isfinite(energy) ? std::sqrt(energy / count) : 0.0);
        return result;
    }

    const double level = std::sqrt(energy / count);
    double gain = std::pow(10.0, target_dbfs / 20.0) / level;
    if (peak * gain > peak_ceiling) {
        gain = peak_ceiling / peak;
        result.peak_limited = true;
    }
    for (std::size_t i = 0; i < n; ++i) x[i] = static_cast<float>(static_cast<double>(x[i]) * gain);

    result.gain = gain;
    result.input_rms_dbfs = to_dbfs(level);
    result.output_rms_dbfs = to_dbfs(level * gain);
    return result;
}

}  // namespace speech::dsp
