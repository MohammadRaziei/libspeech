#include "libspeech/dsp/normalize.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <limits>
#include <stdexcept>
#include <vector>

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
// Sums w_k * energy_k and w_k * len_k (with the loud windows clipped to the weighted clip_percentile of the window
// mean squares when clip_percentile < 100); validates the arguments.
void weighted_sums(const float* x, std::size_t n, const float* weights, std::size_t n_weights, std::size_t window,
                   double clip_percentile, double& energy, double& count) {
    if (window == 0) throw std::invalid_argument("weighted_rms: window must be > 0");
    if (!(clip_percentile > 0.0 && clip_percentile <= 100.0)) {
        throw std::invalid_argument("weighted_rms: clip_percentile must be in (0, 100]");
    }
    if (n > 0 && x == nullptr) throw std::invalid_argument("weighted_rms: null data with a non-zero size");
    if (n_weights > 0 && weights == nullptr) throw std::invalid_argument("weighted_rms: null weights with a non-zero size");
    for (std::size_t k = 0; k < n_weights; ++k) {
        if (!(weights[k] >= 0.0f && weights[k] <= 1.0f)) throw std::invalid_argument("weighted_rms: a weight is outside [0, 1]");
    }
    energy = 0.0;
    count = 0.0;
    struct item {
        double mean_square;  // energy of the window / its length
        double mass;         // w_k * len_k
    };
    std::vector<item> items;
    const bool clip = clip_percentile < 100.0;
    for (std::size_t k = 0; k < n_weights; ++k) {
        const std::size_t begin = k * window;
        if (begin >= n) break;
        const std::size_t end = std::min(n, begin + window);
        const double w = weights[k];
        if (w == 0.0) continue;
        double e = 0.0;
        for (std::size_t i = begin; i < end; ++i) e += static_cast<double>(x[i]) * static_cast<double>(x[i]);
        const double len = static_cast<double>(end - begin);
        energy += w * e;
        count += w * len;
        if (clip) items.push_back({e / len, w * len});
    }
    if (!clip || items.empty() || !std::isfinite(energy)) return;

    // Weighted percentile of the window mean squares: the first one whose cumulative mass reaches p% of the total.
    std::sort(items.begin(), items.end(), [](const item& a, const item& b) { return a.mean_square < b.mean_square; });
    const double target_mass = count * clip_percentile / 100.0;
    double cumulative = 0.0, limit = items.back().mean_square;
    for (const item& it : items) {
        cumulative += it.mass;
        if (cumulative >= target_mass) {
            limit = it.mean_square;
            break;
        }
    }
    energy = 0.0;
    for (const item& it : items) energy += it.mass * std::min(it.mean_square, limit);
}
}  // namespace

double weighted_rms(const float* x, std::size_t n, const float* weights, std::size_t n_weights, std::size_t window,
                    double clip_percentile) {
    double energy, count;
    weighted_sums(x, n, weights, n_weights, window, clip_percentile, energy, count);
    return count > 0.0 ? std::sqrt(energy / count) : 0.0;
}

bool limit_peaks(float* x, std::size_t n, double ceiling, std::size_t lookahead) {
    if (!std::isfinite(ceiling) || !(ceiling > 0.0)) throw std::invalid_argument("limit_peaks: ceiling must be a finite number > 0");
    if (n > 0 && x == nullptr) throw std::invalid_argument("limit_peaks: null data with a non-zero size");
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(x[i])) throw std::invalid_argument("limit_peaks: a sample is NaN or infinite");
    }
    const std::size_t L = std::max<std::size_t>(lookahead, 1);
    std::vector<double> g(n);
    bool limited = false;
    for (std::size_t i = 0; i < n; ++i) {
        const double a = std::fabs(static_cast<double>(x[i]));
        g[i] = a > ceiling ? ceiling / a : 1.0;
        if (a > ceiling) limited = true;
    }
    if (!limited) return false;

    // env[m] is the minimum of g over the L samples ending at m, clamped to [0, n - 1]: [max(m - L + 1, 0), min(m, n - 1)],
    // for m = 0 .. n + L - 2. (m = j + L - 1 for the look-ahead window starting at sample j = m - L + 1.)
    std::vector<double> env(n + L - 1, 1.0);
    std::deque<std::size_t> window_min;  // indices into g, with g increasing from front to back
    for (std::size_t m = 0; m < env.size(); ++m) {
        if (m < n) {
            while (!window_min.empty() && g[window_min.back()] >= g[m]) window_min.pop_back();
            window_min.push_back(m);
        }
        const std::size_t left = m + 1 >= L ? m + 1 - L : 0;
        while (!window_min.empty() && window_min.front() < left) window_min.pop_front();
        env[m] = g[window_min.front()];
    }
    // s[i] = mean of env over j in [i - L + 1, i], i.e. env[i .. i + L - 1] in the shifted index.
    double running = 0.0;
    for (std::size_t m = 0; m < L - 1; ++m) running += env[m];
    for (std::size_t i = 0; i < n; ++i) {
        running += env[i + L - 1];
        const double s = running / static_cast<double>(L);
        running -= env[i];
        x[i] = static_cast<float>(static_cast<double>(x[i]) * s);
    }
    // s <= ceiling / |x| holds in double, but the product rounded to float can land one ulp above a ceiling that is not
    // representable as float; bring such a sample just below it
    const float f_ceiling = static_cast<float>(ceiling);
    for (std::size_t i = 0; i < n; ++i) {
        if (static_cast<double>(x[i]) > ceiling) x[i] = std::nextafter(f_ceiling, 0.0f);
        else if (static_cast<double>(x[i]) < -ceiling) x[i] = -std::nextafter(f_ceiling, 0.0f);
    }
    return true;
}

rms_normalize_result rms_normalize_weighted(float* x, std::size_t n, const float* weights, std::size_t n_weights,
                                            std::size_t window, double target_dbfs, double peak_ceiling,
                                            double clip_percentile, std::size_t limiter_lookahead) {
    if (!std::isfinite(target_dbfs)) throw std::invalid_argument("rms_normalize_weighted: target_dbfs must be finite");
    if (!std::isfinite(peak_ceiling) || !(peak_ceiling > 0.0)) {
        throw std::invalid_argument("rms_normalize_weighted: peak_ceiling must be a finite number > 0");
    }
    double energy, count;
    weighted_sums(x, n, weights, n_weights, window, clip_percentile, energy, count);

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
    const bool use_limiter = limiter_lookahead > 0;
    if (!use_limiter && peak * gain > peak_ceiling) {
        gain = peak_ceiling / peak;
        result.peak_limited = true;
    }
    for (std::size_t i = 0; i < n; ++i) x[i] = static_cast<float>(static_cast<double>(x[i]) * gain);
    if (use_limiter) result.peak_limited = limit_peaks(x, n, peak_ceiling, limiter_lookahead);

    result.gain = gain;
    result.input_rms_dbfs = to_dbfs(level);
    result.output_rms_dbfs = to_dbfs(level * gain);
    return result;
}

void weighted_cmvn(float* features, std::size_t frames, std::size_t dim, const float* weights, bool normalize_variance,
                   double eps) {
    if (dim == 0) throw std::invalid_argument("weighted_cmvn: dim must be > 0");
    if (!std::isfinite(eps) || eps < 0.0) throw std::invalid_argument("weighted_cmvn: eps must be a finite number >= 0");
    if (frames > 0 && (features == nullptr || weights == nullptr)) {
        throw std::invalid_argument("weighted_cmvn: null data with a non-zero size");
    }
    double total = 0.0;
    for (std::size_t t = 0; t < frames; ++t) {
        if (!(weights[t] >= 0.0f && weights[t] <= 1.0f)) throw std::invalid_argument("weighted_cmvn: a weight is outside [0, 1]");
        total += weights[t];
    }
    if (frames == 0 || !(total > 0.0)) return;

    std::vector<double> mean(dim, 0.0), scale(dim, 1.0);
    for (std::size_t t = 0; t < frames; ++t) {
        const double w = weights[t];
        if (w == 0.0) continue;
        for (std::size_t d = 0; d < dim; ++d) mean[d] += w * static_cast<double>(features[t * dim + d]);
    }
    for (std::size_t d = 0; d < dim; ++d) mean[d] /= total;
    if (normalize_variance) {
        std::vector<double> var(dim, 0.0);
        for (std::size_t t = 0; t < frames; ++t) {
            const double w = weights[t];
            if (w == 0.0) continue;
            for (std::size_t d = 0; d < dim; ++d) {
                const double c = static_cast<double>(features[t * dim + d]) - mean[d];
                var[d] += w * c * c;
            }
        }
        for (std::size_t d = 0; d < dim; ++d) scale[d] = 1.0 / std::sqrt(var[d] / total + eps);
    }
    for (std::size_t t = 0; t < frames; ++t) {
        for (std::size_t d = 0; d < dim; ++d) {
            features[t * dim + d] = static_cast<float>((static_cast<double>(features[t * dim + d]) - mean[d]) * scale[d]);
        }
    }
}

}  // namespace speech::dsp
