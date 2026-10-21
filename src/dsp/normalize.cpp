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

}  // namespace speech::dsp
