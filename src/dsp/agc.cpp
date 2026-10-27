#include "libspeech/dsp/agc.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace speech::dsp {

namespace {
double coefficient(double time_ms, double sample_rate) { return 1.0 - std::exp(-1.0 / (time_ms * 1e-3 * sample_rate)); }
}  // namespace

speech_agc::speech_agc(double sample_rate, const speech_agc_params& params) : sample_rate_(sample_rate), params_(params) {
    auto positive = [](double v) { return std::isfinite(v) && v > 0.0; };
    if (!positive(sample_rate)) throw std::invalid_argument("speech_agc: sample_rate must be a finite number > 0");
    if (!positive(params.level_ms) || !positive(params.attack_ms) || !positive(params.release_ms) ||
        !std::isfinite(params.warmup_ms) || params.warmup_ms < 0.0) {
        throw std::invalid_argument("speech_agc: level_ms, attack_ms and release_ms must be finite numbers > 0, warmup_ms >= 0");
    }
    if (!std::isfinite(params.target_dbfs) || !std::isfinite(params.max_gain_db) || !std::isfinite(params.min_gain_db) ||
        params.max_gain_db < params.min_gain_db) {
        throw std::invalid_argument("speech_agc: target_dbfs, max_gain_db and min_gain_db must be finite, max_gain_db >= min_gain_db");
    }
    if (!positive(params.peak_ceiling)) throw std::invalid_argument("speech_agc: peak_ceiling must be a finite number > 0");
    level_samples_ = params.level_ms * 1e-3 * sample_rate;
    if (level_samples_ < 1.0) throw std::invalid_argument("speech_agc: level_ms is shorter than one sample");
    warmup_samples_ = std::min(params.warmup_ms * 1e-3 * sample_rate, level_samples_);
    target_amplitude_ = std::pow(10.0, params.target_dbfs / 20.0);
    min_gain_ = std::pow(10.0, params.min_gain_db / 20.0);
    max_gain_ = std::pow(10.0, params.max_gain_db / 20.0);
    attack_coeff_ = coefficient(params.attack_ms, sample_rate);
    release_coeff_ = coefficient(params.release_ms, sample_rate);
}

void speech_agc::reset() noexcept {
    mean_square_ = 0.0;
    mass_ = 0.0;
    gain_ = 1.0;
}

double speech_agc::level_dbfs() const noexcept {
    return mass_ > 0.0 && mean_square_ > 0.0 ? 10.0 * std::log10(mean_square_) : -std::numeric_limits<double>::infinity();
}

void speech_agc::process(float* x, std::size_t n, const float* weights, std::size_t n_weights, std::size_t window) {
    if (window == 0) throw std::invalid_argument("speech_agc::process: window must be > 0");
    if (n > 0 && x == nullptr) throw std::invalid_argument("speech_agc::process: null data with a non-zero size");
    if (n_weights > 0 && weights == nullptr) throw std::invalid_argument("speech_agc::process: null weights with a non-zero size");
    for (std::size_t k = 0; k < n_weights; ++k) {
        if (!(weights[k] >= 0.0f && weights[k] <= 1.0f)) throw std::invalid_argument("speech_agc::process: a weight is outside [0, 1]");
    }
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(x[i])) throw std::invalid_argument("speech_agc::process: a sample is NaN or infinite");
    }

    const double ceiling = params_.peak_ceiling;
    float f_ceiling = static_cast<float>(ceiling);  // the largest float that is not above the ceiling
    if (static_cast<double>(f_ceiling) > ceiling) f_ceiling = std::nextafter(f_ceiling, 0.0f);
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t k = i / window;
        const double w = k < n_weights ? static_cast<double>(weights[k]) : 0.0;
        const double sample = static_cast<double>(x[i]);
        if (w > 0.0) {
            mass_ += w;
            mean_square_ += (w / std::min(mass_, level_samples_)) * (sample * sample - mean_square_);
        }
        double desired = 1.0;  // not trusted yet (warm-up)
        if (mass_ >= warmup_samples_ && mass_ > 0.0) {
            const double rms = std::sqrt(mean_square_);
            desired = rms > 1e-12 ? std::min(max_gain_, std::max(min_gain_, target_amplitude_ / rms)) : gain_;
        }
        gain_ += (desired < gain_ ? attack_coeff_ : release_coeff_) * (desired - gain_);

        float result = static_cast<float>(sample * gain_);
        if (std::fabs(static_cast<double>(result)) > ceiling) result = result < 0.0f ? -f_ceiling : f_ceiling;  // peak guard
        x[i] = result;
    }
}

}  // namespace speech::dsp
