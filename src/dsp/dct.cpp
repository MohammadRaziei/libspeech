#include "libspeech/dsp/dct.h"

#include <cmath>
#include <stdexcept>

#include "aixlog.hpp"

namespace speech::dsp {

namespace {
constexpr const char* kTag = "speech::dsp::dctII";
constexpr float kPi = 3.14159265358979323846f;
}  // namespace

std::vector<float> dctII(const std::vector<float>& input, int numOutputs, bool orthonormal) {
    const int n = static_cast<int>(input.size());
    if (n <= 0) {
        LOG(ERROR) << TAG(kTag) << "dctII() called with empty input." << std::endl;
        throw std::invalid_argument("dctII() input must not be empty.");
    }

    const int k_max = (numOutputs < 0) ? n : numOutputs;
    if (k_max <= 0 || k_max > n) {
        LOG(ERROR) << TAG(kTag) << "numOutputs=" << numOutputs << " out of range for input of length "
                   << n << std::endl;
        throw std::invalid_argument("numOutputs must be in [1, input.size()].");
    }

    std::vector<float> output(k_max, 0.0f);
    for (int k = 0; k < k_max; ++k) {
        double sum = 0.0;
        for (int i = 0; i < n; ++i) {
            sum += static_cast<double>(input[i]) *
                   std::cos(kPi * (2.0 * i + 1.0) * k / (2.0 * n));
        }
        float value = static_cast<float>(2.0 * sum);

        if (orthonormal) {
            value *= (k == 0) ? std::sqrt(1.0f / (4.0f * n)) : std::sqrt(1.0f / (2.0f * n));
        }
        output[k] = value;
    }

    return output;
}

DctII::DctII(int n, int numOutputs, bool orthonormal) : n_(n), k_(numOutputs) {
    if (n <= 0 || numOutputs <= 0 || numOutputs > n) {
        throw std::invalid_argument("DctII: need 1 <= numOutputs <= n.");
    }
    basis_.resize(static_cast<std::size_t>(k_) * n_);
    for (int k = 0; k < k_; ++k) {
        double scale = 2.0;
        if (orthonormal) {
            scale *= (k == 0) ? std::sqrt(1.0 / (4.0 * n)) : std::sqrt(1.0 / (2.0 * n));
        }
        for (int i = 0; i < n_; ++i) {
            basis_[static_cast<std::size_t>(k) * n_ + i] = static_cast<float>(
                scale * std::cos(3.14159265358979323846 * (2.0 * i + 1.0) * k / (2.0 * n)));
        }
    }
}

void DctII::apply(const float* in, float* out) const {
    for (int k = 0; k < k_; ++k) {
        const float* row = basis_.data() + static_cast<std::size_t>(k) * n_;
        float acc = 0.0f;
        for (int i = 0; i < n_; ++i) {
            acc += row[i] * in[i];
        }
        out[k] = acc;
    }
}

}  // namespace speech::dsp
