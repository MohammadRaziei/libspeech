#include "libspeech/dsp/dct.h"

#include <cmath>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

#include "aixlog.hpp"
#include "libspeech/detail/log.h"
#include "real_fft.h"

namespace speech::dsp {

namespace {
constexpr const char* log_tag = "speech::dsp::dct_ii";
constexpr float pi = 3.14159265358979323846f;  // used by DctIIPlan below
}  // namespace

namespace {

// FFT engines for the power-of-two fast path, one per size, shared process-wide: building
// one costs O(N) sin/cos for twiddles, which would dominate a single small transform.
std::shared_ptr<const rfft_impl::real_frame_fft> unit_window_fft(int n) {
    static std::mutex mutex;
    static std::unordered_map<int, std::shared_ptr<const rfft_impl::real_frame_fft>> cache;
    std::lock_guard<std::mutex> lock(mutex);
    auto it = cache.find(n);
    if (it != cache.end()) return it->second;
    if (cache.size() >= 16) cache.clear();  // sizes are few in practice; stay bounded
    const std::vector<float> ones(static_cast<std::size_t>(n), 1.0f);
    auto fft = std::make_shared<const rfft_impl::real_frame_fft>(n, ones.data());
    cache.emplace(n, fft);
    return fft;
}

// DCT-II through one real FFT (Makhoul): reorder v[i] = x[2i], v[n-1-i] = x[2i+1], then
//   sum_i x[i] cos(pi (2i+1) k / 2n) = Re( exp(-i pi k / 2n) * FFT_n(v)[k] ).
// O(n log n); output k in [0, max_value) before scaling. Requires a supported power-of-two n.
void dct_via_fft(const float* x, int n, int max_value, double* sums) {
    auto fft = unit_window_fft(n);
    std::vector<float> v(static_cast<std::size_t>(n)), re(static_cast<std::size_t>(n) / 2 + 1),
        im(static_cast<std::size_t>(n) / 2 + 1);
    for (int i = 0; i < n / 2; ++i) {
        v[i] = x[2 * i];
        v[n - 1 - i] = x[2 * i + 1];
    }
    fft->spectrum(v.data(), re.data(), im.data());
    constexpr double pi_double = 3.14159265358979323846;
    for (int k = 0; k < max_value; ++k) {
        const double th = pi_double * k / (2.0 * n);
        const double c = std::cos(th), s = std::sin(th);
        if (k <= n / 2) {
            sums[k] = re[k] * c + im[k] * s;          // Re((c - i s)(a + i b))
        } else {
            const int m = n - k;                       // V[k] = conj(V[n-k])
            sums[k] = re[m] * c - im[m] * s;
        }
    }
}

// Direct O(n * max_value) sums without evaluating cos per term: cos(pi (2i+1) k / 2n) is
// table[(k (2i+1)) mod 4n] for a table of 4n values built once (4n cos calls instead of n*max_value).
void dct_direct(const float* x, int n, int max_value, double* sums) {
    const long long period = 4LL * n;
    std::vector<double> table(static_cast<std::size_t>(period));
    constexpr double pi_double = 3.14159265358979323846;
    for (long long j = 0; j < period; ++j) table[j] = std::cos(pi_double * static_cast<double>(j) / (2.0 * n));
    for (int k = 0; k < max_value; ++k) {
        const long long step = (2LL * k) % period;
        long long idx = k % period;  // (2*0+1) * k
        double sum = 0.0;
        for (int i = 0; i < n; ++i) {
            sum += static_cast<double>(x[i]) * table[static_cast<std::size_t>(idx)];
            idx += step;
            if (idx >= period) idx -= period;
        }
        sums[k] = sum;
    }
}

}  // namespace

std::vector<float> dct_ii(const std::vector<float>& input, int num_outputs, bool orthonormal) {
    const int n = static_cast<int>(input.size());
    if (n <= 0) {
        SPEECH_LOG(ERROR) << TAG(log_tag) << "dct_ii() called with empty input." << std::endl;
        throw std::invalid_argument("dct_ii() input must not be empty.");
    }

    const int k_max = (num_outputs < 0) ? n : num_outputs;
    if (k_max <= 0 || k_max > n) {
        SPEECH_LOG(ERROR) << TAG(log_tag) << "num_outputs=" << num_outputs << " out of range for input of length "
                   << n << std::endl;
        throw std::invalid_argument("num_outputs must be in [1, input.size()].");
    }

    std::vector<double> sums(static_cast<std::size_t>(k_max));
    // FFT path for every power-of-two n the real-FFT engine supports, whatever the number of
    // outputs (measured: n=512 -> 13 outputs takes 29 us direct, 9 us via FFT). Other sizes
    // use the table-driven direct sum.
    const bool use_fft = rfft_impl::real_frame_fft::supports(n);
    if (use_fft) {
        dct_via_fft(input.data(), n, k_max, sums.data());
    } else {
        dct_direct(input.data(), n, k_max, sums.data());
    }

    std::vector<float> output(static_cast<std::size_t>(k_max), 0.0f);
    for (int k = 0; k < k_max; ++k) {
        float value = static_cast<float>(2.0 * sums[k]);
        if (orthonormal) {
            value *= (k == 0) ? std::sqrt(1.0f / (4.0f * n)) : std::sqrt(1.0f / (2.0f * n));
        }
        output[k] = value;
    }
    return output;
}

dct_ii_plan::dct_ii_plan(int n, int num_outputs, bool orthonormal) : n_(n), k_(num_outputs) {
    if (n <= 0 || num_outputs <= 0 || num_outputs > n) {
        throw std::invalid_argument("DctIIPlan: need 1 <= num_outputs <= n.");
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

void dct_ii_plan::apply(const float* in, float* out) const {
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
