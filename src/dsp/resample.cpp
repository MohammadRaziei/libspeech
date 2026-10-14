#include "libspeech/dsp/resample.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <list>
#include <mutex>
#include <numeric>
#include <stdexcept>

#include "aixlog.hpp"
#include "libspeech/detail/log.h"
#include "dsp/resample_algorithm.h"  // Vendored AudioFlux C header (src/third_party/audioflux)
#include "simd_internal.h"
#include "window_type_internal.h"

namespace speech::dsp {

namespace {
struct quality_params {
    int zero_num;
    float beta;
    float roll_off;
};
quality_params params_for(resample_quality q) {
    switch (q) {
        case resample_quality::best:   return {64, 14.7696565f, 0.9475937f};
        case resample_quality::high:   return {48, 12.0f, 0.95f};
        case resample_quality::medium: return {32, 11.0f, 0.95f};
        case resample_quality::fast:   return {16, 9.4f, 0.92f};
    }
    throw std::invalid_argument("unknown resample_quality");
}
}  // namespace

namespace {
constexpr const char* log_tag = "speech::dsp::Resample";

// Polyphase kernels. Output i uses weight row (i*down) mod up and the input
// window starting at n_i = floor(i*down/up): exact integer arithmetic, no float
// time accumulation. Rows are lanes-padded with zeros.
constexpr int lanes = 8;

// All kernels compute outputs i in [i0, i1) into out[i]. Output i reads the wlen
// inputs base[n_i - bias + k], k in [0, wlen), with n_i = floor(i*down/up); the
// caller guarantees those indices are valid (the interior of the signal, or a
// small zero-padded edge buffer).
//
// Portable kernel: 8 independent accumulators, vectorizes without -ffast-math
// because the lanes never mix until the final reduction.
void run_polyphase_generic(const float* weights, int wlen, int up, int down, const float* base,
                         std::ptrdiff_t bias, float* out, std::size_t i0, std::size_t i1) {
    const long long start = static_cast<long long>(i0) * down;
    long long n = start / up;
    int ph = static_cast<int>(start % up);
    const int step_n = down / up;
    const int step_p = down % up;
    for (std::size_t i = i0; i < i1; ++i) {
        const float* w = weights + static_cast<std::size_t>(ph) * wlen;
        const float* x = base + (static_cast<std::ptrdiff_t>(n) - bias);
        float acc[lanes] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int k = 0; k < wlen; k += lanes) {
            for (int l = 0; l < lanes; ++l) acc[l] += w[k + l] * x[k + l];
        }
        out[i] = ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]));
        n += step_n;
        ph += step_p;
        if (ph >= up) {
            ph -= up;
            ++n;
        }
    }
}

// ---- x86: AVX2 + FMA (compiled only where LS_HAVE_X86_AVX2; chosen at run time) ----
#if LS_HAVE_X86_AVX2

LS_TARGET_AVX2 inline float hsum256(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}

// Outputs i = ph + up*m (fixed ph, varying m) share one weight row -- a permutation
// of ph, NOT ph itself -- and their input windows are exactly `down` samples apart,
// so each 8-float weight chunk is loaded once and FMA'd against 8 outputs (9 loads
// per 8 FMAs instead of 16).
//
// Loop order matters for long inputs: the outer loop walks groups of 8 consecutive
// m (a ~15 KiB input span at 44.1 -> 16 kHz, resident in L1) and the inner loop
// sweeps every phase over that span. The all-phases weight set (up*wlen floats,
// ~245 KiB here) lives in L2. Sweeping phase-major instead re-streams the ENTIRE
// input once per phase: fine while the input fits in cache (10 s), several times
// slower once it does not (60 s).
LS_TARGET_AVX2 void run_polyphase_avx2(const float* weights, int wlen, int up, int down,
                                     const float* base, std::ptrdiff_t bias, float* out,
                                     std::size_t i0, std::size_t i1) {
    const std::size_t d = static_cast<std::size_t>(down);
    const std::size_t U = static_cast<std::size_t>(up);
    const std::size_t m_lo = i0 / U;
    const std::size_t m_hi = (i1 + U - 1) / U;

    for (std::size_t m0 = m_lo; m0 < m_hi; m0 += 8) {
        for (int ph = 0; ph < up; ++ph) {
            const long long ph_down = static_cast<long long>(ph) * down;
            const float* w = weights + static_cast<std::size_t>(ph_down % up) * wlen;
            const std::size_t n0 = static_cast<std::size_t>(ph_down / up) + m0 * d;
            const float* x = base + (static_cast<std::ptrdiff_t>(n0) - bias);
            const std::size_t first = static_cast<std::size_t>(ph) + U * m0;  // output index of j = 0
            float* o = out + first;

            if (first >= i0 && first + U * 7 < i1) {
                __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0,
                       a6 = a0, a7 = a0;
                for (int k = 0; k < wlen; k += 8) {
                    const __m256 wv = _mm256_load_ps(w + k);
                    a0 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x + k), a0);
                    a1 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x + d + k), a1);
                    a2 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x + 2 * d + k), a2);
                    a3 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x + 3 * d + k), a3);
                    a4 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x + 4 * d + k), a4);
                    a5 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x + 5 * d + k), a5);
                    a6 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x + 6 * d + k), a6);
                    a7 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x + 7 * d + k), a7);
                }
                o[0] = hsum256(a0);
                o[U] = hsum256(a1);
                o[2 * U] = hsum256(a2);
                o[3 * U] = hsum256(a3);
                o[4 * U] = hsum256(a4);
                o[5 * U] = hsum256(a5);
                o[6 * U] = hsum256(a6);
                o[7 * U] = hsum256(a7);
            } else {  // partial group at either end of [i0, i1)
                for (std::size_t j = 0; j < 8; ++j) {
                    const std::size_t idx = first + U * j;
                    if (idx < i0) continue;
                    if (idx >= i1) break;
                    const float* xj = x + j * d;
                    __m256 a = _mm256_setzero_ps();
                    for (int k = 0; k < wlen; k += 8) {
                        a = _mm256_fmadd_ps(_mm256_load_ps(w + k), _mm256_loadu_ps(xj + k), a);
                    }
                    out[idx] = hsum256(a);
                }
            }
        }
    }
}
#endif  // LS_HAVE_X86_AVX2

// ---- ARM64: NEON (compiled only where LS_HAVE_ARM_NEON) ---------------------------
// Same blocked structure as the AVX2 kernel with 4-lane vectors: 8 outputs share
// each weight vector (9 loads per 8 FMAs).
#if LS_HAVE_ARM_NEON
void run_polyphase_neon(const float* weights, int wlen, int up, int down, const float* base,
                      std::ptrdiff_t bias, float* out, std::size_t i0, std::size_t i1) {
    const std::size_t d = static_cast<std::size_t>(down);
    const std::size_t U = static_cast<std::size_t>(up);
    const std::size_t m_lo = i0 / U;
    const std::size_t m_hi = (i1 + U - 1) / U;

    for (std::size_t m0 = m_lo; m0 < m_hi; m0 += 8) {
        for (int ph = 0; ph < up; ++ph) {
            const long long ph_down = static_cast<long long>(ph) * down;
            const float* w = weights + static_cast<std::size_t>(ph_down % up) * wlen;
            const std::size_t n0 = static_cast<std::size_t>(ph_down / up) + m0 * d;
            const float* x = base + (static_cast<std::ptrdiff_t>(n0) - bias);
            const std::size_t first = static_cast<std::size_t>(ph) + U * m0;
            float* o = out + first;

            if (first >= i0 && first + U * 7 < i1) {
                float32x4_t a0 = vdupq_n_f32(0.0f), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0,
                            a6 = a0, a7 = a0;
                for (int k = 0; k < wlen; k += 4) {
                    const float32x4_t wv = vld1q_f32(w + k);
                    a0 = vfmaq_f32(a0, wv, vld1q_f32(x + k));
                    a1 = vfmaq_f32(a1, wv, vld1q_f32(x + d + k));
                    a2 = vfmaq_f32(a2, wv, vld1q_f32(x + 2 * d + k));
                    a3 = vfmaq_f32(a3, wv, vld1q_f32(x + 3 * d + k));
                    a4 = vfmaq_f32(a4, wv, vld1q_f32(x + 4 * d + k));
                    a5 = vfmaq_f32(a5, wv, vld1q_f32(x + 5 * d + k));
                    a6 = vfmaq_f32(a6, wv, vld1q_f32(x + 6 * d + k));
                    a7 = vfmaq_f32(a7, wv, vld1q_f32(x + 7 * d + k));
                }
                o[0] = vaddvq_f32(a0);
                o[U] = vaddvq_f32(a1);
                o[2 * U] = vaddvq_f32(a2);
                o[3 * U] = vaddvq_f32(a3);
                o[4 * U] = vaddvq_f32(a4);
                o[5 * U] = vaddvq_f32(a5);
                o[6 * U] = vaddvq_f32(a6);
                o[7 * U] = vaddvq_f32(a7);
            } else {  // partial group at either end of [i0, i1)
                for (std::size_t j = 0; j < 8; ++j) {
                    const std::size_t idx = first + U * j;
                    if (idx < i0) continue;
                    if (idx >= i1) break;
                    const float* xj = x + j * d;
                    float32x4_t a = vdupq_n_f32(0.0f);
                    for (int k = 0; k < wlen; k += 4) {
                        a = vfmaq_f32(a, vld1q_f32(w + k), vld1q_f32(xj + k));
                    }
                    out[idx] = vaddvq_f32(a);
                }
            }
        }
    }
}
#endif  // LS_HAVE_ARM_NEON

// Picks the kernel for this call: vectorized twin if compiled in AND the CPU/env
// allow it (see libspeech/dsp/simd.h), else the portable one. Every twin computes
// the same sums in the same lane structure, so results agree to rounding.
void run_polyphase_range(const float* weights, int wlen, int up, int down, const float* base,
                       std::ptrdiff_t bias, float* out, std::size_t i0, std::size_t i1) {
    if (i0 >= i1) return;
#if LS_HAVE_X86_AVX2
    if (simd::use_avx2_fma()) {
        run_polyphase_avx2(weights, wlen, up, down, base, bias, out, i0, i1);
        return;
    }
#endif
#if LS_HAVE_ARM_NEON
    if (simd::use_neon()) {
        run_polyphase_neon(weights, wlen, up, down, base, bias, out, i0, i1);
        return;
    }
#endif
    run_polyphase_generic(weights, wlen, up, down, base, bias, out, i0, i1);
}

// Outputs are independent, so long signals are split into chunks of whole 8-m groups
// (each kernel call writes only its own [a, b) range) and run on all cores when OpenMP
// is available. Short inputs stay on one thread: thread start-up would cost more than
// the work.
void run_polyphase(const float* weights, int wlen, int up, int down, const float* base,
                  std::ptrdiff_t bias, float* out, std::size_t i0, std::size_t i1) {
    if (i0 >= i1) return;
    const std::size_t chunk = static_cast<std::size_t>(up) * 8 * 16;  // 128 m-values per task
    const std::size_t span = i1 - i0;
    if (span < 4 * chunk) {
        run_polyphase_range(weights, wlen, up, down, base, bias, out, i0, i1);
        return;
    }
    const long long tasks = static_cast<long long>((span + chunk - 1) / chunk);
#pragma omp parallel for schedule(static)
    for (long long t = 0; t < tasks; ++t) {
        const std::size_t a = i0 + static_cast<std::size_t>(t) * chunk;
        const std::size_t b = std::min(i1, a + chunk);
        run_polyphase_range(weights, wlen, up, down, base, bias, out, a, b);
    }
}
}  // namespace

namespace resample_impl {
struct polyphase {
    int up = 0;     // target / gcd  (number of output phases)
    int down = 0;   // source / gcd
    int lmax = 0;   // taps reaching back from the centre sample
    int rmax = 0;   // taps reaching forward
    float ratio = 1.0f;  // target/(float)source, exactly as AudioFlux computes the output length
    int wlen = 0;   // lmax + rmax, rounded up to a multiple of lanes
    std::vector<float> weight_store;  // over-allocated so rows can start 64-byte aligned
    const float* weights = nullptr;  // [up][wlen], 64-byte aligned, points into weight_store
};
}  // namespace resample_impl
using resample_impl::polyphase;

namespace {
// Process-wide cache of designed filter banks, keyed by everything the design depends on.
// A bank is ~100-250 KiB and costs ~0.6 ms to design, which is a large share of resampling
// a short clip; reusing it makes repeated Resample(src, dst) construction essentially free.
// Bounded by bytes (LRU eviction); banks larger than the budget are simply not cached.
struct bank_key {
    long long up, down;
    int zero_num;
    double beta, roll_off;
    bool operator==(const bank_key& o) const {
        return up == o.up && down == o.down && zero_num == o.zero_num && beta == o.beta &&
               roll_off == o.roll_off;
    }
};

class filter_bank_cache {
   public:
    std::shared_ptr<const polyphase> find(const bank_key& key) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto it = entries_.begin(); it != entries_.end(); ++it) {
            if (it->key == key) {
                entries_.splice(entries_.begin(), entries_, it);  // most recently used first
                return it->bank;
            }
        }
        return nullptr;
    }

    void insert(const bank_key& key, std::shared_ptr<const polyphase> bank, std::size_t bytes) {
        if (bytes > max_bytes) return;
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& e : entries_) {
            if (e.key == key) return;  // another thread designed it first
        }
        entries_.push_front({key, std::move(bank), bytes});
        total_bytes_ += bytes;
        while (total_bytes_ > max_bytes && entries_.size() > 1) {
            total_bytes_ -= entries_.back().bytes;
            entries_.pop_back();
        }
    }

   private:
    static constexpr std::size_t max_bytes = 32u << 20;  // 32 MiB
    struct entry {
        bank_key key;
        std::shared_ptr<const polyphase> bank;
        std::size_t bytes;
    };
    std::mutex mutex_;
    std::list<entry> entries_;
    std::size_t total_bytes_ = 0;
};

filter_bank_cache& bank_cache() {
    static filter_bank_cache cache;
    return cache;
}
}  // namespace

void resample::build_polyphase_from_tables() {
    poly_.reset();
    if (!resample_obj) return;
    const float* interp = nullptr;
    const float* delta = nullptr;
    int interp_length = 0, bit_length = 0, p = 0, q = 0, is_scale = 0, is_continue = 0;
    float ratio = 0.0f;
    if (resampleObj_getTables(resample_obj, &interp, &delta, &interp_length, &bit_length, &ratio, &p,
                              &q, &is_scale, &is_continue) != 0) {
        return;
    }
    if (is_scale || is_continue || p <= 0 || q <= 0) return;

    const float scale = std::min(1.0f, ratio);
    const int step = static_cast<int>(std::floor(scale * bit_length));
    if (step <= 0) return;
    // Bound the filter bank (p phases x ~2*interp_length/step taps): rates with a
    // huge reduced ratio (e.g. 44101 -> 16000) fall back to the direct kernel.
    const int approx_taps = 2 * (interp_length / step) + 2;
    if (static_cast<long long>(p) * approx_taps > (1 << 22)) return;

    struct tap_side { int offset; float delta; int len; };
    auto side = [&](float factor) {
        const float fv = factor * bit_length;
        const int off = static_cast<int>(std::floor(fv));
        return tap_side{off, fv - static_cast<float>(off), (interp_length - off) / step};
    };

    std::vector<tap_side> left(p), right(p);
    int lmax = 0, rmax = 0;
    for (int ph = 0; ph < p; ++ph) {
        const float factor = scale * static_cast<float>(static_cast<double>(ph) / p);
        left[ph] = side(factor);
        right[ph] = side(scale - factor);
        lmax = std::max(lmax, left[ph].len);
        rmax = std::max(rmax, right[ph].len);
    }
    if (lmax <= 0 || rmax <= 0) return;

    auto pl = std::make_unique<polyphase>();
    pl->up = p;
    pl->down = q;
    pl->ratio = ratio;
    pl->lmax = lmax;
    pl->rmax = rmax;
    pl->wlen = (lmax + rmax + lanes - 1) / lanes * lanes;
    pl->weight_store.assign(static_cast<std::size_t>(p) * pl->wlen + 16, 0.0f);
    {
        auto addr = reinterpret_cast<std::uintptr_t>(pl->weight_store.data());
        const std::size_t pad = ((64 - (addr % 64)) % 64) / sizeof(float);
        pl->weights = pl->weight_store.data() + pad;
    }
    for (int ph = 0; ph < p; ++ph) {
        float* row = const_cast<float*>(pl->weights) + static_cast<std::size_t>(ph) * pl->wlen;
        for (int j = 0; j < left[ph].len; ++j) {
            const int idx = left[ph].offset + j * step;
            row[lmax - 1 - j] = interp[idx] + left[ph].delta * delta[idx];
        }
        for (int j = 0; j < right[ph].len; ++j) {
            const int idx = right[ph].offset + j * step;
            row[lmax + j] = interp[idx] + right[ph].delta * delta[idx];
        }
    }
    poly_ = std::move(pl);
}

bool resample::build_polyphase_exact() {
    poly_.reset();
    if (params_.source_rate <= 0 || params_.target_rate <= 0 || params_.is_scale ||
        params_.is_continue || params_.win_type != window_type::kaiser) {
        return false;
    }
    const long long g = std::gcd(params_.source_rate, params_.target_rate);
    const long long up = params_.target_rate / g;    // output phases
    const long long down = params_.source_rate / g;
    const double ratio = static_cast<double>(up) / static_cast<double>(down);
    const double scale = std::min(1.0, ratio);       // low-pass cutoff scaling when downsampling
    const double gain = ratio < 1.0 ? ratio : 1.0;   // AudioFlux scales the kernel by ratio when ratio < 1
    const int zero_num = params_.zero_num;
    const int L = static_cast<int>(std::floor(static_cast<double>(zero_num) / scale)) + 1;  // taps per wing

    // Same size guard as the table-derived bank: huge reduced ratios (44101 -> 16000) fall
    // back to the direct kernel instead of allocating a multi-MB filter bank.
    if (up * 2LL * L > (1LL << 22) || L < 1) return false;

    const bank_key key{up, down, zero_num, static_cast<double>(params_.value),
                      static_cast<double>(params_.roll_off)};
    if (auto cached = bank_cache().find(key)) {
        poly_ = std::move(cached);
        return true;
    }

    // Closed-form kernel, sampled where the polyphase bank needs it. For an output at
    // fractional input position frac = r/up, the tap at input distance t reads
    //     h(t) = gain * T(scale * t),   T(x) = roll_off * sinc(roll_off * x) * Kaiser(x / zero_num)
    // (AudioFlux tabulated exactly this T and linearly interpolated it). All distances
    // t = m/up for integer m, so ONE array H[m] = h(m/up) feeds every tap of every row:
    // left wing  row[r][L-1-j] = H[r + up*j],  right wing  row[r][L+j] = H[(up-r) + up*j].
    const std::size_t Hn = static_cast<std::size_t>(up) * L + 1;
    std::vector<float> H(Hn);

    constexpr double pi = 3.14159265358979323846;
    const double beta = params_.value;
    const double roll_off = params_.roll_off;
    const double w = pi * roll_off * scale / static_cast<double>(up);  // sin(w*m) = sin(pi*roll_off*x)
    const double cw2 = 2.0 * std::cos(w);

    // I0(z) = sum_k (z/2)^(2k) / (k!)^2, in double. Terms peak near k ~ beta/2 and have
    // died below 1e-17 relative by k ~ 1.6*beta + 12.
    const int K = std::max(12, static_cast<int>(std::ceil(1.6 * beta)) + 12);
    std::vector<double> inv_k2(static_cast<std::size_t>(K) + 1);
    for (int k = 1; k <= K; ++k) inv_k2[k] = 1.0 / (static_cast<double>(k) * k);
    const double q0 = 0.25 * beta * beta;
    double i0_beta = 1.0, term0 = 1.0;
    for (int k = 1; k <= K; ++k) {
        term0 *= q0 * inv_k2[k];
        i0_beta += term0;
    }
    const double inv_i0_beta = 1.0 / i0_beta;

    constexpr int block_size = 256;  // block of m processed together: vectorizes, stays in L1
    double sn[block_size], qv[block_size], term[block_size], sum[block_size], xv[block_size];
    for (std::size_t m0 = 0; m0 < Hn; m0 += block_size) {
        const int cnt = static_cast<int>(std::min<std::size_t>(block_size, Hn - m0));

        // sin(w*m) by the Chebyshev recurrence, re-seeded from libm every block so
        // rounding error cannot accumulate across the whole array.
        double s_prev = std::sin(w * (static_cast<double>(m0) - 1.0));
        double s_cur = std::sin(w * static_cast<double>(m0));
        for (int i = 0; i < cnt; ++i) {
            sn[i] = s_cur;
            const double next = cw2 * s_cur - s_prev;
            s_prev = s_cur;
            s_cur = next;
        }

        for (int i = 0; i < cnt; ++i) {
            const double x = scale * static_cast<double>(m0 + i) / static_cast<double>(up);
            xv[i] = x;
            const double v = x / static_cast<double>(zero_num);
            qv[i] = v < 1.0 ? q0 * (1.0 - v * v) : 0.0;
            term[i] = 1.0;
            sum[i] = 1.0;
        }
        for (int k = 1; k <= K; ++k) {
            const double ik = inv_k2[k];
            for (int i = 0; i < cnt; ++i) {
                term[i] *= qv[i] * ik;
                sum[i] += term[i];
            }
        }
        for (int i = 0; i < cnt; ++i) {
            const double x = xv[i];
            double h = 0.0;
            if (x <= static_cast<double>(zero_num)) {
                const double t = (m0 + i == 0) ? roll_off : sn[i] / (pi * x);
                h = gain * t * sum[i] * inv_i0_beta;
            }
            H[m0 + i] = static_cast<float>(h);
        }
    }

    auto pl = std::make_unique<polyphase>();
    pl->up = static_cast<int>(up);
    pl->down = static_cast<int>(down);
    pl->ratio = static_cast<float>(params_.target_rate) / static_cast<float>(params_.source_rate);
    pl->lmax = L;
    pl->rmax = L;
    pl->wlen = (2 * L + lanes - 1) / lanes * lanes;
    pl->weight_store.assign(static_cast<std::size_t>(up) * pl->wlen + 16, 0.0f);
    {
        auto addr = reinterpret_cast<std::uintptr_t>(pl->weight_store.data());
        const std::size_t pad = ((64 - (addr % 64)) % 64) / sizeof(float);
        pl->weights = pl->weight_store.data() + pad;
    }
    for (long long r = 0; r < up; ++r) {
        float* row = const_cast<float*>(pl->weights) + static_cast<std::size_t>(r) * pl->wlen;
        for (int j = 0; j < L; ++j) {
            row[L - 1 - j] = H[static_cast<std::size_t>(r + up * j)];
            row[L + j] = H[static_cast<std::size_t>((up - r) + up * j)];
        }
    }
    const std::size_t bytes = pl->weight_store.size() * sizeof(float);
    std::shared_ptr<const polyphase> shared = std::move(pl);
    bank_cache().insert(key, shared, bytes);
    poly_ = std::move(shared);
    return true;
}

detail::uninit_vector<float> resample::resample_polyphase(const float* data, std::size_t n) {
    const polyphase& pl = *poly_;
    // Same length AudioFlux reports for non-continuous mode: floorf(n * ratio) in float.
    const int out_len_int = static_cast<int>(std::floor(static_cast<float>(static_cast<int>(n)) * pl.ratio));
    if (out_len_int <= 0) {
        throw std::runtime_error("Resampling failed or produced no output.");
    }
    const std::size_t out_len = static_cast<std::size_t>(out_len_int);
    detail::uninit_vector<float> out(out_len);

    // Output i reads inputs [n_i - lead, n_i - lead + wlen), n_i = floor(i*down/up).
    // Where that window lies inside [0, n) the kernel reads the caller's samples in
    // place (no padded copy of the whole signal); only the few outputs near either
    // end see out-of-range samples (zeros) and go through a small padded buffer.
    const long long lead = pl.lmax - 1;
    const long long wlen = pl.wlen;
    const long long up = pl.up;
    const long long down = pl.down;
    const long long len = static_cast<long long>(n);

    auto ceil_div = [](long long a, long long b) { return a <= 0 ? 0LL : (a + b - 1) / b; };
    // interior: n_i >= lead  and  n_i + wlen - lead <= n
    std::size_t i_lo = static_cast<std::size_t>(std::min<long long>(ceil_div(lead * up, down),
                                                                   static_cast<long long>(out_len)));
    std::size_t i_hi = static_cast<std::size_t>(std::min<long long>(
        ceil_div((len - wlen + lead + 1) * up, down), static_cast<long long>(out_len)));
    if (i_hi < i_lo) i_hi = i_lo = 0;  // signal shorter than the filter: everything is "edge"
    if (i_hi == i_lo) i_hi = i_lo = 0;

    run_polyphase(pl.weights, pl.wlen, pl.up, pl.down, data, static_cast<std::ptrdiff_t>(lead), out.data(),
                 i_lo, i_hi);

    auto run_edge = [&](std::size_t a, std::size_t b) {
        if (a >= b) return;
        const long long first = static_cast<long long>(a) * down / up - lead;
        const long long last = (static_cast<long long>(b - 1) * down / up) - lead + wlen;  // exclusive
        std::vector<float> buf(static_cast<std::size_t>(last - first), 0.0f);
        const long long from = std::max<long long>(first, 0);
        const long long to = std::min<long long>(last, len);
        if (to > from) {
            std::copy(data + from, data + to, buf.begin() + (from - first));
        }
        // base[n_i - bias] must be buf[n_i - lead - first]  =>  bias = lead + first
        run_polyphase(pl.weights, pl.wlen, pl.up, pl.down, buf.data(),
                     static_cast<std::ptrdiff_t>(lead + first), out.data(), a, b);
    };
    run_edge(0, i_lo);
    run_edge(i_hi, out_len);
    return out;
}

resample::resample(int source_rate, int target_rate)
    : resample_obj(nullptr), is_identity(source_rate == target_rate) {
    params_.source_rate = source_rate;
    params_.target_rate = target_rate;
    // AudioFlux "Best" preset (Kaiser; zero_num 64, beta 14.7697, roll-off 0.9476).
    // is_scale=0: AudioFlux's is_scale=1 divides the output by sqrt(ratio), an
    // energy-domain normalization (see resampleObj_resample()). For a
    // downsample from 44100->16000 that means dividing by ~0.60, i.e.
    // amplifying the waveform by ~1.66x -- which can push an already
    // near-full-scale signal well past +/-1.0. We want amplitude-preserving
    // resampling by default for a general audio pipeline (feeding VAD,
    // denoisers, playback, etc.); advanced users who want the energy-scaled
    // behavior can use the extended constructor with is_scale=true.
    // Full write-up: /audioflux_issues.md (Issue 2).
    if (is_identity) {
        SPEECH_LOG(DEBUG) << TAG(log_tag) << "Source/target rate both " << source_rate
                   << "Hz; using identity passthrough (skipping AudioFlux)." << std::endl;
        return;
    }

    if (!build_polyphase_exact()) {  // not reachable for the preset except for absurd rate ratios
        ensure_audio_flux();
        build_polyphase_from_tables();
    }
    SPEECH_LOG(DEBUG) << TAG(log_tag) << "Created resampler: " << source_rate << "Hz -> " << target_rate
               << "Hz." << std::endl;
}

resample::resample(int source_rate, int target_rate, resample_quality quality)
    : resample(source_rate, target_rate, params_for(quality).zero_num, /*nbit=*/0, window_type::kaiser,
               params_for(quality).beta, params_for(quality).roll_off, /*is_scale=*/false,
               /*is_continue=*/false) {}

resample::resample(int source_rate, int target_rate, int zero_num, int nbit, window_type win_type,
                     float value, float roll_off, bool is_scale, bool is_continue)
    : resample_obj(nullptr), is_identity(source_rate == target_rate) {
    // Normalize exactly like resampleObj_newWithWindow() so the exact design and the
    // (lazily created) AudioFlux object agree on every parameter.
    params_.source_rate = source_rate;
    params_.target_rate = target_rate;
    params_.custom_window = true;
    params_.zero_num = zero_num > 0 ? zero_num : 64;
    params_.nbit = (nbit > 0 && nbit < 30) ? nbit : 9;
    params_.win_type = win_type > window_type::rect ? win_type : window_type::hann;
    float beta = value >= 0 ? value : 0.0f;
    if (beta == 0.0f) {
        if (params_.win_type == window_type::kaiser) beta = 5.0f;
        else if (params_.win_type == window_type::gauss) beta = 2.5f;
    }
    params_.value = beta;
    params_.roll_off = (roll_off > 0 && roll_off <= 1) ? roll_off : 0.945f;
    params_.is_scale = is_scale;
    params_.is_continue = is_continue;

    if (is_identity) {
        SPEECH_LOG(DEBUG) << TAG(log_tag) << "Source/target rate both " << source_rate
                   << "Hz; using identity passthrough (skipping AudioFlux)." << std::endl;
        return;
    }

    if (!build_polyphase_exact()) {
        ensure_audio_flux();           // continuous / is_scale / non-Kaiser window: AudioFlux's tables
        build_polyphase_from_tables();
    }
    SPEECH_LOG(DEBUG) << TAG(log_tag) << "Created resampler (custom window): " << source_rate << "Hz -> "
               << target_rate << "Hz." << std::endl;
}

void resample::ensure_audio_flux() {
    if (resample_obj || is_identity) {
        return;
    }
    int is_scale = params_.is_scale ? 1 : 0;
    int is_continue = params_.is_continue ? 1 : 0;
    int rc;
    if (!params_.custom_window) {
        ResampleQualityType qual_type = ResampleQuality_Best;
        rc = resampleObj_new(&resample_obj, &qual_type, &is_scale, &is_continue);
    } else {
        int zero_num = params_.zero_num;
        int nbit = params_.nbit;
        ::WindowType audioflux_win_type = to_audioflux(params_.win_type);
        float value = params_.value;
        float roll_off = params_.roll_off;
        rc = resampleObj_newWithWindow(&resample_obj, &zero_num, &nbit, &audioflux_win_type, &value, &roll_off,
                                       &is_scale, &is_continue);
    }
    if (rc != 0) {
        SPEECH_LOG(ERROR) << TAG(log_tag) << "Failed to create resample object (" << params_.source_rate
                   << "Hz -> " << params_.target_rate << "Hz)." << std::endl;
        throw std::runtime_error("Failed to create resample object.");
    }
    resampleObj_setSamplate(resample_obj, params_.source_rate, params_.target_rate);
}

resample::~resample() {
    reset();
}

void resample::set_sample_rate_ratio(float ratio) {
    if (is_identity) {
        SPEECH_LOG(DEBUG) << TAG(log_tag)
                   << "set_sample_rate_ratio() ignored: resampler is in identity passthrough mode."
                   << std::endl;
        return;
    }
    ensure_audio_flux();
    poly_.reset();  // arbitrary float ratio: no rational phase structure
    resampleObj_setSamplateRatio(resample_obj, ratio);
    SPEECH_LOG(DEBUG) << TAG(log_tag) << "Sample rate ratio set to " << ratio << std::endl;
}

void resample::enable_continuous(bool flag) {
    if (is_identity) {
        return;
    }
    ensure_audio_flux();
    poly_.reset();  // continuous mode carries tail state through the direct kernel
    resampleObj_enableContinue(resample_obj, flag ? 1 : 0);
}

detail::uninit_vector<float> resample::resample_flat(const float* data, std::size_t n) {
    if (is_identity) {
        return detail::uninit_vector<float>(data, data + n);
    }
    if (n == 0) {
        SPEECH_LOG(DEBUG) << TAG(log_tag) << "resample() called with empty input; returning empty output."
                   << std::endl;
        return {};
    }

    if (poly_) {
        auto out = resample_polyphase(data, n);
        SPEECH_LOG(DEBUG) << TAG(log_tag) << "Resampled " << n << " -> " << out.size()
                   << " samples (polyphase)." << std::endl;
        return out;
    }

    ensure_audio_flux();
    const int input_length = static_cast<int>(n);
    const int output_length = resampleObj_calDataLength(resample_obj, input_length);
    SPEECH_LOG(TRACE) << TAG(log_tag) << "cal_data_length(" << input_length << ") -> expected output length "
               << output_length << std::endl;

    // The direct AudioFlux kernel ACCUMULATES into its output (data_arr2[i] += ...),
    // so this buffer must start zeroed -- unlike the polyphase path above, which
    // overwrites every sample.
    detail::uninit_vector<float> out(static_cast<std::size_t>(output_length));
    std::fill(out.begin(), out.end(), 0.0f);
    const int actual = resampleObj_resample(resample_obj, const_cast<float*>(data), input_length,
                                            out.data());
    if (actual <= 0) {
        SPEECH_LOG(ERROR) << TAG(log_tag) << "Resampling produced no output (input length=" << input_length
                   << ")." << std::endl;
        throw std::runtime_error("Resampling failed or produced no output.");
    }
    out.resize(static_cast<std::size_t>(actual));
    SPEECH_LOG(DEBUG) << TAG(log_tag) << "Resampled " << input_length << " -> " << actual << " samples."
               << std::endl;
    return out;
}

std::vector<float> resample::process(const std::vector<float>& input_data) {
    auto flat = resample_flat(input_data.data(), input_data.size());
    return std::vector<float>(flat.begin(), flat.end());
}

void resample::reset() {
    if (resample_obj) {
        resampleObj_free(resample_obj);
        resample_obj = nullptr;
    }
}

}  // namespace speech::dsp
