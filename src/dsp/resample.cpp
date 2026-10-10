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

namespace speech::dsp {

namespace {
constexpr const char* kTag = "speech::dsp::Resample";

// Polyphase kernels. Output i uses weight row (i*down) mod up and the input
// window starting at n_i = floor(i*down/up): exact integer arithmetic, no float
// time accumulation. Rows are kLanes-padded with zeros.
constexpr int kLanes = 8;

// All kernels compute outputs i in [i0, i1) into out[i]. Output i reads the wlen
// inputs base[n_i - bias + k], k in [0, wlen), with n_i = floor(i*down/up); the
// caller guarantees those indices are valid (the interior of the signal, or a
// small zero-padded edge buffer).
//
// Portable kernel: 8 independent accumulators, vectorizes without -ffast-math
// because the lanes never mix until the final reduction.
void runPolyphaseGeneric(const float* weights, int wlen, int up, int down, const float* base,
                         std::ptrdiff_t bias, float* out, std::size_t i0, std::size_t i1) {
    const long long start = static_cast<long long>(i0) * down;
    long long n = start / up;
    int ph = static_cast<int>(start % up);
    const int stepN = down / up;
    const int stepP = down % up;
    for (std::size_t i = i0; i < i1; ++i) {
        const float* w = weights + static_cast<std::size_t>(ph) * wlen;
        const float* x = base + (static_cast<std::ptrdiff_t>(n) - bias);
        float acc[kLanes] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int k = 0; k < wlen; k += kLanes) {
            for (int l = 0; l < kLanes; ++l) acc[l] += w[k + l] * x[k + l];
        }
        out[i] = ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]));
        n += stepN;
        ph += stepP;
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
LS_TARGET_AVX2 void runPolyphaseAvx2(const float* weights, int wlen, int up, int down,
                                     const float* base, std::ptrdiff_t bias, float* out,
                                     std::size_t i0, std::size_t i1) {
    const std::size_t d = static_cast<std::size_t>(down);
    const std::size_t U = static_cast<std::size_t>(up);
    const std::size_t mLo = i0 / U;
    const std::size_t mHi = (i1 + U - 1) / U;

    for (std::size_t m0 = mLo; m0 < mHi; m0 += 8) {
        for (int ph = 0; ph < up; ++ph) {
            const long long phDown = static_cast<long long>(ph) * down;
            const float* w = weights + static_cast<std::size_t>(phDown % up) * wlen;
            const std::size_t n0 = static_cast<std::size_t>(phDown / up) + m0 * d;
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
void runPolyphaseNeon(const float* weights, int wlen, int up, int down, const float* base,
                      std::ptrdiff_t bias, float* out, std::size_t i0, std::size_t i1) {
    const std::size_t d = static_cast<std::size_t>(down);
    const std::size_t U = static_cast<std::size_t>(up);
    const std::size_t mLo = i0 / U;
    const std::size_t mHi = (i1 + U - 1) / U;

    for (std::size_t m0 = mLo; m0 < mHi; m0 += 8) {
        for (int ph = 0; ph < up; ++ph) {
            const long long phDown = static_cast<long long>(ph) * down;
            const float* w = weights + static_cast<std::size_t>(phDown % up) * wlen;
            const std::size_t n0 = static_cast<std::size_t>(phDown / up) + m0 * d;
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
void runPolyphaseRange(const float* weights, int wlen, int up, int down, const float* base,
                       std::ptrdiff_t bias, float* out, std::size_t i0, std::size_t i1) {
    if (i0 >= i1) return;
#if LS_HAVE_X86_AVX2
    if (simd::useAvx2Fma()) {
        runPolyphaseAvx2(weights, wlen, up, down, base, bias, out, i0, i1);
        return;
    }
#endif
#if LS_HAVE_ARM_NEON
    if (simd::useNeon()) {
        runPolyphaseNeon(weights, wlen, up, down, base, bias, out, i0, i1);
        return;
    }
#endif
    runPolyphaseGeneric(weights, wlen, up, down, base, bias, out, i0, i1);
}

// Outputs are independent, so long signals are split into chunks of whole 8-m groups
// (each kernel call writes only its own [a, b) range) and run on all cores when OpenMP
// is available. Short inputs stay on one thread: thread start-up would cost more than
// the work.
void runPolyphase(const float* weights, int wlen, int up, int down, const float* base,
                  std::ptrdiff_t bias, float* out, std::size_t i0, std::size_t i1) {
    if (i0 >= i1) return;
    const std::size_t chunk = static_cast<std::size_t>(up) * 8 * 16;  // 128 m-values per task
    const std::size_t span = i1 - i0;
    if (span < 4 * chunk) {
        runPolyphaseRange(weights, wlen, up, down, base, bias, out, i0, i1);
        return;
    }
    const long long tasks = static_cast<long long>((span + chunk - 1) / chunk);
#pragma omp parallel for schedule(static)
    for (long long t = 0; t < tasks; ++t) {
        const std::size_t a = i0 + static_cast<std::size_t>(t) * chunk;
        const std::size_t b = std::min(i1, a + chunk);
        runPolyphaseRange(weights, wlen, up, down, base, bias, out, a, b);
    }
}
}  // namespace

namespace resample_impl {
struct Polyphase {
    int up = 0;     // target / gcd  (number of output phases)
    int down = 0;   // source / gcd
    int lmax = 0;   // taps reaching back from the centre sample
    int rmax = 0;   // taps reaching forward
    float ratio = 1.0f;  // target/(float)source, exactly as AudioFlux computes the output length
    int wlen = 0;   // lmax + rmax, rounded up to a multiple of kLanes
    std::vector<float> weightStore;  // over-allocated so rows can start 64-byte aligned
    const float* weights = nullptr;  // [up][wlen], 64-byte aligned, points into weightStore
};
}  // namespace resample_impl
using resample_impl::Polyphase;

namespace {
// Process-wide cache of designed filter banks, keyed by everything the design depends on.
// A bank is ~100-250 KiB and costs ~0.6 ms to design, which is a large share of resampling
// a short clip; reusing it makes repeated Resample(src, dst) construction essentially free.
// Bounded by bytes (LRU eviction); banks larger than the budget are simply not cached.
struct BankKey {
    long long up, down;
    int zeroNum;
    double beta, rollOff;
    bool operator==(const BankKey& o) const {
        return up == o.up && down == o.down && zeroNum == o.zeroNum && beta == o.beta &&
               rollOff == o.rollOff;
    }
};

class BankCache {
   public:
    std::shared_ptr<const Polyphase> find(const BankKey& key) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto it = entries_.begin(); it != entries_.end(); ++it) {
            if (it->key == key) {
                entries_.splice(entries_.begin(), entries_, it);  // most recently used first
                return it->bank;
            }
        }
        return nullptr;
    }

    void insert(const BankKey& key, std::shared_ptr<const Polyphase> bank, std::size_t bytes) {
        if (bytes > kMaxBytes) return;
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& e : entries_) {
            if (e.key == key) return;  // another thread designed it first
        }
        entries_.push_front({key, std::move(bank), bytes});
        totalBytes_ += bytes;
        while (totalBytes_ > kMaxBytes && entries_.size() > 1) {
            totalBytes_ -= entries_.back().bytes;
            entries_.pop_back();
        }
    }

   private:
    static constexpr std::size_t kMaxBytes = 32u << 20;  // 32 MiB
    struct Entry {
        BankKey key;
        std::shared_ptr<const Polyphase> bank;
        std::size_t bytes;
    };
    std::mutex mutex_;
    std::list<Entry> entries_;
    std::size_t totalBytes_ = 0;
};

BankCache& bankCache() {
    static BankCache cache;
    return cache;
}
}  // namespace

void Resample::buildPolyphaseFromTables() {
    poly_.reset();
    if (!resampleObj) return;
    const float* interp = nullptr;
    const float* delta = nullptr;
    int interpLength = 0, bitLength = 0, p = 0, q = 0, isScale = 0, isContinue = 0;
    float ratio = 0.0f;
    if (resampleObj_getTables(resampleObj, &interp, &delta, &interpLength, &bitLength, &ratio, &p,
                              &q, &isScale, &isContinue) != 0) {
        return;
    }
    if (isScale || isContinue || p <= 0 || q <= 0) return;

    const float scale = std::min(1.0f, ratio);
    const int step = static_cast<int>(std::floor(scale * bitLength));
    if (step <= 0) return;
    // Bound the filter bank (p phases x ~2*interpLength/step taps): rates with a
    // huge reduced ratio (e.g. 44101 -> 16000) fall back to the direct kernel.
    const int approxTaps = 2 * (interpLength / step) + 2;
    if (static_cast<long long>(p) * approxTaps > (1 << 22)) return;

    struct Side { int offset; float delta; int len; };
    auto side = [&](float factor) {
        const float fv = factor * bitLength;
        const int off = static_cast<int>(std::floor(fv));
        return Side{off, fv - static_cast<float>(off), (interpLength - off) / step};
    };

    std::vector<Side> left(p), right(p);
    int lmax = 0, rmax = 0;
    for (int ph = 0; ph < p; ++ph) {
        const float factor = scale * static_cast<float>(static_cast<double>(ph) / p);
        left[ph] = side(factor);
        right[ph] = side(scale - factor);
        lmax = std::max(lmax, left[ph].len);
        rmax = std::max(rmax, right[ph].len);
    }
    if (lmax <= 0 || rmax <= 0) return;

    auto pl = std::make_unique<Polyphase>();
    pl->up = p;
    pl->down = q;
    pl->ratio = ratio;
    pl->lmax = lmax;
    pl->rmax = rmax;
    pl->wlen = (lmax + rmax + kLanes - 1) / kLanes * kLanes;
    pl->weightStore.assign(static_cast<std::size_t>(p) * pl->wlen + 16, 0.0f);
    {
        auto addr = reinterpret_cast<std::uintptr_t>(pl->weightStore.data());
        const std::size_t pad = ((64 - (addr % 64)) % 64) / sizeof(float);
        pl->weights = pl->weightStore.data() + pad;
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

bool Resample::buildPolyphaseExact() {
    poly_.reset();
    if (params_.sourceRate <= 0 || params_.targetRate <= 0 || params_.isScale ||
        params_.isContinue || params_.winType != Window_Kaiser) {
        return false;
    }
    const long long g = std::gcd(params_.sourceRate, params_.targetRate);
    const long long up = params_.targetRate / g;    // output phases
    const long long down = params_.sourceRate / g;
    const double ratio = static_cast<double>(up) / static_cast<double>(down);
    const double scale = std::min(1.0, ratio);       // low-pass cutoff scaling when downsampling
    const double gain = ratio < 1.0 ? ratio : 1.0;   // AudioFlux scales the kernel by ratio when ratio < 1
    const int zeroNum = params_.zeroNum;
    const int L = static_cast<int>(std::floor(static_cast<double>(zeroNum) / scale)) + 1;  // taps per wing

    // Same size guard as the table-derived bank: huge reduced ratios (44101 -> 16000) fall
    // back to the direct kernel instead of allocating a multi-MB filter bank.
    if (up * 2LL * L > (1LL << 22) || L < 1) return false;

    const BankKey key{up, down, zeroNum, static_cast<double>(params_.value),
                      static_cast<double>(params_.rollOff)};
    if (auto cached = bankCache().find(key)) {
        poly_ = std::move(cached);
        return true;
    }

    // Closed-form kernel, sampled where the polyphase bank needs it. For an output at
    // fractional input position frac = r/up, the tap at input distance t reads
    //     h(t) = gain * T(scale * t),   T(x) = rollOff * sinc(rollOff * x) * Kaiser(x / zeroNum)
    // (AudioFlux tabulated exactly this T and linearly interpolated it). All distances
    // t = m/up for integer m, so ONE array H[m] = h(m/up) feeds every tap of every row:
    // left wing  row[r][L-1-j] = H[r + up*j],  right wing  row[r][L+j] = H[(up-r) + up*j].
    const std::size_t Hn = static_cast<std::size_t>(up) * L + 1;
    std::vector<float> H(Hn);

    constexpr double kPi = 3.14159265358979323846;
    const double beta = params_.value;
    const double rollOff = params_.rollOff;
    const double w = kPi * rollOff * scale / static_cast<double>(up);  // sin(w*m) = sin(pi*rollOff*x)
    const double cw2 = 2.0 * std::cos(w);

    // I0(z) = sum_k (z/2)^(2k) / (k!)^2, in double. Terms peak near k ~ beta/2 and have
    // died below 1e-17 relative by k ~ 1.6*beta + 12.
    const int K = std::max(12, static_cast<int>(std::ceil(1.6 * beta)) + 12);
    std::vector<double> invK2(static_cast<std::size_t>(K) + 1);
    for (int k = 1; k <= K; ++k) invK2[k] = 1.0 / (static_cast<double>(k) * k);
    const double q0 = 0.25 * beta * beta;
    double i0Beta = 1.0, term0 = 1.0;
    for (int k = 1; k <= K; ++k) {
        term0 *= q0 * invK2[k];
        i0Beta += term0;
    }
    const double invI0Beta = 1.0 / i0Beta;

    constexpr int kBlock = 256;  // block of m processed together: vectorizes, stays in L1
    double sn[kBlock], qv[kBlock], term[kBlock], sum[kBlock], xv[kBlock];
    for (std::size_t m0 = 0; m0 < Hn; m0 += kBlock) {
        const int cnt = static_cast<int>(std::min<std::size_t>(kBlock, Hn - m0));

        // sin(w*m) by the Chebyshev recurrence, re-seeded from libm every block so
        // rounding error cannot accumulate across the whole array.
        double sPrev = std::sin(w * (static_cast<double>(m0) - 1.0));
        double sCur = std::sin(w * static_cast<double>(m0));
        for (int i = 0; i < cnt; ++i) {
            sn[i] = sCur;
            const double next = cw2 * sCur - sPrev;
            sPrev = sCur;
            sCur = next;
        }

        for (int i = 0; i < cnt; ++i) {
            const double x = scale * static_cast<double>(m0 + i) / static_cast<double>(up);
            xv[i] = x;
            const double v = x / static_cast<double>(zeroNum);
            qv[i] = v < 1.0 ? q0 * (1.0 - v * v) : 0.0;
            term[i] = 1.0;
            sum[i] = 1.0;
        }
        for (int k = 1; k <= K; ++k) {
            const double ik = invK2[k];
            for (int i = 0; i < cnt; ++i) {
                term[i] *= qv[i] * ik;
                sum[i] += term[i];
            }
        }
        for (int i = 0; i < cnt; ++i) {
            const double x = xv[i];
            double h = 0.0;
            if (x <= static_cast<double>(zeroNum)) {
                const double t = (m0 + i == 0) ? rollOff : sn[i] / (kPi * x);
                h = gain * t * sum[i] * invI0Beta;
            }
            H[m0 + i] = static_cast<float>(h);
        }
    }

    auto pl = std::make_unique<Polyphase>();
    pl->up = static_cast<int>(up);
    pl->down = static_cast<int>(down);
    pl->ratio = static_cast<float>(params_.targetRate) / static_cast<float>(params_.sourceRate);
    pl->lmax = L;
    pl->rmax = L;
    pl->wlen = (2 * L + kLanes - 1) / kLanes * kLanes;
    pl->weightStore.assign(static_cast<std::size_t>(up) * pl->wlen + 16, 0.0f);
    {
        auto addr = reinterpret_cast<std::uintptr_t>(pl->weightStore.data());
        const std::size_t pad = ((64 - (addr % 64)) % 64) / sizeof(float);
        pl->weights = pl->weightStore.data() + pad;
    }
    for (long long r = 0; r < up; ++r) {
        float* row = const_cast<float*>(pl->weights) + static_cast<std::size_t>(r) * pl->wlen;
        for (int j = 0; j < L; ++j) {
            row[L - 1 - j] = H[static_cast<std::size_t>(r + up * j)];
            row[L + j] = H[static_cast<std::size_t>((up - r) + up * j)];
        }
    }
    const std::size_t bytes = pl->weightStore.size() * sizeof(float);
    std::shared_ptr<const Polyphase> shared = std::move(pl);
    bankCache().insert(key, shared, bytes);
    poly_ = std::move(shared);
    return true;
}

detail::UninitVector<float> Resample::resamplePolyphase(const float* data, std::size_t n) {
    const Polyphase& pl = *poly_;
    // Same length AudioFlux reports for non-continuous mode: floorf(n * ratio) in float.
    const int outLenInt = static_cast<int>(std::floor(static_cast<float>(static_cast<int>(n)) * pl.ratio));
    if (outLenInt <= 0) {
        throw std::runtime_error("Resampling failed or produced no output.");
    }
    const std::size_t outLen = static_cast<std::size_t>(outLenInt);
    detail::UninitVector<float> out(outLen);

    // Output i reads inputs [n_i - lead, n_i - lead + wlen), n_i = floor(i*down/up).
    // Where that window lies inside [0, n) the kernel reads the caller's samples in
    // place (no padded copy of the whole signal); only the few outputs near either
    // end see out-of-range samples (zeros) and go through a small padded buffer.
    const long long lead = pl.lmax - 1;
    const long long wlen = pl.wlen;
    const long long up = pl.up;
    const long long down = pl.down;
    const long long len = static_cast<long long>(n);

    auto ceilDiv = [](long long a, long long b) { return a <= 0 ? 0LL : (a + b - 1) / b; };
    // interior: n_i >= lead  and  n_i + wlen - lead <= n
    std::size_t iLo = static_cast<std::size_t>(std::min<long long>(ceilDiv(lead * up, down),
                                                                   static_cast<long long>(outLen)));
    std::size_t iHi = static_cast<std::size_t>(std::min<long long>(
        ceilDiv((len - wlen + lead + 1) * up, down), static_cast<long long>(outLen)));
    if (iHi < iLo) iHi = iLo = 0;  // signal shorter than the filter: everything is "edge"
    if (iHi == iLo) iHi = iLo = 0;

    runPolyphase(pl.weights, pl.wlen, pl.up, pl.down, data, static_cast<std::ptrdiff_t>(lead), out.data(),
                 iLo, iHi);

    auto runEdge = [&](std::size_t a, std::size_t b) {
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
        runPolyphase(pl.weights, pl.wlen, pl.up, pl.down, buf.data(),
                     static_cast<std::ptrdiff_t>(lead + first), out.data(), a, b);
    };
    runEdge(0, iLo);
    runEdge(iHi, outLen);
    return out;
}

Resample::Resample(int sourceRate, int targetRate)
    : resampleObj(nullptr), isIdentity(sourceRate == targetRate) {
    params_.sourceRate = sourceRate;
    params_.targetRate = targetRate;
    // AudioFlux "Best" preset (Kaiser; zeroNum 64, beta 14.7697, roll-off 0.9476).
    // isScale=0: AudioFlux's isScale=1 divides the output by sqrt(ratio), an
    // energy-domain normalization (see resampleObj_resample()). For a
    // downsample from 44100->16000 that means dividing by ~0.60, i.e.
    // amplifying the waveform by ~1.66x -- which can push an already
    // near-full-scale signal well past +/-1.0. We want amplitude-preserving
    // resampling by default for a general audio pipeline (feeding VAD,
    // denoisers, playback, etc.); advanced users who want the energy-scaled
    // behavior can use the extended constructor with isScale=true.
    // Full write-up: /audioflux_issues.md (Issue 2).
    if (isIdentity) {
        SPEECH_LOG(DEBUG) << TAG(kTag) << "Source/target rate both " << sourceRate
                   << "Hz; using identity passthrough (skipping AudioFlux)." << std::endl;
        return;
    }

    if (!buildPolyphaseExact()) {  // not reachable for the preset except for absurd rate ratios
        ensureAudioFlux();
        buildPolyphaseFromTables();
    }
    SPEECH_LOG(DEBUG) << TAG(kTag) << "Created resampler: " << sourceRate << "Hz -> " << targetRate
               << "Hz." << std::endl;
}

Resample::Resample(int sourceRate, int targetRate, int zeroNum, int nbit, WindowType winType,
                     float value, float rollOff, bool isScale, bool isContinue)
    : resampleObj(nullptr), isIdentity(sourceRate == targetRate) {
    // Normalize exactly like resampleObj_newWithWindow() so the exact design and the
    // (lazily created) AudioFlux object agree on every parameter.
    params_.sourceRate = sourceRate;
    params_.targetRate = targetRate;
    params_.customWindow = true;
    params_.zeroNum = zeroNum > 0 ? zeroNum : 64;
    params_.nbit = (nbit > 0 && nbit < 30) ? nbit : 9;
    params_.winType = winType > Window_Rect ? winType : Window_Hann;
    float beta = value >= 0 ? value : 0.0f;
    if (beta == 0.0f) {
        if (params_.winType == Window_Kaiser) beta = 5.0f;
        else if (params_.winType == Window_Gauss) beta = 2.5f;
    }
    params_.value = beta;
    params_.rollOff = (rollOff > 0 && rollOff <= 1) ? rollOff : 0.945f;
    params_.isScale = isScale;
    params_.isContinue = isContinue;

    if (isIdentity) {
        SPEECH_LOG(DEBUG) << TAG(kTag) << "Source/target rate both " << sourceRate
                   << "Hz; using identity passthrough (skipping AudioFlux)." << std::endl;
        return;
    }

    if (!buildPolyphaseExact()) {
        ensureAudioFlux();           // continuous / isScale / non-Kaiser window: AudioFlux's tables
        buildPolyphaseFromTables();
    }
    SPEECH_LOG(DEBUG) << TAG(kTag) << "Created resampler (custom window): " << sourceRate << "Hz -> "
               << targetRate << "Hz." << std::endl;
}

void Resample::ensureAudioFlux() {
    if (resampleObj || isIdentity) {
        return;
    }
    int isScale = params_.isScale ? 1 : 0;
    int isContinue = params_.isContinue ? 1 : 0;
    int rc;
    if (!params_.customWindow) {
        ResampleQualityType qualType = ResampleQuality_Best;
        rc = resampleObj_new(&resampleObj, &qualType, &isScale, &isContinue);
    } else {
        int zeroNum = params_.zeroNum;
        int nbit = params_.nbit;
        WindowType winType = params_.winType;
        float value = params_.value;
        float rollOff = params_.rollOff;
        rc = resampleObj_newWithWindow(&resampleObj, &zeroNum, &nbit, &winType, &value, &rollOff,
                                       &isScale, &isContinue);
    }
    if (rc != 0) {
        SPEECH_LOG(ERROR) << TAG(kTag) << "Failed to create resample object (" << params_.sourceRate
                   << "Hz -> " << params_.targetRate << "Hz)." << std::endl;
        throw std::runtime_error("Failed to create resample object.");
    }
    resampleObj_setSamplate(resampleObj, params_.sourceRate, params_.targetRate);
}

Resample::~Resample() {
    reset();
}

void Resample::setSampleRateRatio(float ratio) {
    if (isIdentity) {
        SPEECH_LOG(DEBUG) << TAG(kTag)
                   << "setSampleRateRatio() ignored: resampler is in identity passthrough mode."
                   << std::endl;
        return;
    }
    ensureAudioFlux();
    poly_.reset();  // arbitrary float ratio: no rational phase structure
    resampleObj_setSamplateRatio(resampleObj, ratio);
    SPEECH_LOG(DEBUG) << TAG(kTag) << "Sample rate ratio set to " << ratio << std::endl;
}

void Resample::enableContinuous(bool flag) {
    if (isIdentity) {
        return;
    }
    ensureAudioFlux();
    poly_.reset();  // continuous mode carries tail state through the direct kernel
    resampleObj_enableContinue(resampleObj, flag ? 1 : 0);
}

detail::UninitVector<float> Resample::resampleFlat(const float* data, std::size_t n) {
    if (isIdentity) {
        return detail::UninitVector<float>(data, data + n);
    }
    if (n == 0) {
        SPEECH_LOG(DEBUG) << TAG(kTag) << "resample() called with empty input; returning empty output."
                   << std::endl;
        return {};
    }

    if (poly_) {
        auto out = resamplePolyphase(data, n);
        SPEECH_LOG(DEBUG) << TAG(kTag) << "Resampled " << n << " -> " << out.size()
                   << " samples (polyphase)." << std::endl;
        return out;
    }

    ensureAudioFlux();
    const int inputLength = static_cast<int>(n);
    const int outputLength = resampleObj_calDataLength(resampleObj, inputLength);
    SPEECH_LOG(TRACE) << TAG(kTag) << "calDataLength(" << inputLength << ") -> expected output length "
               << outputLength << std::endl;

    // The direct AudioFlux kernel ACCUMULATES into its output (dataArr2[i] += ...),
    // so this buffer must start zeroed -- unlike the polyphase path above, which
    // overwrites every sample.
    detail::UninitVector<float> out(static_cast<std::size_t>(outputLength));
    std::fill(out.begin(), out.end(), 0.0f);
    const int actual = resampleObj_resample(resampleObj, const_cast<float*>(data), inputLength,
                                            out.data());
    if (actual <= 0) {
        SPEECH_LOG(ERROR) << TAG(kTag) << "Resampling produced no output (input length=" << inputLength
                   << ")." << std::endl;
        throw std::runtime_error("Resampling failed or produced no output.");
    }
    out.resize(static_cast<std::size_t>(actual));
    SPEECH_LOG(DEBUG) << TAG(kTag) << "Resampled " << inputLength << " -> " << actual << " samples."
               << std::endl;
    return out;
}

std::vector<float> Resample::resample(const std::vector<float>& inputData) {
    auto flat = resampleFlat(inputData.data(), inputData.size());
    return std::vector<float>(flat.begin(), flat.end());
}

void Resample::reset() {
    if (resampleObj) {
        resampleObj_free(resampleObj);
        resampleObj = nullptr;
    }
}

}  // namespace speech::dsp
