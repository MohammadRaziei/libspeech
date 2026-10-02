#include "real_fft.h"

#include <cmath>
#include <stdexcept>

#include "simd_internal.h"

namespace speech::dsp::rfft_impl {

// Gives the free kernel functions access to RealFrameFFT's tables.
struct KernelAccess {
    static const float* window(const RealFrameFFT& f) { return f.window_.data(); }
    static const std::uint16_t* rev(const RealFrameFFT& f) { return f.rev_.data(); }
    static const float* twr(const RealFrameFFT& f) { return f.twr_.data(); }
    static const float* twi(const RealFrameFFT& f) { return f.twi_.data(); }
    static const int* stageOffset(const RealFrameFFT& f) { return f.stageOffset_.data(); }
    static int numStages(const RealFrameFFT& f) { return static_cast<int>(f.stageOffset_.size()); }
    static const float* postR(const RealFrameFFT& f) { return f.postR_.data(); }
    static const float* postI(const RealFrameFFT& f) { return f.postI_.data(); }
    static int m(const RealFrameFFT& f) { return f.m_; }
};

namespace {

constexpr int kMaxM = RealFrameFFT::kMaxLength / 2;

// The whole frame -> half-spectrum pipeline (see the note below on vectorization).
LS_ALWAYS_INLINE void kernelBody(const RealFrameFFT& f, const float* LS_RESTRICT frame,
                                 float* LS_RESTRICT outRe, float* LS_RESTRICT outIm) {
    const int m = KernelAccess::m(f);
    const float* LS_RESTRICT window = KernelAccess::window(f);
    const std::uint16_t* LS_RESTRICT rev = KernelAccess::rev(f);

    float zrBuf[kMaxM];
    float ziBuf[kMaxM];
    float* LS_RESTRICT zr = zrBuf;
    float* LS_RESTRICT zi = ziBuf;

    // 1. window + pack + bit-reverse in one pass: z[rev[n]] = w*x[2n] + i*w*x[2n+1]
    for (int n = 0; n < m; ++n) {
        const int j = rev[n];
        zr[j] = frame[2 * n] * window[2 * n];
        zi[j] = frame[2 * n + 1] * window[2 * n + 1];
    }

    // 2a. butterfly lengths 2 and 4 fused (twiddles are 1 and -i): radix-4 on
    //     the bit-reversed input.
    for (int g = 0; g < m; g += 4) {
        const float p0r = zr[g] + zr[g + 1], p0i = zi[g] + zi[g + 1];
        const float p1r = zr[g] - zr[g + 1], p1i = zi[g] - zi[g + 1];
        const float p2r = zr[g + 2] + zr[g + 3], p2i = zi[g + 2] + zi[g + 3];
        const float p3r = zr[g + 2] - zr[g + 3], p3i = zi[g + 2] - zi[g + 3];
        zr[g] = p0r + p2r;
        zi[g] = p0i + p2i;
        zr[g + 2] = p0r - p2r;
        zi[g + 2] = p0i - p2i;
        zr[g + 1] = p1r + p3i;  // p1 + (-i)*p3
        zi[g + 1] = p1i - p3r;
        zr[g + 3] = p1r - p3i;  // p1 - (-i)*p3
        zi[g + 3] = p1i + p3r;
    }

    // 2b. remaining stages, butterfly lengths 8 .. m. The inner loop runs over
    //     contiguous k with split re/im arrays, which vectorizes cleanly.
    const float* twr = KernelAccess::twr(f);
    const float* twi = KernelAccess::twi(f);
    const int* stageOffset = KernelAccess::stageOffset(f);
    const int numStages = KernelAccess::numStages(f);
    int len = 8;
    for (int s = 0; s < numStages; ++s, len <<= 1) {
        const int half = len >> 1;
        const float* LS_RESTRICT wr = twr + stageOffset[s];
        const float* LS_RESTRICT wi = twi + stageOffset[s];
        for (int base = 0; base < m; base += len) {
            float* LS_RESTRICT ar = zr + base;
            float* LS_RESTRICT ai = zi + base;
            float* LS_RESTRICT br = zr + base + half;
            float* LS_RESTRICT bi = zi + base + half;
            for (int k = 0; k < half; ++k) {
                const float tr = br[k] * wr[k] - bi[k] * wi[k];
                const float ti = br[k] * wi[k] + bi[k] * wr[k];
                br[k] = ar[k] - tr;
                bi[k] = ai[k] - ti;
                ar[k] += tr;
                ai[k] += ti;
            }
        }
    }

    // 3. split: X[k] = E[k] + W^k * O[k], recovered from Z[k] and Z[m-k].
    //    Bins k and m-k are produced together from the same pair of inputs.
    const float* LS_RESTRICT pr = KernelAccess::postR(f);
    const float* LS_RESTRICT pi = KernelAccess::postI(f);
    outRe[0] = zr[0] + zi[0];
    outIm[0] = 0.0f;
    outRe[m] = zr[0] - zi[0];
    outIm[m] = 0.0f;
    const int quarter = m >> 1;
    for (int k = 1; k < quarter; ++k) {
        const int kk = m - k;
        const float a = zr[k], b = zi[k], c = zr[kk], d = zi[kk];
        const float er = 0.5f * (a + c), ei = 0.5f * (b - d);
        const float orr = 0.5f * (b + d), oi = 0.5f * (c - a);
        const float wr = pr[k], wi = pi[k];
        outRe[k] = er + wr * orr - wi * oi;
        outIm[k] = ei + wr * oi + wi * orr;
        // Mirror bin m-k: same E (conjugated roles), O' = (orr, -oi), W^(m-k) = (-wr, wi).
        outRe[kk] = er - wr * orr - wi * (-oi);
        outIm[kk] = -ei - wr * (-oi) + wi * orr;
    }
    {   // k = m/2 pairs with itself
        const int k = quarter;
        const float a = zr[k], b = zi[k];
        // c = a, d = b: E = (a, 0), O = (b, 0), W^(m/2) = exp(-i*pi/2) = (0, -1)
        outRe[k] = a;
        outIm[k] = -b;
    }
}

// One kernel for every target, deliberately: the compiler vectorizes the butterfly
// loops for whatever baseline ISA it is building for (SSE2 / NEON / ...). An AVX2+FMA
// clone of this same body was built and measured against it (n_fft 64..8192, x86-64)
// and was never faster -- the loops are short (half = 4..128) and 256-bit vectors only
// add epilogue and lane-crossing cost -- so it was removed rather than kept as dead
// complexity. Explicit SIMD lives where it measurably wins: the resampler.
}  // namespace

RealFrameFFT::RealFrameFFT(int fftLength, const float* window)
    : n_(fftLength), m_(fftLength / 2) {
    if (!supports(fftLength)) {
        throw std::invalid_argument("RealFrameFFT: fftLength must be a power of two in [16, 8192].");
    }
    window_.assign(window, window + n_);

    int bits = 0;
    while ((1 << bits) < m_) ++bits;
    rev_.resize(m_);
    for (int i = 0; i < m_; ++i) {
        int r = 0;
        for (int b = 0; b < bits; ++b) {
            if (i & (1 << b)) r |= 1 << (bits - 1 - b);
        }
        rev_[i] = static_cast<std::uint16_t>(r);
    }

    const double twoPi = 6.283185307179586476925286766559;
    for (int len = 8; len <= m_; len <<= 1) {
        stageOffset_.push_back(static_cast<int>(twr_.size()));
        for (int k = 0; k < len / 2; ++k) {
            const double a = twoPi * k / len;
            twr_.push_back(static_cast<float>(std::cos(a)));
            twi_.push_back(static_cast<float>(-std::sin(a)));
        }
    }

    postR_.resize(m_ / 2 + 1);
    postI_.resize(m_ / 2 + 1);
    for (int k = 0; k <= m_ / 2; ++k) {
        const double a = twoPi * k / n_;
        postR_[k] = static_cast<float>(std::cos(a));
        postI_[k] = static_cast<float>(-std::sin(a));
    }
}

void RealFrameFFT::spectrum(const float* frame, float* re, float* im) const {
    kernelBody(*this, frame, re, im);
}

void RealFrameFFT::power(const float* frame, float* power) const {
    float re[kMaxLength / 2 + 1];
    float im[kMaxLength / 2 + 1];
    kernelBody(*this, frame, re, im);
    const int bins = numBins();
    for (int k = 0; k < bins; ++k) {
        power[k] = re[k] * re[k] + im[k] * im[k];
    }
}

}  // namespace speech::dsp::rfft_impl
