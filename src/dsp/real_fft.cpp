#include "real_fft.h"

#include <cmath>
#include <stdexcept>
#include <vector>

#include "simd_internal.h"

namespace speech::dsp::rfft_impl {

// Gives the free kernel functions access to RealFrameFFT's tables.
struct kernel_access {
    static const float* window(const real_frame_fft& f) { return f.window_.data(); }
    static const std::uint16_t* rev(const real_frame_fft& f) { return f.rev_.data(); }
    static const float* twr(const real_frame_fft& f) { return f.twr_.data(); }
    static const float* twi(const real_frame_fft& f) { return f.twi_.data(); }
    static const int* stage_offset(const real_frame_fft& f) { return f.stage_offset_.data(); }
    static int num_stages(const real_frame_fft& f) { return static_cast<int>(f.stage_offset_.size()); }
    static const float* post_r(const real_frame_fft& f) { return f.post_r_.data(); }
    static const float* post_i(const real_frame_fft& f) { return f.post_i_.data(); }
    static int m(const real_frame_fft& f) { return f.m_; }
};

namespace {

constexpr int max_m = real_frame_fft::max_length / 2;

// In-place m-point complex FFT (m = fft_length/2) of split re/im arrays that already hold the
// input in bit-reversed order. Shared by the forward kernel and the inverse one.
LS_ALWAYS_INLINE void butterflies(const real_frame_fft& f, float* LS_RESTRICT zr, float* LS_RESTRICT zi) {
    const int m = kernel_access::m(f);
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
    const float* twr = kernel_access::twr(f);
    const float* twi = kernel_access::twi(f);
    const int* stage_offset = kernel_access::stage_offset(f);
    const int num_stages = kernel_access::num_stages(f);
    int len = 8;
    for (int s = 0; s < num_stages; ++s, len <<= 1) {
        const int half = len >> 1;
        const float* LS_RESTRICT wr = twr + stage_offset[s];
        const float* LS_RESTRICT wi = twi + stage_offset[s];
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
}

// The whole frame -> half-spectrum pipeline (see the note below on vectorization).
// S is the stride of the output elements: 1 writes split planes (out_re[k], out_im[k]); 2 writes
// interleaved complex numbers (pass out_re = p, out_im = p + 1), straight into e.g. a complex64 array.
template <int S>
LS_ALWAYS_INLINE void kernel_body(const real_frame_fft& f, const float* LS_RESTRICT frame,
                                 float* LS_RESTRICT out_re, float* LS_RESTRICT out_im,
                                 float* LS_RESTRICT zr, float* LS_RESTRICT zi) {
    const int m = kernel_access::m(f);
    const float* LS_RESTRICT window = kernel_access::window(f);
    const std::uint16_t* LS_RESTRICT rev = kernel_access::rev(f);

    // 1. window + pack + bit-reverse in one pass: z[rev[n]] = w*x[2n] + i*w*x[2n+1]
    for (int n = 0; n < m; ++n) {
        const int j = rev[n];
        zr[j] = frame[2 * n] * window[2 * n];
        zi[j] = frame[2 * n + 1] * window[2 * n + 1];
    }

    butterflies(f, zr, zi);

    // 3. split: X[k] = E[k] + W^k * O[k], recovered from Z[k] and Z[m-k].
    //    Bins k and m-k are produced together from the same pair of inputs.
    const float* LS_RESTRICT pr = kernel_access::post_r(f);
    const float* LS_RESTRICT pi = kernel_access::post_i(f);
    out_re[0] = zr[0] + zi[0];
    out_im[0] = 0.0f;
    out_re[S * m] = zr[0] - zi[0];
    out_im[S * m] = 0.0f;
    const int quarter = m >> 1;
    for (int k = 1; k < quarter; ++k) {
        const int kk = m - k;
        const float a = zr[k], b = zi[k], c = zr[kk], d = zi[kk];
        const float er = 0.5f * (a + c), ei = 0.5f * (b - d);
        const float orr = 0.5f * (b + d), oi = 0.5f * (c - a);
        const float wr = pr[k], wi = pi[k];
        out_re[S * k] = er + wr * orr - wi * oi;
        out_im[S * k] = ei + wr * oi + wi * orr;
        // Mirror bin m-k: same E (conjugated roles), O' = (orr, -oi), W^(m-k) = (-wr, wi).
        out_re[S * kk] = er - wr * orr - wi * (-oi);
        out_im[S * kk] = -ei - wr * (-oi) + wi * orr;
    }
    {   // k = m/2 pairs with itself
        const int k = quarter;
        const float a = zr[k], b = zi[k];
        // c = a, d = b: E = (a, 0), O = (b, 0), W^(m/2) = exp(-i*pi/2) = (0, -1)
        out_re[S * k] = a;
        out_im[S * k] = -b;
    }
}

// The whole half-spectrum -> frame pipeline: the forward kernel run backwards.
//   X[k] = E[k] + W^k O[k]   (W = exp(-2*pi*i/N))  gives, for k in [0, m),
//   E[k] = (X[k] + conj(X[m-k])) / 2,   O[k] = W^-k (X[k] - conj(X[m-k])) / 2,
//   Z[k] = E[k] + i*O[k],               and Z[m-k] = conj(E[k]) + i*conj(O[k]),
// where Z is the m-point spectrum of z[n] = x[2n] + i*x[2n+1]. An inverse FFT of Z is a forward
// FFT of conj(Z), conjugated and divided by m, so the same butterflies are reused.
// S is the stride of the input elements, as in kernel_body().
template <int S>
LS_ALWAYS_INLINE void inverse_body(const real_frame_fft& f, const float* LS_RESTRICT in_re,
                                  const float* LS_RESTRICT in_im, float* LS_RESTRICT frame,
                                  float* LS_RESTRICT zr, float* LS_RESTRICT zi) {
    const int m = kernel_access::m(f);
    const std::uint16_t* LS_RESTRICT rev = kernel_access::rev(f);
    const float* LS_RESTRICT pr = kernel_access::post_r(f);
    const float* LS_RESTRICT pi = kernel_access::post_i(f);

    // 1. conj(Z), written at bit-reversed positions. k = 0 (rev = 0) pairs with the Nyquist bin
    //    and has no imaginary part to use; k = m/2 pairs with itself.
    zr[0] = 0.5f * (in_re[0] + in_re[S * m]);
    zi[0] = -(0.5f * (in_re[0] - in_re[S * m]));
    const int quarter = m >> 1;
    for (int k = 1; k < quarter; ++k) {
        const int kk = m - k;
        const float xr = in_re[S * k], xi = in_im[S * k], yr = in_re[S * kk], yi = in_im[S * kk];
        const float er = 0.5f * (xr + yr), ei = 0.5f * (xi - yi);
        const float dr = 0.5f * (xr - yr), di = 0.5f * (xi + yi);  // W^k * O[k]
        const float c = pr[k], s = -pi[k];                         // W^-k = c + i*s
        const float orr = dr * c - di * s, oi = dr * s + di * c;   // O[k]
        zr[rev[k]] = er - oi;   // Z[k]    = (er - oi,  ei + orr)  -> stored conjugated
        zi[rev[k]] = -(ei + orr);
        zr[rev[kk]] = er + oi;  // Z[m-k]  = (er + oi, -ei + orr)  -> stored conjugated
        zi[rev[kk]] = ei - orr;
    }
    zr[rev[quarter]] = in_re[S * quarter];  // Z[m/2] = (re, -im), conjugated
    zi[rev[quarter]] = in_im[S * quarter];

    // 2. y = FFT(conj(Z)) in place.
    butterflies(f, zr, zi);

    // 3. z = conj(y) / m, interleaved back into the real frame.
    const float scale = 1.0f / static_cast<float>(m);
    for (int n = 0; n < m; ++n) {
        frame[2 * n] = zr[n] * scale;
        frame[2 * n + 1] = -zi[n] * scale;
    }
}

// The two m-point work arrays: on the stack up to the STFT/MFCC sizes (no allocation on the
// per-frame hot path), on the heap for the larger single-frame FFTs.
template <typename Body>
LS_ALWAYS_INLINE void with_scratch(int m, Body&& body) {
    if (m <= max_m) {
        float zr[max_m];
        float zi[max_m];
        body(zr, zi);
    } else {
        std::vector<float> buf(2 * static_cast<std::size_t>(m));
        body(buf.data(), buf.data() + m);
    }
}

// One kernel for every target, deliberately: the compiler vectorizes the butterfly
// loops for whatever baseline ISA it is building for (SSE2 / NEON / ...). An AVX2+FMA
// clone of this same body was built and measured against it (n_fft 64..8192, x86-64)
// and was never faster -- the loops are short (half = 4..128) and 256-bit vectors only
// add epilogue and lane-crossing cost -- so it was removed rather than kept as dead
// complexity. Explicit SIMD lives where it measurably wins: the resampler.
}  // namespace

real_frame_fft::real_frame_fft(int fft_length, const float* window)
    : n_(fft_length), m_(fft_length / 2) {
    if (!supports_large(fft_length)) {
        throw std::invalid_argument("RealFrameFFT: fft_length must be a power of two in [16, 131072].");
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

    const double two_pi = 6.283185307179586476925286766559;
    for (int len = 8; len <= m_; len <<= 1) {
        stage_offset_.push_back(static_cast<int>(twr_.size()));
        for (int k = 0; k < len / 2; ++k) {
            const double a = two_pi * k / len;
            twr_.push_back(static_cast<float>(std::cos(a)));
            twi_.push_back(static_cast<float>(-std::sin(a)));
        }
    }

    post_r_.resize(m_ / 2 + 1);
    post_i_.resize(m_ / 2 + 1);
    for (int k = 0; k <= m_ / 2; ++k) {
        const double a = two_pi * k / n_;
        post_r_[k] = static_cast<float>(std::cos(a));
        post_i_[k] = static_cast<float>(-std::sin(a));
    }
}

void real_frame_fft::spectrum(const float* frame, float* re, float* im) const {
    with_scratch(m_, [&](float* zr, float* zi) { kernel_body<1>(*this, frame, re, im, zr, zi); });
}

void real_frame_fft::spectrum_interleaved(const float* frame, float* reim) const {
    with_scratch(m_, [&](float* zr, float* zi) { kernel_body<2>(*this, frame, reim, reim + 1, zr, zi); });
}

void real_frame_fft::inverse(const float* re, const float* im, float* frame) const {
    with_scratch(m_, [&](float* zr, float* zi) { inverse_body<1>(*this, re, im, frame, zr, zi); });
}

void real_frame_fft::inverse_interleaved(const float* reim, float* frame) const {
    with_scratch(m_, [&](float* zr, float* zi) { inverse_body<2>(*this, reim, reim + 1, frame, zr, zi); });
}

void real_frame_fft::power(const float* frame, float* power) const {
    if (n_ > max_length) {
        throw std::logic_error("RealFrameFFT::power: fft_length above 8192 is not supported here.");
    }
    float re[max_length / 2 + 1];
    float im[max_length / 2 + 1];
    with_scratch(m_, [&](float* zr, float* zi) { kernel_body<1>(*this, frame, re, im, zr, zi); });
    const int bins = num_bins();
    for (int k = 0; k < bins; ++k) {
        power[k] = re[k] * re[k] + im[k] * im[k];
    }
}

}  // namespace speech::dsp::rfft_impl
