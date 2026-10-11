// Tests for the raw-buffer FFT API (forward_into & co.) and its real-input fast path, which must
// agree with AudioFlux's complex FFT (the reference it replaces) at every length it covers.
#include "utest.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "libspeech/dsp/fft.h"
#include "dsp/fft_algorithm.h"  // vendored AudioFlux C API: the reference kernel (private include dir)

using speech::dsp::fft;

namespace {

std::vector<float> random_vec(std::size_t n, std::uint32_t seed) {
    std::uint32_t s = seed;
    std::vector<float> v(n);
    for (auto& x : v) {
        s = s * 1664525u + 1013904223u;
        x = static_cast<float>(s >> 8) / static_cast<float>(1u << 23) - 1.0f;
    }
    return v;
}

float max_abs(const std::vector<float>& v) {
    float m = 0.0f;
    for (float x : v) m = std::max(m, std::fabs(x));
    return m;
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

struct reference_result {
    std::vector<float> re, im;
};

// AudioFlux's complex FFT through its own C API, real-only input when `imag` is null.
reference_result audio_flux_forward(int radix2_exp, const std::vector<float>& real, const std::vector<float>* imag) {
    FFTObj obj = nullptr;
    fftObj_new(&obj, radix2_exp);
    reference_result r{std::vector<float>(real.size()), std::vector<float>(real.size())};
    fftObj_fft(obj, const_cast<float*>(real.data()), imag ? const_cast<float*>(imag->data()) : nullptr,
               r.re.data(), r.im.data());
    fftObj_free(obj);
    return r;
}

}  // namespace

// Real input, every supported size and the lengths just outside it (N=2 and N=4 are below the
// engine's range, N=2^17 is its top end): same spectrum as AudioFlux, and an exactly
// conjugate-symmetric one.
UTEST(fft_into, real_input_matches_audio_flux_and_is_conjugate_symmetric) {
    for (int radix : {1, 2, 4, 6, 10, 13, 14, 16, 17}) {
        const int N = 1 << radix;
        fft fft(radix);
        const std::vector<float> x = random_vec(N, 100u + radix);
        const reference_result ref = audio_flux_forward(radix, x, nullptr);

        std::vector<float> re(N), im(N);
        fft.forward_into(x.data(), nullptr, re.data(), im.data());

        const float tol = 1e-4f * std::max(1.0f, std::max(max_abs(ref.re), max_abs(ref.im)));
        ASSERT_LT(max_abs_diff(ref.re, re), tol);
        ASSERT_LT(max_abs_diff(ref.im, im), tol);
        for (int k = 1; k < N / 2; ++k) {
            ASSERT_EQ(re[N - k], re[k]);
            ASSERT_EQ(im[N - k], -im[k]);
        }
        ASSERT_EQ(im[0], 0.0f);
    }
}

// Independent of AudioFlux: the real-input path against a direct double-precision DFT.
UTEST(fft_into, real_input_matches_a_direct_dft) {
    for (int radix : {4, 6}) {
        const int N = 1 << radix;
        fft fft(radix);
        const std::vector<float> x = random_vec(N, 5u);
        std::vector<float> re(N), im(N);
        fft.forward_into(x.data(), nullptr, re.data(), im.data());
        for (int k = 0; k < N; ++k) {
            double sr = 0.0, si = 0.0;
            for (int n = 0; n < N; ++n) {
                const double a = -2.0 * 3.14159265358979323846 * k * n / N;
                sr += x[n] * std::cos(a);
                si += x[n] * std::sin(a);
            }
            ASSERT_NEAR(re[k], sr, 1e-4);
            ASSERT_NEAR(im[k], si, 1e-4);
        }
    }
}

// Complex (or imag-only) input still goes through AudioFlux itself, so it is bit-identical.
UTEST(fft_into, complex_input_is_audio_fluxs_own_result) {
    const int radix = 9, N = 1 << radix;
    fft fft(radix);
    const std::vector<float> x = random_vec(N, 1u), y = random_vec(N, 2u);
    const reference_result ref = audio_flux_forward(radix, x, &y);
    std::vector<float> re(N), im(N);
    fft.forward_into(x.data(), y.data(), re.data(), im.data());
    ASSERT_EQ(max_abs_diff(ref.re, re), 0.0f);
    ASSERT_EQ(max_abs_diff(ref.im, im), 0.0f);
}

UTEST(fft_into, real_forward_then_inverse_recovers_the_signal) {
    for (int radix : {12, 16}) {  // one size on the stack path, one on the heap path
        const int N = 1 << radix;
        fft fft(radix);
        const std::vector<float> x = random_vec(N, 9u);
        std::vector<float> re(N), im(N), back(N), back_im(N);
        fft.forward_into(x.data(), nullptr, re.data(), im.data());
        fft.inverse_into(re.data(), im.data(), back.data(), back_im.data());
        ASSERT_LT(max_abs_diff(x, back), 2e-4f);
        ASSERT_LT(max_abs(back_im), 2e-4f);
    }
}

// The vector API is a thin wrapper over the raw-buffer one and must keep returning the same values.
UTEST(fft_into, vector_api_agrees_with_the_raw_buffer_api) {
    const int radix = 10, N = 1 << radix;
    fft fft(radix);
    const std::vector<float> x = random_vec(N, 3u), y = random_vec(N, 4u);

    const auto fwd = fft.forward(x);
    std::vector<float> re(N), im(N);
    fft.forward_into(x.data(), nullptr, re.data(), im.data());
    ASSERT_EQ(max_abs_diff(fwd.first, re), 0.0f);
    ASSERT_EQ(max_abs_diff(fwd.second, im), 0.0f);

    const auto inv = fft.inverse(x, y);
    fft.inverse_into(x.data(), y.data(), re.data(), im.data());
    ASSERT_EQ(max_abs_diff(inv.first, re), 0.0f);
    ASSERT_EQ(max_abs_diff(inv.second, im), 0.0f);

    std::vector<float> out(N);
    for (bool norm : {true, false}) {
        fft.dct_into(x.data(), out.data(), norm);
        ASSERT_EQ(max_abs_diff(fft.dct(x, norm), out), 0.0f);
        fft.idct_into(x.data(), out.data(), norm);
        ASSERT_EQ(max_abs_diff(fft.idct(x, norm), out), 0.0f);
    }
}

UTEST(fft_into, idct_into_does_not_modify_its_input) {
    fft fft(8);
    const std::vector<float> data = random_vec(256, 6u);
    std::vector<float> copy = data, out(256);
    fft.idct_into(copy.data(), out.data(), true);
    ASSERT_EQ(max_abs_diff(data, copy), 0.0f);
}

UTEST(fft_into, null_buffers_throw) {
    fft fft(6);
    std::vector<float> a(64), b(64);
    bool out_null = false, in_null = false, dct_null = false;
    try { fft.forward_into(a.data(), nullptr, nullptr, b.data()); } catch (const std::invalid_argument&) { out_null = true; }
    try { fft.inverse_into(nullptr, a.data(), a.data(), b.data()); } catch (const std::invalid_argument&) { in_null = true; }
    try { fft.dct_into(nullptr, a.data()); } catch (const std::invalid_argument&) { dct_null = true; }
    ASSERT_TRUE(out_null);
    ASSERT_TRUE(in_null);
    ASSERT_TRUE(dct_null);
}
