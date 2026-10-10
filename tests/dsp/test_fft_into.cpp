// Tests for the raw-buffer FFT API (forwardInto & co.) and its real-input fast path, which must
// agree with AudioFlux's complex FFT (the reference it replaces) at every length it covers.
#include "utest.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "libspeech/dsp/fft.h"
#include "dsp/fft_algorithm.h"  // vendored AudioFlux C API: the reference kernel (private include dir)

using speech::dsp::FFT;

namespace {

std::vector<float> randomVec(std::size_t n, std::uint32_t seed) {
    std::uint32_t s = seed;
    std::vector<float> v(n);
    for (auto& x : v) {
        s = s * 1664525u + 1013904223u;
        x = static_cast<float>(s >> 8) / static_cast<float>(1u << 23) - 1.0f;
    }
    return v;
}

float maxAbs(const std::vector<float>& v) {
    float m = 0.0f;
    for (float x : v) m = std::max(m, std::fabs(x));
    return m;
}

float maxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

struct Reference {
    std::vector<float> re, im;
};

// AudioFlux's complex FFT through its own C API, real-only input when `imag` is null.
Reference audioFluxForward(int radix2Exp, const std::vector<float>& real, const std::vector<float>* imag) {
    FFTObj obj = nullptr;
    fftObj_new(&obj, radix2Exp);
    Reference r{std::vector<float>(real.size()), std::vector<float>(real.size())};
    fftObj_fft(obj, const_cast<float*>(real.data()), imag ? const_cast<float*>(imag->data()) : nullptr,
               r.re.data(), r.im.data());
    fftObj_free(obj);
    return r;
}

}  // namespace

// Real input, every supported size and the lengths just outside it (N=2 and N=4 are below the
// engine's range, N=2^17 is its top end): same spectrum as AudioFlux, and an exactly
// conjugate-symmetric one.
UTEST(FftInto, RealInputMatchesAudioFluxAndIsConjugateSymmetric) {
    for (int radix : {1, 2, 4, 6, 10, 13, 14, 16, 17}) {
        const int N = 1 << radix;
        FFT fft(radix);
        const std::vector<float> x = randomVec(N, 100u + radix);
        const Reference ref = audioFluxForward(radix, x, nullptr);

        std::vector<float> re(N), im(N);
        fft.forwardInto(x.data(), nullptr, re.data(), im.data());

        const float tol = 1e-4f * std::max(1.0f, std::max(maxAbs(ref.re), maxAbs(ref.im)));
        ASSERT_LT(maxAbsDiff(ref.re, re), tol);
        ASSERT_LT(maxAbsDiff(ref.im, im), tol);
        for (int k = 1; k < N / 2; ++k) {
            ASSERT_EQ(re[N - k], re[k]);
            ASSERT_EQ(im[N - k], -im[k]);
        }
        ASSERT_EQ(im[0], 0.0f);
    }
}

// Independent of AudioFlux: the real-input path against a direct double-precision DFT.
UTEST(FftInto, RealInputMatchesADirectDft) {
    for (int radix : {4, 6}) {
        const int N = 1 << radix;
        FFT fft(radix);
        const std::vector<float> x = randomVec(N, 5u);
        std::vector<float> re(N), im(N);
        fft.forwardInto(x.data(), nullptr, re.data(), im.data());
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
UTEST(FftInto, ComplexInputIsAudioFluxsOwnResult) {
    const int radix = 9, N = 1 << radix;
    FFT fft(radix);
    const std::vector<float> x = randomVec(N, 1u), y = randomVec(N, 2u);
    const Reference ref = audioFluxForward(radix, x, &y);
    std::vector<float> re(N), im(N);
    fft.forwardInto(x.data(), y.data(), re.data(), im.data());
    ASSERT_EQ(maxAbsDiff(ref.re, re), 0.0f);
    ASSERT_EQ(maxAbsDiff(ref.im, im), 0.0f);
}

UTEST(FftInto, RealForwardThenInverseRecoversTheSignal) {
    for (int radix : {12, 16}) {  // one size on the stack path, one on the heap path
        const int N = 1 << radix;
        FFT fft(radix);
        const std::vector<float> x = randomVec(N, 9u);
        std::vector<float> re(N), im(N), back(N), backIm(N);
        fft.forwardInto(x.data(), nullptr, re.data(), im.data());
        fft.inverseInto(re.data(), im.data(), back.data(), backIm.data());
        ASSERT_LT(maxAbsDiff(x, back), 2e-4f);
        ASSERT_LT(maxAbs(backIm), 2e-4f);
    }
}

// The vector API is a thin wrapper over the raw-buffer one and must keep returning the same values.
UTEST(FftInto, VectorApiAgreesWithTheRawBufferApi) {
    const int radix = 10, N = 1 << radix;
    FFT fft(radix);
    const std::vector<float> x = randomVec(N, 3u), y = randomVec(N, 4u);

    const auto fwd = fft.forward(x);
    std::vector<float> re(N), im(N);
    fft.forwardInto(x.data(), nullptr, re.data(), im.data());
    ASSERT_EQ(maxAbsDiff(fwd.first, re), 0.0f);
    ASSERT_EQ(maxAbsDiff(fwd.second, im), 0.0f);

    const auto inv = fft.inverse(x, y);
    fft.inverseInto(x.data(), y.data(), re.data(), im.data());
    ASSERT_EQ(maxAbsDiff(inv.first, re), 0.0f);
    ASSERT_EQ(maxAbsDiff(inv.second, im), 0.0f);

    std::vector<float> out(N);
    for (bool norm : {true, false}) {
        fft.dctInto(x.data(), out.data(), norm);
        ASSERT_EQ(maxAbsDiff(fft.dct(x, norm), out), 0.0f);
        fft.idctInto(x.data(), out.data(), norm);
        ASSERT_EQ(maxAbsDiff(fft.idct(x, norm), out), 0.0f);
    }
}

UTEST(FftInto, IdctIntoDoesNotModifyItsInput) {
    FFT fft(8);
    const std::vector<float> data = randomVec(256, 6u);
    std::vector<float> copy = data, out(256);
    fft.idctInto(copy.data(), out.data(), true);
    ASSERT_EQ(maxAbsDiff(data, copy), 0.0f);
}

UTEST(FftInto, NullBuffersThrow) {
    FFT fft(6);
    std::vector<float> a(64), b(64);
    bool outNull = false, inNull = false, dctNull = false;
    try { fft.forwardInto(a.data(), nullptr, nullptr, b.data()); } catch (const std::invalid_argument&) { outNull = true; }
    try { fft.inverseInto(nullptr, a.data(), a.data(), b.data()); } catch (const std::invalid_argument&) { inNull = true; }
    try { fft.dctInto(nullptr, a.data()); } catch (const std::invalid_argument&) { dctNull = true; }
    ASSERT_TRUE(outNull);
    ASSERT_TRUE(inNull);
    ASSERT_TRUE(dctNull);
}
