// Tests for the flat / one-sided istft(): the real-FFT kernel must reproduce AudioFlux's complex
// stftObj_istft() (the reference it replaces) on arbitrary input, and round-trip the STFT.
#include "utest.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "libspeech/dsp/stft.h"
#include "libspeech/dsp/window.h"
#include "stft_algorithm.h"  // vendored AudioFlux C API: the reference kernel (private include dir)

using speech::dsp::STFT;

namespace {

// Deterministic pseudo-random floats in [-1, 1).
struct Rng {
    std::uint32_t s = 12345u;
    float next() {
        s = s * 1664525u + 1013904223u;
        return static_cast<float>(s >> 8) / static_cast<float>(1u << 23) - 1.0f;
    }
};

std::vector<float> randomVec(std::size_t n, std::uint32_t seed) {
    Rng r;
    r.s = seed;
    std::vector<float> v(n);
    for (auto& x : v) x = r.next();
    return v;
}

float maxAbs(const std::vector<float>& v) {
    float m = 0.0f;
    for (float x : v) m = std::max(m, std::fabs(x));
    return m;
}

float maxAbsDiff(const std::vector<float>& a, const speech::detail::UninitVector<float>& b) {
    float m = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

// AudioFlux's own complex istft on a FULL spectrum: the reference.
std::vector<float> referenceIstft(int radix2Exp, int hop, const std::vector<float>& re,
                                  const std::vector<float>& im, int frames, int method) {
    STFTObj obj = nullptr;
    WindowType window = Window_Hann;
    int slide = hop;
    int isContinue = 0;
    stftObj_new(&obj, radix2Exp, &window, hop > 0 ? &slide : nullptr, &isContinue);
    std::vector<float> out(static_cast<std::size_t>(stftObj_calDataLength(obj, frames)), 0.0f);
    stftObj_istft(obj, const_cast<float*>(re.data()), const_cast<float*>(im.data()), frames, method,
                  out.data());
    stftObj_free(obj);
    return out;
}

}  // namespace

// Random, NOT conjugate-symmetric full spectra: AudioFlux returns the real part of the inverse
// transform, and so must we. Covers several sizes, both methods, custom hops, and enough frames to
// run the multi-block / parallel path (N=512: 256 frames per block).
UTEST(Istft, MatchesAudioFluxKernelOnArbitrarySpectra) {
    struct Case { int radix; int hop; int frames; };
    const Case cases[] = {{4, 0, 300}, {6, 0, 300}, {6, 32, 200}, {9, 0, 300}, {9, 256, 40}, {12, 0, 70}};
    for (const Case& c : cases) {
        const int N = 1 << c.radix;
        STFT stft(c.radix, Window_Hann, c.hop);
        const std::size_t total = static_cast<std::size_t>(c.frames) * N;
        const std::vector<float> re = randomVec(total, 1u + c.radix);
        const std::vector<float> im = randomVec(total, 99u + c.radix);
        for (int method : {0, 1}) {
            const std::vector<float> ref = referenceIstft(c.radix, c.hop, re, im, c.frames, method);
            const auto out = stft.istft(re.data(), im.data(), c.frames, /*onesided=*/false, method);
            ASSERT_EQ(out.size(), ref.size());
            ASSERT_LT(maxAbsDiff(ref, out), 1e-4f * std::max(1.0f, maxAbs(ref)));
        }
    }
}

// A one-sided input (with junk in the DC/Nyquist imaginary parts, which belong to no real signal)
// equals the reference run on its conjugate-completed full spectrum.
UTEST(Istft, OnesidedMatchesAudioFluxOnTheMirroredSpectrum) {
    for (int radix : {5, 9, 11}) {
        const int N = 1 << radix, half = N / 2, bins = half + 1, frames = 150;
        STFT stft(radix);
        const std::vector<float> re = randomVec(static_cast<std::size_t>(frames) * bins, 7u);
        const std::vector<float> im = randomVec(static_cast<std::size_t>(frames) * bins, 8u);
        std::vector<float> fullRe(static_cast<std::size_t>(frames) * N), fullIm(fullRe.size());
        for (int f = 0; f < frames; ++f) {
            for (int k = 0; k < bins; ++k) {
                const float r = re[static_cast<std::size_t>(f) * bins + k];
                const float i = (k == 0 || k == half) ? 0.0f : im[static_cast<std::size_t>(f) * bins + k];
                fullRe[static_cast<std::size_t>(f) * N + k] = r;
                fullIm[static_cast<std::size_t>(f) * N + k] = i;
                if (k > 0 && k < half) {
                    fullRe[static_cast<std::size_t>(f) * N + N - k] = r;
                    fullIm[static_cast<std::size_t>(f) * N + N - k] = -i;
                }
            }
        }
        for (int method : {0, 1}) {
            const std::vector<float> ref = referenceIstft(radix, 0, fullRe, fullIm, frames, method);
            const auto out = stft.istft(re.data(), im.data(), frames, /*onesided=*/true, method);
            ASSERT_LT(maxAbsDiff(ref, out), 1e-4f * std::max(1.0f, maxAbs(ref)));
        }
    }
}

// stft -> istft returns the signal (away from the first/last fftLength samples, where the window
// tapers to zero), for the full and the one-sided spectrum and both methods.
UTEST(Istft, RoundTripsTheStft) {
    for (int radix : {6, 9, 11}) {
        const int N = 1 << radix;
        STFT stft(radix);
        const std::size_t len = static_cast<std::size_t>(N) * 12;
        const std::vector<float> x = randomVec(len, 3u);
        for (bool onesided : {false, true}) {
            const auto spec = stft.spectrogram(x.data(), x.size(), onesided);
            for (int method : {0, 1}) {
                const auto y = stft.istft(spec, method);
                ASSERT_TRUE(y.size() >= len - stft.slideLength() * 4);
                float worst = 0.0f;
                for (std::size_t i = N; i + N < y.size(); ++i) worst = std::max(worst, std::fabs(y[i] - x[i]));
                ASSERT_LT(worst, 2e-4f);
            }
        }
    }
}

// Lengths the real-FFT engine does not support (N=8 and N=16384) go through AudioFlux and must
// still work, for both input shapes.
UTEST(Istft, UnsupportedLengthsFallBackToAudioFlux) {
    for (int radix : {3, 14}) {
        const int N = 1 << radix;
        STFT stft(radix);
        const std::size_t len = static_cast<std::size_t>(N) * 6;
        const std::vector<float> x = randomVec(len, 5u);
        for (bool onesided : {false, true}) {
            const auto spec = stft.spectrogram(x.data(), x.size(), onesided);
            const auto y = stft.istft(spec, 0);
            float worst = 0.0f;
            for (std::size_t i = N; i + N < y.size(); ++i) worst = std::max(worst, std::fabs(y[i] - x[i]));
            ASSERT_LT(worst, 2e-4f);
        }
    }
}

// The nested-vector API (source compatible) is now a thin wrapper and must agree with the flat one.
UTEST(Istft, LegacyNestedVectorApiAgreesWithTheFlatOne) {
    STFT stft(8);
    const std::vector<float> x = randomVec(256 * 10, 11u);
    const auto spec = stft.spectrogram(x.data(), x.size(), false);
    const auto flat = stft.istft(spec, 0);
    const auto nested = stft.stft(x);
    const std::vector<float> viaNested = stft.istft(nested.first, nested.second, 0);
    ASSERT_EQ(viaNested.size(), flat.size());
    ASSERT_LT(maxAbsDiff(viaNested, flat), 1e-6f);
}

UTEST(Istft, RejectsBadArguments) {
    STFT stft(6);
    const std::vector<float> x = randomVec(64 * 6, 1u);
    const auto spec = stft.spectrogram(x.data(), x.size(), false);

    bool zeroFrames = false, badMethod = false, badNegativeMethod = false, nullPtr = false;
    try { stft.istft(spec.real.data(), spec.imag.data(), 0, false, 0); } catch (const std::invalid_argument&) { zeroFrames = true; }
    try { stft.istft(spec, 2); } catch (const std::invalid_argument&) { badMethod = true; }
    try { stft.istft(spec, -1); } catch (const std::invalid_argument&) { badNegativeMethod = true; }
    try { stft.istft(nullptr, nullptr, 3, false, 0); } catch (const std::invalid_argument&) { nullPtr = true; }
    ASSERT_TRUE(zeroFrames);
    ASSERT_TRUE(badMethod);
    ASSERT_TRUE(badNegativeMethod);
    ASSERT_TRUE(nullPtr);

    speech::dsp::Spectrogram wrong = spec;
    wrong.numBins = 7;  // neither 64 nor 33
    bool wrongBins = false;
    try { stft.istft(wrong, 0); } catch (const std::invalid_argument&) { wrongBins = true; }
    ASSERT_TRUE(wrongBins);
}
