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

using speech::dsp::window_type;

using speech::dsp::stft;

namespace {

// Deterministic pseudo-random floats in [-1, 1).
struct rng {
    std::uint32_t s = 12345u;
    float next() {
        s = s * 1664525u + 1013904223u;
        return static_cast<float>(s >> 8) / static_cast<float>(1u << 23) - 1.0f;
    }
};

std::vector<float> random_vec(std::size_t n, std::uint32_t seed) {
    rng r;
    r.s = seed;
    std::vector<float> v(n);
    for (auto& x : v) x = r.next();
    return v;
}

float max_abs(const std::vector<float>& v) {
    float m = 0.0f;
    for (float x : v) m = std::max(m, std::fabs(x));
    return m;
}

float max_abs_diff(const std::vector<float>& a, const speech::detail::uninit_vector<float>& b) {
    float m = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

// AudioFlux's own complex istft on a FULL spectrum: the reference.
std::vector<float> reference_istft(int radix2_exp, int hop, const std::vector<float>& re,
                                  const std::vector<float>& im, int frames, int method) {
    STFTObj obj = nullptr;
    ::WindowType window = ::Window_Hann;  // AudioFlux's own enum: this talks to its C API
    int slide = hop;
    int is_continue = 0;
    stftObj_new(&obj, radix2_exp, &window, hop > 0 ? &slide : nullptr, &is_continue);
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
UTEST(istft, matches_audio_flux_kernel_on_arbitrary_spectra) {
    struct test_case { int radix; int hop; int frames; };
    const test_case cases[] = {{4, 0, 300}, {6, 0, 300}, {6, 32, 200}, {9, 0, 300}, {9, 256, 40}, {12, 0, 70}};
    for (const test_case& c : cases) {
        const int N = 1 << c.radix;
        stft stft(c.radix, window_type::hann, c.hop);
        const std::size_t total = static_cast<std::size_t>(c.frames) * N;
        const std::vector<float> re = random_vec(total, 1u + c.radix);
        const std::vector<float> im = random_vec(total, 99u + c.radix);
        for (int method : {0, 1}) {
            const std::vector<float> ref = reference_istft(c.radix, c.hop, re, im, c.frames, method);
            const auto out = stft.istft(re.data(), im.data(), c.frames, /*onesided=*/false, method);
            ASSERT_EQ(out.size(), ref.size());
            ASSERT_LT(max_abs_diff(ref, out), 1e-4f * std::max(1.0f, max_abs(ref)));
        }
    }
}

// A one-sided input (with junk in the DC/Nyquist imaginary parts, which belong to no real signal)
// equals the reference run on its conjugate-completed full spectrum.
UTEST(istft, onesided_matches_audio_flux_on_the_mirrored_spectrum) {
    for (int radix : {5, 9, 11}) {
        const int N = 1 << radix, half = N / 2, bins = half + 1, frames = 150;
        stft stft(radix);
        const std::vector<float> re = random_vec(static_cast<std::size_t>(frames) * bins, 7u);
        const std::vector<float> im = random_vec(static_cast<std::size_t>(frames) * bins, 8u);
        std::vector<float> full_re(static_cast<std::size_t>(frames) * N), full_im(full_re.size());
        for (int f = 0; f < frames; ++f) {
            for (int k = 0; k < bins; ++k) {
                const float r = re[static_cast<std::size_t>(f) * bins + k];
                const float i = (k == 0 || k == half) ? 0.0f : im[static_cast<std::size_t>(f) * bins + k];
                full_re[static_cast<std::size_t>(f) * N + k] = r;
                full_im[static_cast<std::size_t>(f) * N + k] = i;
                if (k > 0 && k < half) {
                    full_re[static_cast<std::size_t>(f) * N + N - k] = r;
                    full_im[static_cast<std::size_t>(f) * N + N - k] = -i;
                }
            }
        }
        for (int method : {0, 1}) {
            const std::vector<float> ref = reference_istft(radix, 0, full_re, full_im, frames, method);
            const auto out = stft.istft(re.data(), im.data(), frames, /*onesided=*/true, method);
            ASSERT_LT(max_abs_diff(ref, out), 1e-4f * std::max(1.0f, max_abs(ref)));
        }
    }
}

// stft -> istft returns the signal (away from the first/last fft_length samples, where the window
// tapers to zero), for the full and the one-sided spectrum and both methods.
UTEST(istft, round_trips_the_stft) {
    for (int radix : {6, 9, 11}) {
        const int N = 1 << radix;
        stft stft(radix);
        const std::size_t len = static_cast<std::size_t>(N) * 12;
        const std::vector<float> x = random_vec(len, 3u);
        for (bool onesided : {false, true}) {
            const auto spec = stft.spectrogram(x.data(), x.size(), onesided);
            for (int method : {0, 1}) {
                const auto y = stft.istft(spec, method);
                ASSERT_TRUE(y.size() >= len - stft.slide_length() * 4);
                float worst = 0.0f;
                for (std::size_t i = N; i + N < y.size(); ++i) worst = std::max(worst, std::fabs(y[i] - x[i]));
                ASSERT_LT(worst, 2e-4f);
            }
        }
    }
}

// Lengths the real-FFT engine does not support (N=8 and N=16384) go through AudioFlux and must
// still work, for both input shapes.
UTEST(istft, unsupported_lengths_fall_back_to_audio_flux) {
    for (int radix : {3, 14}) {
        const int N = 1 << radix;
        stft stft(radix);
        const std::size_t len = static_cast<std::size_t>(N) * 6;
        const std::vector<float> x = random_vec(len, 5u);
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
UTEST(istft, legacy_nested_vector_api_agrees_with_the_flat_one) {
    stft stft(8);
    const std::vector<float> x = random_vec(256 * 10, 11u);
    const auto spec = stft.spectrogram(x.data(), x.size(), false);
    const auto flat = stft.istft(spec, 0);
    const auto nested = stft.compute(x);
    const std::vector<float> via_nested = stft.istft(nested.first, nested.second, 0);
    ASSERT_EQ(via_nested.size(), flat.size());
    ASSERT_LT(max_abs_diff(via_nested, flat), 1e-6f);
}

UTEST(istft, rejects_bad_arguments) {
    stft stft(6);
    const std::vector<float> x = random_vec(64 * 6, 1u);
    const auto spec = stft.spectrogram(x.data(), x.size(), false);

    bool zero_frames = false, bad_method = false, bad_negative_method = false, null_ptr = false;
    try { stft.istft(spec.real.data(), spec.imag.data(), 0, false, 0); } catch (const std::invalid_argument&) { zero_frames = true; }
    try { stft.istft(spec, 2); } catch (const std::invalid_argument&) { bad_method = true; }
    try { stft.istft(spec, -1); } catch (const std::invalid_argument&) { bad_negative_method = true; }
    try { stft.istft(nullptr, nullptr, 3, false, 0); } catch (const std::invalid_argument&) { null_ptr = true; }
    ASSERT_TRUE(zero_frames);
    ASSERT_TRUE(bad_method);
    ASSERT_TRUE(bad_negative_method);
    ASSERT_TRUE(null_ptr);

    speech::dsp::complex_spectrogram wrong = spec;
    wrong.num_bins = 7;  // neither 64 nor 33
    bool wrong_bins = false;
    try { stft.istft(wrong, 0); } catch (const std::invalid_argument&) { wrong_bins = true; }
    ASSERT_TRUE(wrong_bins);
}
