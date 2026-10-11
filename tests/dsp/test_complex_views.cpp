// Complex data in both memory layouts -- interleaved (std::complex<float>) and split planes -- and
// the interleaved-complex overloads of STFT and FFT. They run the same kernels as the planar calls
// (only the element stride differs), so the results must match them EXACTLY, not just closely.
#include "utest.h"

#include <complex>
#include <cstdint>
#include <iterator>
#include <stdexcept>
#include <vector>

#include "libspeech/dsp/fft.h"
#include "libspeech/dsp/stft.h"
#include "libspeech/span.h"

using speech::dsp::window_type;

using cx = std::complex<float>;
using speech::frame_view;
using speech::span;
using speech::split_complex_frames;
using speech::split_complex_span;
using speech::dsp::fft;
using speech::dsp::stft;

namespace {
std::vector<float> noise(std::size_t n, std::uint32_t seed) {
    std::uint32_t s = seed;
    std::vector<float> v(n);
    for (auto& x : v) {
        s = s * 1664525u + 1013904223u;
        x = static_cast<float>(s >> 8) / static_cast<float>(1u << 23) - 1.0f;
    }
    return v;
}
template <typename Fn>
bool throws_invalid_argument(Fn&& fn) {
    try { fn(); } catch (const std::invalid_argument&) { return true; } catch (...) { return false; }
    return false;
}
}  // namespace

UTEST(complex_views, as_complex_and_as_floats_share_the_same_memory) {
    std::vector<float> f = {1, 2, 3, 4, 5, 6};
    span<cx> c = speech::as_complex(span<float>(f));
    ASSERT_EQ(c.size(), 3u);
    ASSERT_TRUE(static_cast<void*>(c.data()) == static_cast<void*>(f.data()));  // no copy
    ASSERT_TRUE(c[1] == cx(3.0f, 4.0f));
    c[2] = cx(-1.0f, -2.0f);  // writes through to the floats
    ASSERT_EQ(f[4], -1.0f);
    ASSERT_EQ(f[5], -2.0f);

    span<const float> back = speech::as_floats(span<const cx>(c));
    ASSERT_EQ(back.size(), 6u);
    ASSERT_TRUE(back.data() == f.data());

    std::vector<float> odd(5);
    ASSERT_TRUE(throws_invalid_argument([&] { (void)speech::as_complex(span<float>(odd)); }));
}

// Interleaved data iterates as plain std::complex<float>&: the iterator is a pointer.
UTEST(complex_views, interleaved_frames_yield_real_complex_references) {
    std::vector<cx> data(12);
    for (std::size_t i = 0; i < data.size(); ++i) data[i] = cx(float(i), -float(i));
    frame_view<cx> frames(data.data(), 3, 4);  // 3 frames of 4 complex values
    float sum = 0.0f;
    for (span<cx> frame : frames) {
        for (cx& z : frame) {  // a real reference into `data`
            z *= 2.0f;
            sum += z.real();
        }
    }
    ASSERT_EQ(sum, 2.0f * (11 * 12 / 2));
    ASSERT_TRUE(data[5] == cx(10.0f, -10.0f));
}

// Split planes cannot be reinterpret_cast to complex; the proxy hands out two references instead.
UTEST(complex_views, split_planes_iterate_with_a_proxy_and_write_back) {
    std::vector<float> re = {1, 2, 3}, im = {10, 20, 30};
    split_complex_span<float> s{span<float>(re), span<float>(im)};
    ASSERT_EQ(s.size(), 3u);

    cx second = s[1];  // implicit conversion
    ASSERT_TRUE(second == cx(2.0f, 20.0f));

    s[0] = cx(-1.0f, -10.0f);  // writes both planes
    ASSERT_EQ(re[0], -1.0f);
    ASSERT_EQ(im[0], -10.0f);

    s[2] = s[1];  // value assignment between proxies, not rebinding
    ASSERT_EQ(re[2], 2.0f);
    ASSERT_EQ(im[2], 20.0f);

    float energy = 0.0f;
    for (cx bin : s) energy += std::norm(bin);
    ASSERT_EQ(energy, 1.0f + 100.0f + 4.0f + 400.0f + 4.0f + 400.0f);

    // read-only view of the same planes, and the random-access operations
    split_complex_span<const float> ro = s;
    ASSERT_EQ(ro.end() - ro.begin(), 3);
    ASSERT_TRUE(cx(*(ro.begin() + 1)) == cx(2.0f, 20.0f));
    ASSERT_TRUE(cx(ro.begin()[2]) == cx(2.0f, 20.0f));
    ASSERT_TRUE(ro.begin() < ro.end());

    std::vector<float> shorter(2);
    ASSERT_TRUE(throws_invalid_argument([&] { split_complex_span<float>{span<float>(re), span<float>(shorter)}; }));
}

UTEST(complex_views, spectrogram_frames_walk_bin_by_bin) {
    stft stft(6);
    const std::vector<float> x = noise(64 * 6, 3u);
    auto spec = stft.spectrogram(x.data(), x.size(), true);

    split_complex_frames<const float> frames = std::as_const(spec).frames();
    ASSERT_EQ(frames.size(), static_cast<std::size_t>(spec.num_frames));
    ASSERT_EQ(frames.frame_length(), static_cast<std::size_t>(spec.num_bins));

    std::size_t seen = 0;
    float worst = 0.0f;
    for (auto frame : frames) {
        for (cx bin : frame) {
            const float dr = bin.real() - spec.real[seen];
            const float di = bin.imag() - spec.imag[seen];
            worst = std::max(worst, std::max(std::abs(dr), std::abs(di)));
            ++seen;
        }
    }
    ASSERT_EQ(seen, spec.real.size());
    ASSERT_EQ(worst, 0.0f);

    // the mutable form writes into the ComplexSpectrogram's planes
    spec.frames()[1][2] = cx(7.0f, 8.0f);
    ASSERT_EQ(spec.real[1 * spec.num_bins + 2], 7.0f);
    ASSERT_EQ(spec.imag[1 * spec.num_bins + 2], 8.0f);
}

// stft straight into interleaved complex == the planar result, bit for bit. Sizes cover the
// stack path, the unsupported-length fallback (N=8), and enough frames for the parallel path.
UTEST(complex_stft, interleaved_output_matches_planar_exactly) {
    for (int radix : {3, 4, 9, 13}) {
        const int N = 1 << radix;
        stft stft(radix, window_type::hann, std::max(1, N / 4));
        const std::size_t len = static_cast<std::size_t>(N) * (radix == 13 ? 6 : 40);
        const std::vector<float> x = noise(len, 10u + radix);
        for (bool onesided : {false, true}) {
            const auto planar = stft.spectrogram(x.data(), x.size(), onesided);
            std::vector<cx> out(static_cast<std::size_t>(planar.num_frames) * planar.num_bins);
            frame_view<cx> view(out.data(), planar.num_frames, planar.num_bins);
            const int frames = onesided ? stft.stft_onesided_into(span<const float>(x), view)
                                        : stft.stft_into(span<const float>(x), view);
            ASSERT_EQ(frames, planar.num_frames);
            for (std::size_t i = 0; i < out.size(); ++i) {
                ASSERT_EQ(out[i].real(), planar.real[i]);
                ASSERT_EQ(out[i].imag(), planar.imag[i]);
            }
        }
    }
}

UTEST(complex_stft, inverse_from_interleaved_matches_planar_exactly_and_round_trips) {
    for (int radix : {3, 4, 9, 12}) {
        const int N = 1 << radix;
        stft stft(radix);
        const std::vector<float> x = noise(static_cast<std::size_t>(N) * 20, 20u + radix);
        for (bool onesided : {false, true}) {
            const auto planar = stft.spectrogram(x.data(), x.size(), onesided);
            std::vector<cx> interleaved(static_cast<std::size_t>(planar.num_frames) * planar.num_bins);
            for (std::size_t i = 0; i < interleaved.size(); ++i) interleaved[i] = cx(planar.real[i], planar.imag[i]);
            frame_view<const cx> view(interleaved.data(), planar.num_frames, planar.num_bins);
            for (int method : {0, 1}) {
                const auto a = stft.istft(planar, method);
                const auto b = stft.istft(view, method);
                ASSERT_EQ(a.size(), b.size());
                for (std::size_t i = 0; i < a.size(); ++i) ASSERT_EQ(a[i], b[i]);
            }
            const auto y = stft.istft(view, 0);
            float worst = 0.0f;
            for (std::size_t i = N; i + N < y.size(); ++i) worst = std::max(worst, std::abs(y[i] - x[i]));
            ASSERT_LT(worst, 2e-4f);
        }
    }
}

UTEST(complex_stft, rejects_views_that_do_not_fit) {
    stft stft(6);
    const std::vector<float> x = noise(64 * 4, 1u);
    std::vector<cx> out(64 * 10);
    // wrong row length for the full / one-sided call
    ASSERT_TRUE(throws_invalid_argument([&] { stft.stft_into(span<const float>(x), frame_view<cx>(out.data(), 4, 33)); }));
    ASSERT_TRUE(throws_invalid_argument([&] { stft.stft_onesided_into(span<const float>(x), frame_view<cx>(out.data(), 4, 64)); }));
    // too few frames
    ASSERT_TRUE(throws_invalid_argument([&] { stft.stft_into(span<const float>(x), frame_view<cx>(out.data(), 1, 64)); }));
    // strided (hop != row length) rows are not one block
    ASSERT_TRUE(throws_invalid_argument([&] { stft.stft_into(span<const float>(x), frame_view<cx>(out.data(), 4, 64, 65)); }));
    // istft: bad row length, empty
    ASSERT_TRUE(throws_invalid_argument([&] { stft.istft(frame_view<const cx>(out.data(), 4, 40), 0); }));
    ASSERT_TRUE(throws_invalid_argument([&] { stft.istft(frame_view<const cx>(), 0); }));
    ASSERT_TRUE(throws_invalid_argument([&] { stft.istft(frame_view<const cx>(out.data(), 4, 64), 2); }));
}

// FFT: real -> complex writes the result in place, complex -> complex goes through the planar
// kernel; both must equal the planar API exactly, including the heap path (65536) and the
// below-engine length (8 samples).
UTEST(complex_fft, real_to_complex_matches_planar_exactly) {
    for (int radix : {3, 4, 10, 16}) {
        const int N = 1 << radix;
        fft fft(radix);
        const std::vector<float> x = noise(N, 30u + radix);
        std::vector<float> re(N), im(N);
        fft.forward_into(x.data(), nullptr, re.data(), im.data());
        std::vector<cx> out(N);
        fft.forward_into(span<const float>(x), span<cx>(out));
        for (int i = 0; i < N; ++i) {
            ASSERT_EQ(out[i].real(), re[i]);
            ASSERT_EQ(out[i].imag(), im[i]);
        }
    }
}

UTEST(complex_fft, complex_to_complex_matches_planar_exactly_and_works_in_place) {
    const int radix = 9, N = 1 << radix;
    fft fft(radix);
    const std::vector<float> a = noise(N, 1u), b = noise(N, 2u);
    std::vector<cx> in(N);
    for (int i = 0; i < N; ++i) in[i] = cx(a[i], b[i]);

    for (bool inverse : {false, true}) {
        std::vector<float> re(N), im(N);
        if (inverse) fft.inverse_into(a.data(), b.data(), re.data(), im.data());
        else fft.forward_into(a.data(), b.data(), re.data(), im.data());

        std::vector<cx> out(N);
        if (inverse) fft.inverse_into(span<const cx>(in), span<cx>(out));
        else fft.forward_into(span<const cx>(in), span<cx>(out));
        for (int i = 0; i < N; ++i) {
            ASSERT_EQ(out[i].real(), re[i]);
            ASSERT_EQ(out[i].imag(), im[i]);
        }

        std::vector<cx> in_place = in;  // same buffer as input and output
        if (inverse) fft.inverse_into(span<const cx>(in_place), span<cx>(in_place));
        else fft.forward_into(span<const cx>(in_place), span<cx>(in_place));
        for (int i = 0; i < N; ++i) ASSERT_TRUE(in_place[i] == out[i]);
    }
}

UTEST(complex_fft, rejects_wrong_sizes) {
    fft fft(6);
    std::vector<float> real(64), short_real(63);
    std::vector<cx> out(64), short_out(63);
    ASSERT_TRUE(throws_invalid_argument([&] { fft.forward_into(span<const float>(short_real), span<cx>(out)); }));
    ASSERT_TRUE(throws_invalid_argument([&] { fft.forward_into(span<const float>(real), span<cx>(short_out)); }));
    ASSERT_TRUE(throws_invalid_argument([&] { fft.forward_into(span<const cx>(short_out), span<cx>(out)); }));
    ASSERT_TRUE(throws_invalid_argument([&] { fft.inverse_into(span<const cx>(out), span<cx>(short_out)); }));
}
