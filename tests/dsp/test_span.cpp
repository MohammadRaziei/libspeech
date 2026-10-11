// Tests for Span / frame_view and for the view-based overloads of STFT and FFT, which must give
// exactly the results of the pointer-based calls they sit next to.
#include "utest.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "libspeech/dsp/fft.h"
#include "libspeech/dsp/stft.h"
#include "libspeech/span.h"

using speech::dsp::window_type;

using speech::frame_view;
using speech::span;
using speech::dsp::fft;
using speech::dsp::stft;

namespace {
std::vector<float> ramp(std::size_t n) {
    std::vector<float> v(n);
    std::iota(v.begin(), v.end(), 0.0f);
    return v;
}
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

// ---- Span ---------------------------------------------------------------------------------

UTEST(span, converts_from_vectors_arrays_and_mutable_to_const) {
    std::vector<float> v = ramp(5);
    const std::vector<float>& cv = v;
    float arr[3] = {7.0f, 8.0f, 9.0f};

    span<float> a(v);
    span<const float> b(cv);   // const vector -> const view
    span<const float> c(a);    // span<float> -> span<const float>
    span<float> d(arr);
    ASSERT_EQ(a.size(), 5u);
    ASSERT_EQ(b.data(), v.data());
    ASSERT_EQ(c.size(), 5u);
    ASSERT_EQ(d.size(), 3u);
    ASSERT_EQ(d[2], 9.0f);
    ASSERT_TRUE(span<float>().empty());

    a[0] = 42.0f;  // a mutable view writes through
    ASSERT_EQ(v[0], 42.0f);
    float sum = 0.0f;
    for (float x : b) sum += x;  // range-for
    ASSERT_EQ(sum, 42.0f + 1 + 2 + 3 + 4);
}

UTEST(span, subspan_slices_and_checks_bounds) {
    const std::vector<float> v = ramp(10);
    span<const float> s(v);
    ASSERT_EQ(s.subspan(3).size(), 7u);
    ASSERT_EQ(s.subspan(3, 4)[0], 3.0f);
    ASSERT_EQ(s.subspan(10).size(), 0u);  // an empty tail is fine
    bool offset_throws = false, count_throws = false;
    try { (void)s.subspan(11); } catch (const std::out_of_range&) { offset_throws = true; }
    try { (void)s.subspan(8, 3); } catch (const std::out_of_range&) { count_throws = true; }
    ASSERT_TRUE(offset_throws);
    ASSERT_TRUE(count_throws);
}

// ---- frame_view ----------------------------------------------------------------------------

UTEST(frame_view, rows_are_walked_frame_by_frame) {
    const std::vector<float> v = ramp(12);  // 4 rows of 3
    const auto frames = frame_view<const float>::rows(span<const float>(v), 3);
    ASSERT_EQ(frames.size(), 4u);
    ASSERT_EQ(frames.frame_length(), 3u);
    ASSERT_TRUE(frames.contiguous());
    ASSERT_EQ(frames.extent(), 12u);

    std::size_t i = 0;
    for (span<const float> frame : frames) {  // the iterator hands out one Span per frame
        ASSERT_EQ(frame.size(), 3u);
        ASSERT_EQ(frame[0], static_cast<float>(i * 3));
        ASSERT_EQ(frame[2], static_cast<float>(i * 3 + 2));
        ++i;
    }
    ASSERT_EQ(i, 4u);
    ASSERT_EQ(frames[2][1], 7.0f);
    ASSERT_EQ(frames.front()[0], 0.0f);
    ASSERT_EQ(frames.back()[2], 11.0f);
}

UTEST(frame_view, sliding_frames_overlap_like_the_stft) {
    const std::vector<float> v = ramp(20);
    const auto frames = frame_view<const float>::sliding(span<const float>(v), 8, 4);
    ASSERT_EQ(frames.size(), 4u);      // (20 - 8) / 4 + 1
    ASSERT_EQ(frames.hop(), 4u);
    ASSERT_TRUE(!frames.contiguous());
    ASSERT_EQ(frames.extent(), 20u);
    ASSERT_EQ(frames[1][0], 4.0f);
    ASSERT_EQ(frames[1][7], 11.0f);    // overlaps frame 0's last four samples
    ASSERT_EQ(frames[3][7], 19.0f);

    ASSERT_EQ(frame_view<const float>::sliding(span<const float>(v.data(), 7), 8, 4).size(), 0u);  // too short
    ASSERT_TRUE(throws_invalid_argument([&] { frame_view<const float>::sliding(span<const float>(v), 0, 4); }));
    ASSERT_TRUE(throws_invalid_argument([&] { frame_view<const float>::sliding(span<const float>(v), 8, 0); }));
}

UTEST(frame_view, iterator_supports_random_access_arithmetic) {
    const std::vector<float> v = ramp(30);
    const auto frames = frame_view<const float>::rows(span<const float>(v), 5);  // 6 rows
    auto it = frames.begin();
    ASSERT_EQ((*(it + 2))[0], 10.0f);
    ASSERT_EQ(it[3][0], 15.0f);
    ASSERT_EQ(frames.end() - frames.begin(), 6);
    ASSERT_TRUE(it < frames.end());
    ASSERT_TRUE(it != frames.end());
    it += 5;
    ASSERT_EQ((*it)[4], 29.0f);
    --it;
    ASSERT_EQ((*it)[0], 20.0f);
    ASSERT_EQ(std::distance(frames.begin(), frames.end()), 6);
}

UTEST(frame_view, validates_shape_and_converts_to_const) {
    std::vector<float> v = ramp(10);
    ASSERT_TRUE(throws_invalid_argument([&] { frame_view<float>::rows(span<float>(v), 3); }));  // 10 % 3 != 0
    ASSERT_TRUE(throws_invalid_argument([&] { frame_view<float>::rows(span<float>(v), 0); }));
    ASSERT_TRUE(throws_invalid_argument([&] { frame_view<float>(nullptr, 2, 4); }));

    frame_view<float> mut = frame_view<float>::rows(span<float>(v), 5);
    frame_view<const float> ro = mut;  // frame_view<float> -> frame_view<const float>
    mut[1][0] = -1.0f;                // writes go through to the buffer
    ASSERT_EQ(v[5], -1.0f);
    ASSERT_EQ(ro[1][0], -1.0f);

    ASSERT_EQ(mut.slice(1, 1).size(), 1u);
    ASSERT_EQ(mut.slice(1, 1)[0][0], -1.0f);
    bool slice_throws = false;
    try { (void)mut.slice(1, 2); } catch (const std::out_of_range&) { slice_throws = true; }
    ASSERT_TRUE(slice_throws);
    ASSERT_EQ(mut.flat().size(), 10u);
    ASSERT_TRUE(frame_view<const float>().empty());
}

// ---- STFT overloads -----------------------------------------------------------------------

UTEST(span_stft, frames_of_the_signal_are_what_the_stft_analyses) {
    stft stft(8, window_type::hann, 64);
    const std::vector<float> x = noise(256 * 5, 1u);
    const auto frames = stft.frames(span<const float>(x));
    ASSERT_EQ(static_cast<int>(frames.size()), stft.cal_time_length(static_cast<int>(x.size())));
    ASSERT_EQ(frames.frame_length(), 256u);
    ASSERT_EQ(frames[3].data(), x.data() + 3 * 64);
}

UTEST(span_stft, view_forms_give_exactly_the_pointer_results) {
    stft stft(9, window_type::hann, 128);
    const std::vector<float> x = noise(512 * 9, 2u);
    const int frames = stft.cal_time_length(static_cast<int>(x.size()));

    for (bool onesided : {false, true}) {
        const int bins = stft.num_bins(onesided);
        std::vector<float> re_a(static_cast<std::size_t>(frames) * bins), im_a(re_a.size());
        std::vector<float> re_b(re_a.size()), im_b(re_a.size());
        if (onesided) {
            stft.stft_onesided_into(x.data(), x.size(), re_a.data(), im_a.data());
            const int got = stft.stft_onesided_into(span<const float>(x), frame_view<float>::rows(span<float>(re_b), bins),
                                                  frame_view<float>::rows(span<float>(im_b), bins));
            ASSERT_EQ(got, frames);
        } else {
            stft.stft_into(x.data(), x.size(), re_a.data(), im_a.data());
            const int got = stft.stft_into(span<const float>(x), frame_view<float>::rows(span<float>(re_b), bins),
                                          frame_view<float>::rows(span<float>(im_b), bins));
            ASSERT_EQ(got, frames);
        }
        ASSERT_TRUE(re_a == re_b);
        ASSERT_TRUE(im_a == im_b);

        // and back: the view istft (full vs one-sided inferred from the row length) equals the pointer one
        for (int method : {0, 1}) {
            const auto via_ptr = stft.istft(re_a.data(), im_a.data(), frames, onesided, method);
            const auto via_view = stft.istft(frame_view<const float>::rows(span<const float>(re_a), bins),
                                            frame_view<const float>::rows(span<const float>(im_a), bins), method);
            ASSERT_EQ(via_ptr.size(), via_view.size());
            ASSERT_TRUE(std::equal(via_ptr.begin(), via_ptr.end(), via_view.begin()));
        }
    }
}

UTEST(span_stft, spectrogram_exposes_its_planes_as_frames) {
    stft stft(8, window_type::hann);
    const std::vector<float> x = noise(256 * 6, 3u);
    auto spec = stft.spectrogram(x.data(), x.size(), /*onesided=*/true);
    ASSERT_EQ(spec.real_frames().size(), static_cast<std::size_t>(spec.num_frames));
    ASSERT_EQ(spec.real_frames().frame_length(), 129u);
    int i = 0;
    for (span<const float> frame : static_cast<const speech::dsp::complex_spectrogram&>(spec).real_frames()) {
        ASSERT_EQ(frame.data(), spec.real.data() + static_cast<std::size_t>(i) * 129);
        ++i;
    }
    ASSERT_EQ(i, spec.num_frames);
    // the ComplexSpectrogram form of istft and the view form agree
    const auto a = stft.istft(spec, 0);
    const auto b = stft.istft(spec.real_frames(), spec.imag_frames(), 0);
    ASSERT_TRUE(std::equal(a.begin(), a.end(), b.begin()));
}

UTEST(span_stft, rejects_views_that_do_not_fit) {
    stft stft(6);  // 64-sample frames
    const std::vector<float> x = noise(64 * 6, 4u);
    const int frames = stft.cal_time_length(static_cast<int>(x.size()));
    std::vector<float> re(static_cast<std::size_t>(frames) * 64), im(re.size());

    // wrong row length, too few frames, strided (non-contiguous) output
    ASSERT_TRUE(throws_invalid_argument([&] {
        stft.stft_into(span<const float>(x), frame_view<float>::rows(span<float>(re), 32), frame_view<float>::rows(span<float>(im), 32));
    }));
    ASSERT_TRUE(throws_invalid_argument([&] {
        stft.stft_into(span<const float>(x), frame_view<float>(re.data(), 1, 64), frame_view<float>(im.data(), 1, 64));
    }));
    ASSERT_TRUE(throws_invalid_argument([&] {
        stft.stft_into(span<const float>(x), frame_view<float>(re.data(), frames, 64, 65), frame_view<float>(im.data(), frames, 64, 65));
    }));

    // istft: bad row length, mismatched real/imag, output too small, no frames
    const auto ok_re = frame_view<const float>(re.data(), frames, 64);
    const auto ok_im = frame_view<const float>(im.data(), frames, 64);
    std::vector<float> small(10);
    ASSERT_TRUE(throws_invalid_argument([&] { stft.istft(frame_view<const float>(re.data(), frames, 48), frame_view<const float>(im.data(), frames, 48)); }));
    ASSERT_TRUE(throws_invalid_argument([&] { stft.istft(ok_re, ok_im.slice(0, frames - 1)); }));
    ASSERT_TRUE(throws_invalid_argument([&] { stft.istft_into(ok_re, ok_im, 0, span<float>(small)); }));
    ASSERT_TRUE(throws_invalid_argument([&] { stft.istft(frame_view<const float>(), frame_view<const float>()); }));
}

// ---- FFT overloads ------------------------------------------------------------------------

UTEST(span_fft, view_forms_give_exactly_the_pointer_results) {
    const int radix = 10, N = 1 << radix;
    fft fft(radix);
    const std::vector<float> x = noise(N, 5u), y = noise(N, 6u);
    std::vector<float> re_a(N), im_a(N), re_b(N), im_b(N);

    fft.forward_into(x.data(), nullptr, re_a.data(), im_a.data());
    fft.forward_into(x, {}, re_b, im_b);  // vectors convert to Span; {} = no imaginary part
    ASSERT_TRUE(re_a == re_b);
    ASSERT_TRUE(im_a == im_b);

    fft.forward_into(x.data(), y.data(), re_a.data(), im_a.data());
    fft.forward_into(x, y, re_b, im_b);  // complex input
    ASSERT_TRUE(re_a == re_b);
    ASSERT_TRUE(im_a == im_b);

    fft.inverse_into(x.data(), y.data(), re_a.data(), im_a.data());
    fft.inverse_into(x, y, re_b, im_b);
    ASSERT_TRUE(re_a == re_b);
    ASSERT_TRUE(im_a == im_b);

    for (bool norm : {true, false}) {
        fft.dct_into(x.data(), re_a.data(), norm);
        fft.dct_into(x, re_b, norm);
        ASSERT_TRUE(re_a == re_b);
        fft.idct_into(x.data(), re_a.data(), norm);
        fft.idct_into(x, re_b, norm);
        ASSERT_TRUE(re_a == re_b);
    }
}

UTEST(span_fft, rejects_wrong_sizes) {
    fft fft(6);  // 64 samples
    std::vector<float> ok(64), short_v(63), out(64), out_imag(64), short_out(32);
    ASSERT_TRUE(throws_invalid_argument([&] { fft.forward_into(short_v, {}, out, out_imag); }));
    ASSERT_TRUE(throws_invalid_argument([&] { fft.forward_into(ok, short_v, out, out_imag); }));
    ASSERT_TRUE(throws_invalid_argument([&] { fft.forward_into(ok, {}, short_out, out_imag); }));
    ASSERT_TRUE(throws_invalid_argument([&] { fft.inverse_into(ok, {}, out, out_imag); }));  // inverse needs both parts
    ASSERT_TRUE(throws_invalid_argument([&] { fft.dct_into(short_v, out); }));
    ASSERT_TRUE(throws_invalid_argument([&] { fft.idct_into(ok, short_out); }));
}
