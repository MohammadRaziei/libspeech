// Tests for the allocation-free / copy-free fast paths: DctIIPlan (precomputed
// basis), STFT::spectrogram/stft_into (flat output), MFCC::compute_flat (blocked,
// sparse mel filters) and Resample's polyphase kernel + resample_flat.
//
// Each fast path is pinned to an independent reference (the original
// algorithm, written out naively here, or the old kernel kept as a fallback),
// not just to itself.
#include "utest.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "libspeech/dsp/dct.h"
#include "libspeech/dsp/mfcc.h"
#include "libspeech/dsp/resample.h"
#include "libspeech/dsp/simd.h"
#include "libspeech/dsp/stft.h"
#include "libspeech/dsp/window.h"
#include "stft_algorithm.h"  // vendored AudioFlux C API: the reference the real-FFT path must match

using speech::dsp::window_type;

namespace {

std::vector<float> speech_like(int n, int sample_rate) {
    std::vector<float> s(n);
    for (int i = 0; i < n; ++i) {
        const double t = static_cast<double>(i) / sample_rate;
        s[i] = static_cast<float>(0.4 * std::sin(2.0 * M_PI * 220.0 * t) +
                                  0.2 * std::sin(2.0 * M_PI * 1900.0 * t + 0.3 * std::sin(7.0 * t)));
    }
    return s;
}

}  // namespace

// ---- DctIIPlan ------------------------------------------------------------

UTEST(fast_paths, dct_ii_class_matches_function) {
    std::vector<float> in(26);
    for (size_t i = 0; i < in.size(); ++i) in[i] = std::sin(0.37f * i) * 5.0f - 1.0f;

    for (bool ortho : {true, false}) {
        auto ref = speech::dsp::dct_ii(in, 13, ortho);
        speech::dsp::dct_ii_plan dct(26, 13, ortho);
        std::vector<float> out(13);
        dct.apply(in.data(), out.data());
        for (int k = 0; k < 13; ++k) {
            ASSERT_TRUE(std::fabs(out[k] - ref[k]) < 1e-4f * std::max(1.0f, std::fabs(ref[k])));
        }
    }
}

UTEST(fast_paths, dct_ii_rejects_bad_sizes) {
    bool threw = false;
    try {
        speech::dsp::dct_ii_plan bad(10, 11);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    ASSERT_TRUE(threw);
}

// ---- dct_ii (free function) ---------------------------------------------

namespace {
// The original definition, evaluated naively in double precision.
std::vector<double> naive_dct(const std::vector<float>& x, int max_value, bool ortho) {
    const int n = static_cast<int>(x.size());
    std::vector<double> y(max_value);
    for (int k = 0; k < max_value; ++k) {
        double s = 0.0;
        for (int i = 0; i < n; ++i) s += x[i] * std::cos(M_PI * (2.0 * i + 1.0) * k / (2.0 * n));
        s *= 2.0;
        if (ortho) s *= (k == 0) ? std::sqrt(1.0 / (4.0 * n)) : std::sqrt(1.0 / (2.0 * n));
        y[k] = s;
    }
    return y;
}
}  // namespace

UTEST(fast_paths, dct_ii_matches_the_definition_for_every_size_and_path) {
    // Power-of-two sizes 16..8192 take the FFT path; 16384 is beyond the engine and 26/100/1000
    // are not powers of two, so those use the table-driven direct sum.
    for (int n : {16, 26, 64, 100, 512, 1000, 4096, 8192, 16384}) {
        std::vector<float> x(n);
        for (int i = 0; i < n; ++i) x[i] = std::sin(0.013f * i) * 3.0f + std::cos(0.9f * i) - 0.5f;
        for (int max_value : {1, 13, 25, 40, n}) {
            if (max_value > n || (n >= 8192 && max_value == n && n > 8192)) continue;  // keep the O(n^2) reference affordable
            for (bool ortho : {true, false}) {
                auto got = speech::dsp::dct_ii(x, max_value, ortho);
                auto ref = naive_dct(x, max_value, ortho);
                ASSERT_EQ(got.size(), static_cast<size_t>(max_value));
                double scale = 1.0;
                for (double v : ref) scale = std::max(scale, std::fabs(v));
                for (int k = 0; k < max_value; ++k) ASSERT_TRUE(std::fabs(got[k] - ref[k]) < 2e-5 * scale);
            }
        }
    }
}

// ---- STFT -------------------------------------------------------------

UTEST(fast_paths, spectrogram_matches_nested_stft) {
    auto sig = speech_like(16000, 16000);
    speech::dsp::stft a(9, window_type::hann, 128);
    speech::dsp::stft b(9, window_type::hann, 128);

    auto nested = a.compute(sig);
    auto flat = b.spectrogram(sig.data(), sig.size());

    ASSERT_EQ(flat.fft_length, 512);
    ASSERT_EQ(static_cast<size_t>(flat.num_frames), nested.first.size());
    for (int f = 0; f < flat.num_frames; ++f) {
        for (int k = 0; k < flat.fft_length; ++k) {
            const size_t i = static_cast<size_t>(f) * flat.fft_length + k;
            ASSERT_EQ(flat.real[i], nested.first[f][k]);
            ASSERT_EQ(flat.imag[i], nested.second[f][k]);
        }
    }
}

UTEST(fast_paths, stft_matches_naive_dft_at_every_supported_size) {
    // Independent reference: a double-precision O(N^2) DFT of the windowed frame.
    for (int expo = 4; expo <= 13; ++expo) {
        const int N = 1 << expo, hop = std::max(1, N / 4);
        auto sig = speech_like(N * 2 + hop, 16000);
        speech::dsp::stft st(expo, window_type::hann, hop);
        auto sp = st.spectrogram(sig.data(), sig.size(), /*onesided=*/false);
        auto w = speech::dsp::window::generate(window_type::hann, N);
        ASSERT_TRUE(sp.num_frames >= 2);
        ASSERT_EQ(sp.num_bins, N);
        for (int f = 0; f < 2; ++f) {
            for (int k = 0; k < N; k += std::max(1, N / 32)) {
                double re = 0.0, im = 0.0;
                for (int n = 0; n < N; ++n) {
                    const double v = static_cast<double>(sig[static_cast<size_t>(f) * hop + n]) * w[n];
                    const double a = -2.0 * M_PI * static_cast<double>(k) * n / N;
                    re += v * std::cos(a);
                    im += v * std::sin(a);
                }
                const size_t i = static_cast<size_t>(f) * N + k;
                ASSERT_TRUE(std::hypot(sp.real[i] - re, sp.imag[i] - im) < 1e-4 * std::max(1.0, std::hypot(re, im)));
            }
        }
    }
}

UTEST(fast_paths, real_fft_applies_the_same_window_as_audio_flux_for_every_window_type) {
    // ReferenceResult: AudioFlux's own complex STFT, driven through its C API exactly as the
    // pre-optimization implementation did. The real-FFT path builds its window with
    // window_calFFTWindow(), so this pins that the two stay in lockstep.
    const window_type types[] = {window_type::rect,    window_type::hann,     window_type::hamming,
                                window_type::blackman, window_type::bartlett, window_type::triang,
                                window_type::flattop, window_type::blackman_harris, window_type::blackman_nuttall,
                                window_type::bartlett_hann, window_type::bohman};
    const int expo = 9, N = 1 << expo, hop = 128;
    auto sig = speech_like(N * 4, 16000);
    for (window_type type : types) {
        speech::dsp::stft st(expo, type, hop);
        auto got = st.spectrogram(sig.data(), sig.size(), false);

        ::OpaqueSTFT* ref = nullptr;
        int slide = hop, is_continue = 0;
        ::WindowType t = static_cast<::WindowType>(static_cast<int>(type));  // AudioFlux's own enum
        ASSERT_EQ(stftObj_new(&ref, expo, &t, &slide, &is_continue), 0);
        std::vector<float> re(static_cast<size_t>(got.num_frames) * N), im(re.size());
        stftObj_stft(ref, sig.data(), static_cast<int>(sig.size()), re.data(), im.data());
        stftObj_free(ref);

        float max_mag = 0.0f;
        for (size_t i = 0; i < re.size(); ++i) max_mag = std::max(max_mag, std::hypot(re[i], im[i]));
        for (size_t i = 0; i < re.size(); ++i) {
            ASSERT_TRUE(std::fabs(got.real[i] - re[i]) < 1e-5f * max_mag);
            ASSERT_TRUE(std::fabs(got.imag[i] - im[i]) < 1e-5f * max_mag);
        }
    }
}

UTEST(fast_paths, onesided_is_the_first_half_of_the_full_spectrum) {
    // Includes sizes outside the real-FFT engine's range (N=8 and N=16384), which
    // take the AudioFlux fallback.
    for (int expo : {3, 4, 9, 13, 14}) {
        const int N = 1 << expo, hop = std::max(1, N / 4);
        auto sig = speech_like(N * 2 + hop, 16000);
        speech::dsp::stft st(expo, window_type::hann, hop);
        auto full = st.spectrogram(sig.data(), sig.size(), false);
        auto half = st.spectrogram(sig.data(), sig.size(), true);
        ASSERT_EQ(half.num_frames, full.num_frames);
        ASSERT_EQ(half.num_bins, N / 2 + 1);
        for (int f = 0; f < full.num_frames; ++f) {
            for (int k = 0; k < half.num_bins; ++k) {
                ASSERT_EQ(half.real[static_cast<size_t>(f) * half.num_bins + k],
                          full.real[static_cast<size_t>(f) * N + k]);
                ASSERT_EQ(half.imag[static_cast<size_t>(f) * half.num_bins + k],
                          full.imag[static_cast<size_t>(f) * N + k]);
            }
        }
    }
}

UTEST(fast_paths, full_spectrum_is_conjugate_symmetric) {
    auto sig = speech_like(4096, 16000);
    speech::dsp::stft st(9, window_type::hann, 128);
    auto sp = st.spectrogram(sig.data(), sig.size(), false);
    for (int f = 0; f < sp.num_frames; ++f) {
        const size_t o = static_cast<size_t>(f) * 512;
        for (int k = 1; k < 256; ++k) {
            ASSERT_EQ(sp.real[o + 512 - k], sp.real[o + k]);
            ASSERT_EQ(sp.imag[o + 512 - k], -sp.imag[o + k]);
        }
        ASSERT_EQ(sp.imag[o], 0.0f);        // DC bin of a real signal is real
        ASSERT_EQ(sp.imag[o + 256], 0.0f);  // so is Nyquist
    }
}

UTEST(fast_paths, power_spectra_match_onesided_stft) {
    auto sig = speech_like(8192, 16000);
    speech::dsp::stft st(9, window_type::hann, 128);
    auto half = st.spectrogram(sig.data(), sig.size(), true);
    std::vector<float> power(static_cast<size_t>(10) * 257);
    st.power_spectra(sig.data(), sig.size(), 5, 10, power.data());  // frames 5..14
    for (int f = 0; f < 10; ++f) {
        for (int k = 0; k < 257; ++k) {
            const size_t i = static_cast<size_t>(f + 5) * 257 + k;
            const float ref = half.real[i] * half.real[i] + half.imag[i] * half.imag[i];
            ASSERT_TRUE(std::fabs(power[static_cast<size_t>(f) * 257 + k] - ref) < 1e-4f * std::max(1.0f, ref));
        }
    }
}

UTEST(fast_paths, stft_shorter_than_one_frame_is_empty) {
    std::vector<float> sig(100, 0.5f);
    speech::dsp::stft st(9, window_type::hann, 128);
    auto sp = st.spectrogram(sig.data(), sig.size());
    ASSERT_EQ(sp.num_frames, 0);
    ASSERT_TRUE(sp.real.empty());
    ASSERT_EQ(st.stft_into(sig.data(), sig.size(), nullptr, nullptr), 0);
    ASSERT_TRUE(st.compute(sig).first.empty());
}

UTEST(fast_paths, stft_into_fully_overwrites_poisoned_buffers) {
    // The flat buffers are allocated uninitialized, so every element must be written.
    auto sig = speech_like(8000, 16000);
    speech::dsp::stft st(9, window_type::hann, 128);
    const int frames = st.cal_time_length(static_cast<int>(sig.size()));
    std::vector<float> re(static_cast<size_t>(frames) * 512, NAN);
    std::vector<float> im(static_cast<size_t>(frames) * 512, NAN);
    ASSERT_EQ(st.stft_into(sig.data(), sig.size(), re.data(), im.data()), frames);
    for (size_t i = 0; i < re.size(); ++i) {
        ASSERT_TRUE(std::isfinite(re[i]));
        ASSERT_TRUE(std::isfinite(im[i]));
    }
}

// ---- MFCC -------------------------------------------------------------

namespace {
// The original MFCC algorithm, naively: full STFT, dense triangular mel bank,
// per-frame std::cos DCT. Mel construction mirrors MFCC's own.
std::vector<std::vector<float>> reference_mfcc(const std::vector<float>& sig, int sample_rate) {
    const int expo = 9, hop = 128, num_mel = 26, num_coef = 13;
    const int fft_length = 1 << expo;
    const int num_bins = fft_length / 2 + 1;
    speech::dsp::stft st(expo, window_type::hann, hop);
    auto spec = st.compute(sig);

    auto hz2mel = [](float hz) { return 2595.0f * std::log10(1.0f + hz / 700.0f); };
    auto mel2hz = [](float mel) { return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f); };
    std::vector<int> pts(num_mel + 2);
    for (int i = 0; i < num_mel + 2; ++i) {
        const float mel = hz2mel(0.0f) + (hz2mel(sample_rate / 2.0f) - hz2mel(0.0f)) * i / (num_mel + 1);
        pts[i] = std::clamp(static_cast<int>(std::floor((fft_length + 1) * mel2hz(mel) / sample_rate)),
                            0, num_bins - 1);
    }
    std::vector<std::vector<float>> bank(num_mel, std::vector<float>(num_bins, 0.0f));
    for (int m = 0; m < num_mel; ++m) {
        const int l = pts[m], c = pts[m + 1], r = pts[m + 2];
        for (int k = l; k < c; ++k) bank[m][k] = static_cast<float>(k - l) / (c - l);
        for (int k = c; k < r; ++k) bank[m][k] = static_cast<float>(r - k) / (r - c);
    }
    std::vector<std::vector<float>> out;
    for (size_t f = 0; f < spec.first.size(); ++f) {
        std::vector<float> power(num_bins), log_mel(num_mel);
        for (int k = 0; k < num_bins; ++k) {
            power[k] = spec.first[f][k] * spec.first[f][k] + spec.second[f][k] * spec.second[f][k];
        }
        for (int m = 0; m < num_mel; ++m) {
            float e = 0.0f;
            for (int k = 0; k < num_bins; ++k) e += bank[m][k] * power[k];
            log_mel[m] = std::log(e + 1e-10f);
        }
        out.push_back(speech::dsp::dct_ii(log_mel, num_coef, true));
    }
    return out;
}

speech::dsp::mfcc::params mfcc_params() {
    speech::dsp::mfcc::params p;
    p.sample_rate = 16000;
    p.num_mel_filters = 26;
    p.num_coefficients = 13;
    p.radix2_exp = 9;
    p.slide_length = 128;
    return p;
}
}  // namespace

UTEST(fast_paths, mfcc_matches_original_algorithm) {
    // 200 frames: 3 full 64-frame blocks plus a partial one.
    auto sig = speech_like(512 + 128 * 199, 16000);
    speech::dsp::mfcc mfcc(mfcc_params());
    auto got = mfcc.compute_flat(sig.data(), sig.size());
    auto ref = reference_mfcc(sig, 16000);

    ASSERT_EQ(static_cast<size_t>(got.num_frames), ref.size());
    ASSERT_EQ(got.num_coefficients, 13);
    for (int f = 0; f < got.num_frames; ++f) {
        for (int c = 0; c < 13; ++c) {
            const float a = got.data[static_cast<size_t>(f) * 13 + c];
            ASSERT_TRUE(std::fabs(a - ref[f][c]) < 1e-3f * std::max(1.0f, std::fabs(ref[f][c])));
        }
    }
}

UTEST(fast_paths, mfcc_is_independent_of_block_boundaries) {
    // Frame f of a long signal must equal frame f of any prefix containing it,
    // whichever 64-frame block it lands in.
    auto sig = speech_like(512 + 128 * 149, 16000);  // 150 frames
    speech::dsp::mfcc mfcc(mfcc_params());
    auto full = mfcc.compute_flat(sig.data(), sig.size());
    for (int frames : {1, 2, 63, 64, 65, 127, 128, 129}) {
        const size_t len = 512 + 128 * static_cast<size_t>(frames - 1);
        auto part = mfcc.compute_flat(sig.data(), len);
        ASSERT_EQ(part.num_frames, frames);
        for (int f = 0; f < frames; ++f) {
            for (int c = 0; c < 13; ++c) {
                ASSERT_EQ(part.data[static_cast<size_t>(f) * 13 + c],
                          full.data[static_cast<size_t>(f) * 13 + c]);
            }
        }
    }
}

UTEST(fast_paths, mfcc_reuse_is_deterministic_and_short_input_is_empty) {
    auto sig = speech_like(16000, 16000);
    speech::dsp::mfcc mfcc(mfcc_params());
    auto a = mfcc.compute_flat(sig.data(), sig.size());
    auto b = mfcc.compute_flat(sig.data(), sig.size());
    ASSERT_EQ(a.data.size(), b.data.size());
    for (size_t i = 0; i < a.data.size(); ++i) ASSERT_EQ(a.data[i], b.data[i]);

    auto none = mfcc.compute_flat(sig.data(), 100);
    ASSERT_EQ(none.num_frames, 0);
    ASSERT_TRUE(none.data.empty());
}

// ---- Resample ---------------------------------------------------------

UTEST(fast_paths, polyphase_resample_matches_direct_kernel) {
    const int src = 44100, dst = 16000, n = 44100;
    std::vector<float> x(n);
    for (int i = 0; i < n; ++i) {
        x[i] = static_cast<float>(0.4 * std::sin(2.0 * M_PI * 300.0 * i / src) +
                                  0.1 * std::sin(2.0 * M_PI * 5000.0 * i / src));
    }
    speech::dsp::resample fast(src, dst);   // polyphase bank built (integer rates)
    speech::dsp::resample direct(src, dst);
    direct.enable_continuous(false);          // drops the bank -> original strided kernel

    auto yf = fast.process(x);
    auto yd = direct.process(x);
    ASSERT_EQ(yf.size(), yd.size());

    double max_diff = 0.0, max_err = 0.0;
    for (size_t i = 0; i < yf.size(); ++i) {
        ASSERT_TRUE(std::isfinite(yf[i]));
        max_diff = std::max(max_diff, static_cast<double>(std::fabs(yf[i] - yd[i])));
        if (i > 1000 && i + 1000 < yf.size()) {  // interior: compare with the analytic signal
            const double t = static_cast<double>(i) / dst;
            const double ref = 0.4 * std::sin(2.0 * M_PI * 300.0 * t) + 0.1 * std::sin(2.0 * M_PI * 5000.0 * t);
            max_err = std::max(max_err, std::fabs(yf[i] - ref));
        }
    }
    ASSERT_TRUE(max_diff < 5e-3);  // the two kernels differ only in table-interpolation rounding
    ASSERT_TRUE(max_err < 5e-3);   // and the fast one is a correct resampler
}

UTEST(fast_paths, polyphase_matches_direct_kernel_across_lengths_and_rates) {
    // Lengths straddle the interior/edge split (input windows that fit inside the
    // signal are read in place, the rest go through a small zero-padded buffer), and
    // the rate pairs cover down-, up- and near-unity ratios with different phase counts.
    struct pair_ { int src, dst; };
    for (pair_ rates : {pair_{44100, 16000}, pair_{48000, 16000}, pair_{22050, 16000}, pair_{16000, 48000},
                       pair_{48000, 44100}}) {
        for (int n : {50, 400, 1000, 3000, 5000, 12345, 44100, 100003}) {
            std::vector<float> x(n);
            for (int i = 0; i < n; ++i) {
                x[i] = static_cast<float>(0.4 * std::sin(2.0 * M_PI * 300.0 * i / rates.src) +
                                          0.1 * std::sin(2.0 * M_PI * 2000.0 * i / rates.src));
            }
            speech::dsp::resample fast(rates.src, rates.dst);
            speech::dsp::resample direct(rates.src, rates.dst);
            direct.enable_continuous(false);  // drops the polyphase bank -> original kernel

            std::vector<float> yf, yd;
            try {
                yf = fast.process(x);
            } catch (const std::runtime_error&) {
                bool also_throws = false;  // zero-length output: both kernels must agree
                try { direct.process(x); } catch (const std::runtime_error&) { also_throws = true; }
                ASSERT_TRUE(also_throws);
                continue;
            }
            yd = direct.process(x);
            ASSERT_EQ(yf.size(), yd.size());
            double max_diff = 0.0;
            for (size_t i = 0; i < yf.size(); ++i) {
                ASSERT_TRUE(std::isfinite(yf[i]));
                max_diff = std::max(max_diff, static_cast<double>(std::fabs(yf[i] - yd[i])));
            }
            ASSERT_TRUE(max_diff < 6e-3);
        }
    }
}

namespace {
double tone_at(int i, int sample_rate) {
    const double t = static_cast<double>(i) / sample_rate;
    return 0.1 * std::sin(2.0 * M_PI * 300.0 * t) + 0.1 * std::sin(2.0 * M_PI * 1200.0 * t + 0.7) +
           0.1 * std::sin(2.0 * M_PI * 2500.0 * t + 1.4);
}
}  // namespace

UTEST(fast_paths, exact_kaiser_design_is_accurate_against_the_analytic_signal) {
    // The polyphase bank is the closed-form windowed sinc, not a lookup table: in-band tones
    // come out within float rounding of the true signal (the old table-driven kernel was
    // only good to ~1e-3 at large down-ratios).
    struct pair_ { int src, dst; };
    for (pair_ r : {pair_{44100, 16000}, pair_{48000, 16000}, pair_{44100, 22050}, pair_{48000, 44100},
                   pair_{16000, 48000}, pair_{22050, 16000}}) {
        const int n = r.src;  // 1 s
        std::vector<float> x(n);
        for (int i = 0; i < n; ++i) x[i] = static_cast<float>(tone_at(i, r.src));
        speech::dsp::resample rs(r.src, r.dst);
        auto y = rs.process(x);
        const int edge = r.dst / 50;
        double max_err = 0.0;
        for (size_t i = edge; i + edge < y.size(); ++i) {
            max_err = std::max(max_err, std::fabs(y[i] - tone_at(static_cast<int>(i), r.dst)));
        }
        ASSERT_TRUE(max_err < 2e-6);
    }
}

UTEST(fast_paths, resample_long_input_runs_every_chunk_and_matches_the_direct_kernel) {
    // Long enough to be split into parallel chunks (>= 4 chunks of 128 groups of 8 phases-worth).
    const int src = 44100, dst = 16000, n = src * 12;
    std::vector<float> x(n);
    for (int i = 0; i < n; ++i) x[i] = static_cast<float>(tone_at(i, src));
    speech::dsp::resample fast(src, dst);
    speech::dsp::resample direct(src, dst);
    direct.enable_continuous(false);
    auto yf = fast.process(x);
    auto yd = direct.process(x);
    ASSERT_EQ(yf.size(), yd.size());
    ASSERT_EQ(yf.size(), static_cast<size_t>(dst) * 12);
    double max_diff = 0.0, max_err = 0.0;
    for (size_t i = 0; i < yf.size(); ++i) {
        max_diff = std::max(max_diff, static_cast<double>(std::fabs(yf[i] - yd[i])));
        if (i > 1000 && i + 1000 < yf.size()) {
            max_err = std::max(max_err, std::fabs(yf[i] - tone_at(static_cast<int>(i), dst)));
        }
    }
    ASSERT_TRUE(max_diff < 3e-3);  // the old kernel's own error bound
    ASSERT_TRUE(max_err < 2e-6);   // chunk seams included
}

UTEST(fast_paths, cached_banks_are_shared_correctly_and_keyed_by_every_parameter) {
    auto x = speech_like(30000, 44100);
    speech::dsp::resample a(44100, 16000);
    speech::dsp::resample b(44100, 16000);  // same key: served from the cache
    auto ya = a.process(x);
    auto yb = b.process(x);
    ASSERT_EQ(ya.size(), yb.size());
    for (size_t i = 0; i < ya.size(); ++i) ASSERT_EQ(ya[i], yb[i]);

    // Different rates, same source: a distinct bank, not the cached one.
    speech::dsp::resample c(44100, 22050);
    ASSERT_EQ(c.process(x).size(), static_cast<size_t>(std::floor(x.size() * 22050.0 / 44100.0)));

    // Same rates, different Kaiser beta: must NOT reuse the default-quality bank.
    speech::dsp::resample d(44100, 16000, 64, 9, window_type::kaiser, 4.0f, 0.9475937f, false, false);
    auto yd = d.process(x);
    ASSERT_EQ(yd.size(), ya.size());
    double max_diff = 0.0;
    for (size_t i = 0; i < ya.size(); ++i) max_diff = std::max(max_diff, static_cast<double>(std::fabs(ya[i] - yd[i])));
    ASSERT_TRUE(max_diff > 1e-4);  // a lower beta has a visibly different filter
}

UTEST(fast_paths, non_kaiser_custom_window_still_resamples_via_audio_flux_tables) {
    auto x = speech_like(20000, 44100);
    speech::dsp::resample hann(44100, 16000, 32, 9, window_type::hann, 0.0f, 0.945f, false, false);
    speech::dsp::resample hann_direct(44100, 16000, 32, 9, window_type::hann, 0.0f, 0.945f, false, false);
    hann_direct.enable_continuous(false);
    auto y = hann.process(x);
    auto yd = hann_direct.process(x);
    ASSERT_EQ(y.size(), yd.size());
    for (size_t i = 0; i < y.size(); ++i) {
        ASSERT_TRUE(std::isfinite(y[i]));
        ASSERT_TRUE(std::fabs(y[i] - yd[i]) < 6e-3f);
    }
}

UTEST(fast_paths, resample_flat_equals_resample_and_is_repeatable) {
    auto x = speech_like(30000, 44100);
    speech::dsp::resample r(44100, 16000);
    auto a = r.process(x);
    auto b = r.resample_flat(x.data(), x.size());
    auto c = r.resample_flat(x.data(), x.size());
    ASSERT_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        ASSERT_EQ(a[i], b[i]);
        ASSERT_EQ(b[i], c[i]);
    }
}

UTEST(fast_paths, resample_handles_input_shorter_than_the_filter) {
    // Fewer samples than the filter spans: every output touches the zero-padded edges.
    for (int n : {3, 10, 100, 500}) {
        std::vector<float> x(n, 0.25f);
        speech::dsp::resample r(44100, 16000);
        auto y = r.process(x);
        ASSERT_EQ(y.size(), static_cast<size_t>(std::floor(n * 16000.0 / 44100.0)));
        for (float v : y) ASSERT_TRUE(std::isfinite(v));
    }
}

UTEST(fast_paths, resample_that_yields_no_output_throws_on_both_kernels) {
    // 1 input sample at 44.1k -> 16k rounds down to 0 output samples. The original
    // kernel throws here; the polyphase path must keep that contract.
    std::vector<float> x(1, 0.25f);
    for (bool polyphase : {true, false}) {
        speech::dsp::resample r(44100, 16000);
        if (!polyphase) r.enable_continuous(false);
        bool threw = false;
        try {
            r.process(x);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        ASSERT_TRUE(threw);
    }
}

UTEST(fast_paths, resample_identity_and_upsample) {
    auto x = speech_like(8000, 16000);
    speech::dsp::resample same(16000, 16000);
    auto y = same.process(x);
    ASSERT_EQ(y.size(), x.size());
    for (size_t i = 0; i < x.size(); ++i) ASSERT_EQ(y[i], x[i]);

    speech::dsp::resample up(16000, 48000);  // ratio > 1 exercises the other table branch
    auto z = up.process(x);
    ASSERT_EQ(z.size(), x.size() * 3);
    for (float v : z) ASSERT_TRUE(std::isfinite(v));
}

// ---- SIMD backend selection ---------------------------------------------

namespace {
// Restores the global SIMD switch even if an assertion returns early.
struct simd_guard {
    bool saved = speech::dsp::simd::is_enabled();
    ~simd_guard() { speech::dsp::simd::set_enabled(saved); }
};
}  // namespace

UTEST(fast_paths, simd_backend_name_and_runtime_switch) {
    simd_guard guard;
    const std::string name = speech::dsp::simd::backend_name();
    ASSERT_TRUE(name == "generic" || name == "avx2+fma" || name == "neon");

    speech::dsp::simd::set_enabled(false);
    ASSERT_FALSE(speech::dsp::simd::is_enabled());
    ASSERT_TRUE(speech::dsp::simd::active_backend() == speech::dsp::simd::backend::generic);
    ASSERT_STREQ("generic", speech::dsp::simd::backend_name());

    speech::dsp::simd::set_enabled(true);
    ASSERT_STREQ(name.c_str(), speech::dsp::simd::backend_name());
}

UTEST(fast_paths, simd_and_generic_kernels_agree) {
    // Whatever vectorized backend this binary/CPU selects must match the portable
    // kernels to float rounding (different summation order only).
    simd_guard guard;
    auto sig16 = speech_like(24000, 16000);
    auto sig44 = speech_like(60000, 44100);
    // A noiseless synthetic tone leaves the upper mel bands at the float rounding
    // floor (~1e-11 power), where MFCC's log() turns rounding-order differences
    // between any two correct FFTs into O(0.1) errors. A 1% noise floor keeps every
    // band well above that, so the comparison measures the kernels, not log(noise).
    unsigned lcg = 12345u;
    for (float& v : sig16) {
        lcg = lcg * 1664525u + 1013904223u;
        v += 0.01f * (static_cast<float>(lcg >> 8) / 16777216.0f - 0.5f);
    }

    auto run = [&](bool simd_on, std::vector<float>& stft, std::vector<float>& mfcc,
                   std::vector<float>& rs) {
        speech::dsp::simd::set_enabled(simd_on);  // objects below pick their kernel at construction
        speech::dsp::stft st(9, window_type::hann, 128);
        auto sp = st.spectrogram(sig16.data(), sig16.size(), true);
        stft.assign(sp.real.begin(), sp.real.end());
        stft.insert(stft.end(), sp.imag.begin(), sp.imag.end());

        speech::dsp::mfcc m(mfcc_params());
        auto mm = m.compute_flat(sig16.data(), sig16.size());
        mfcc.assign(mm.data.begin(), mm.data.end());

        speech::dsp::resample r(44100, 16000);
        auto y = r.resample_flat(sig44.data(), sig44.size());
        rs.assign(y.begin(), y.end());
    };

    std::vector<float> s1, m1, r1, s0, m0, r0;
    run(true, s1, m1, r1);
    run(false, s0, m0, r0);

    ASSERT_EQ(s1.size(), s0.size());
    ASSERT_EQ(m1.size(), m0.size());
    ASSERT_EQ(r1.size(), r0.size());
    for (size_t i = 0; i < s1.size(); ++i) ASSERT_TRUE(std::fabs(s1[i] - s0[i]) < 1e-3f * std::max(1.0f, std::fabs(s0[i])));
    for (size_t i = 0; i < m1.size(); ++i) ASSERT_TRUE(std::fabs(m1[i] - m0[i]) < 1e-3f * std::max(1.0f, std::fabs(m0[i])));
    for (size_t i = 0; i < r1.size(); ++i) ASSERT_TRUE(std::fabs(r1[i] - r0[i]) < 1e-5f);
}
