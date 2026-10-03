// Tests for the allocation-free / copy-free fast paths: DctII (precomputed
// basis), STFT::spectrogram/stftInto (flat output), MFCC::computeFlat (blocked,
// sparse mel filters) and Resample's polyphase kernel + resampleFlat.
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

namespace {

std::vector<float> speechLike(int n, int sampleRate) {
    std::vector<float> s(n);
    for (int i = 0; i < n; ++i) {
        const double t = static_cast<double>(i) / sampleRate;
        s[i] = static_cast<float>(0.4 * std::sin(2.0 * M_PI * 220.0 * t) +
                                  0.2 * std::sin(2.0 * M_PI * 1900.0 * t + 0.3 * std::sin(7.0 * t)));
    }
    return s;
}

}  // namespace

// ---- DctII ------------------------------------------------------------

UTEST(FastPaths, DctIIClassMatchesFunction) {
    std::vector<float> in(26);
    for (size_t i = 0; i < in.size(); ++i) in[i] = std::sin(0.37f * i) * 5.0f - 1.0f;

    for (bool ortho : {true, false}) {
        auto ref = speech::dsp::dctII(in, 13, ortho);
        speech::dsp::DctII dct(26, 13, ortho);
        std::vector<float> out(13);
        dct.apply(in.data(), out.data());
        for (int k = 0; k < 13; ++k) {
            ASSERT_TRUE(std::fabs(out[k] - ref[k]) < 1e-4f * std::max(1.0f, std::fabs(ref[k])));
        }
    }
}

UTEST(FastPaths, DctIIRejectsBadSizes) {
    bool threw = false;
    try {
        speech::dsp::DctII bad(10, 11);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    ASSERT_TRUE(threw);
}

// ---- STFT -------------------------------------------------------------

UTEST(FastPaths, SpectrogramMatchesNestedStft) {
    auto sig = speechLike(16000, 16000);
    speech::dsp::STFT a(9, Window_Hann, 128);
    speech::dsp::STFT b(9, Window_Hann, 128);

    auto nested = a.stft(sig);
    auto flat = b.spectrogram(sig.data(), sig.size());

    ASSERT_EQ(flat.fftLength, 512);
    ASSERT_EQ(static_cast<size_t>(flat.numFrames), nested.first.size());
    for (int f = 0; f < flat.numFrames; ++f) {
        for (int k = 0; k < flat.fftLength; ++k) {
            const size_t i = static_cast<size_t>(f) * flat.fftLength + k;
            ASSERT_EQ(flat.real[i], nested.first[f][k]);
            ASSERT_EQ(flat.imag[i], nested.second[f][k]);
        }
    }
}

UTEST(FastPaths, StftMatchesNaiveDftAtEverySupportedSize) {
    // Independent reference: a double-precision O(N^2) DFT of the windowed frame.
    for (int expo = 4; expo <= 13; ++expo) {
        const int N = 1 << expo, hop = std::max(1, N / 4);
        auto sig = speechLike(N * 2 + hop, 16000);
        speech::dsp::STFT st(expo, Window_Hann, hop);
        auto sp = st.spectrogram(sig.data(), sig.size(), /*onesided=*/false);
        auto w = speech::dsp::window::generate(Window_Hann, N);
        ASSERT_TRUE(sp.numFrames >= 2);
        ASSERT_EQ(sp.numBins, N);
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

UTEST(FastPaths, RealFftAppliesTheSameWindowAsAudioFluxForEveryWindowType) {
    // Reference: AudioFlux's own complex STFT, driven through its C API exactly as the
    // pre-optimization implementation did. The real-FFT path builds its window with
    // window_calFFTWindow(), so this pins that the two stay in lockstep.
    const WindowType types[] = {Window_Rect,    Window_Hann,     Window_Hamm,
                                Window_Blackman, Window_Bartlett, Window_Triang,
                                Window_Flattop, Window_Blackman_Harris, Window_Blackman_Nuttall,
                                Window_Bartlett_Hann, Window_Bohman};
    const int expo = 9, N = 1 << expo, hop = 128;
    auto sig = speechLike(N * 4, 16000);
    for (WindowType type : types) {
        speech::dsp::STFT st(expo, type, hop);
        auto got = st.spectrogram(sig.data(), sig.size(), false);

        ::OpaqueSTFT* ref = nullptr;
        int slide = hop, isContinue = 0;
        WindowType t = type;
        ASSERT_EQ(stftObj_new(&ref, expo, &t, &slide, &isContinue), 0);
        std::vector<float> re(static_cast<size_t>(got.numFrames) * N), im(re.size());
        stftObj_stft(ref, sig.data(), static_cast<int>(sig.size()), re.data(), im.data());
        stftObj_free(ref);

        float maxMag = 0.0f;
        for (size_t i = 0; i < re.size(); ++i) maxMag = std::max(maxMag, std::hypot(re[i], im[i]));
        for (size_t i = 0; i < re.size(); ++i) {
            ASSERT_TRUE(std::fabs(got.real[i] - re[i]) < 1e-5f * maxMag);
            ASSERT_TRUE(std::fabs(got.imag[i] - im[i]) < 1e-5f * maxMag);
        }
    }
}

UTEST(FastPaths, OnesidedIsTheFirstHalfOfTheFullSpectrum) {
    // Includes sizes outside the real-FFT engine's range (N=8 and N=16384), which
    // take the AudioFlux fallback.
    for (int expo : {3, 4, 9, 13, 14}) {
        const int N = 1 << expo, hop = std::max(1, N / 4);
        auto sig = speechLike(N * 2 + hop, 16000);
        speech::dsp::STFT st(expo, Window_Hann, hop);
        auto full = st.spectrogram(sig.data(), sig.size(), false);
        auto half = st.spectrogram(sig.data(), sig.size(), true);
        ASSERT_EQ(half.numFrames, full.numFrames);
        ASSERT_EQ(half.numBins, N / 2 + 1);
        for (int f = 0; f < full.numFrames; ++f) {
            for (int k = 0; k < half.numBins; ++k) {
                ASSERT_EQ(half.real[static_cast<size_t>(f) * half.numBins + k],
                          full.real[static_cast<size_t>(f) * N + k]);
                ASSERT_EQ(half.imag[static_cast<size_t>(f) * half.numBins + k],
                          full.imag[static_cast<size_t>(f) * N + k]);
            }
        }
    }
}

UTEST(FastPaths, FullSpectrumIsConjugateSymmetric) {
    auto sig = speechLike(4096, 16000);
    speech::dsp::STFT st(9, Window_Hann, 128);
    auto sp = st.spectrogram(sig.data(), sig.size(), false);
    for (int f = 0; f < sp.numFrames; ++f) {
        const size_t o = static_cast<size_t>(f) * 512;
        for (int k = 1; k < 256; ++k) {
            ASSERT_EQ(sp.real[o + 512 - k], sp.real[o + k]);
            ASSERT_EQ(sp.imag[o + 512 - k], -sp.imag[o + k]);
        }
        ASSERT_EQ(sp.imag[o], 0.0f);        // DC bin of a real signal is real
        ASSERT_EQ(sp.imag[o + 256], 0.0f);  // so is Nyquist
    }
}

UTEST(FastPaths, PowerSpectraMatchOnesidedStft) {
    auto sig = speechLike(8192, 16000);
    speech::dsp::STFT st(9, Window_Hann, 128);
    auto half = st.spectrogram(sig.data(), sig.size(), true);
    std::vector<float> power(static_cast<size_t>(10) * 257);
    st.powerSpectra(sig.data(), sig.size(), 5, 10, power.data());  // frames 5..14
    for (int f = 0; f < 10; ++f) {
        for (int k = 0; k < 257; ++k) {
            const size_t i = static_cast<size_t>(f + 5) * 257 + k;
            const float ref = half.real[i] * half.real[i] + half.imag[i] * half.imag[i];
            ASSERT_TRUE(std::fabs(power[static_cast<size_t>(f) * 257 + k] - ref) < 1e-4f * std::max(1.0f, ref));
        }
    }
}

UTEST(FastPaths, StftShorterThanOneFrameIsEmpty) {
    std::vector<float> sig(100, 0.5f);
    speech::dsp::STFT st(9, Window_Hann, 128);
    auto sp = st.spectrogram(sig.data(), sig.size());
    ASSERT_EQ(sp.numFrames, 0);
    ASSERT_TRUE(sp.real.empty());
    ASSERT_EQ(st.stftInto(sig.data(), sig.size(), nullptr, nullptr), 0);
    ASSERT_TRUE(st.stft(sig).first.empty());
}

UTEST(FastPaths, StftIntoFullyOverwritesPoisonedBuffers) {
    // The flat buffers are allocated uninitialized, so every element must be written.
    auto sig = speechLike(8000, 16000);
    speech::dsp::STFT st(9, Window_Hann, 128);
    const int frames = st.calTimeLength(static_cast<int>(sig.size()));
    std::vector<float> re(static_cast<size_t>(frames) * 512, NAN);
    std::vector<float> im(static_cast<size_t>(frames) * 512, NAN);
    ASSERT_EQ(st.stftInto(sig.data(), sig.size(), re.data(), im.data()), frames);
    for (size_t i = 0; i < re.size(); ++i) {
        ASSERT_TRUE(std::isfinite(re[i]));
        ASSERT_TRUE(std::isfinite(im[i]));
    }
}

// ---- MFCC -------------------------------------------------------------

namespace {
// The original MFCC algorithm, naively: full STFT, dense triangular mel bank,
// per-frame std::cos DCT. Mel construction mirrors MFCC's own.
std::vector<std::vector<float>> referenceMfcc(const std::vector<float>& sig, int sampleRate) {
    const int expo = 9, hop = 128, numMel = 26, numCoef = 13;
    const int fftLength = 1 << expo;
    const int numBins = fftLength / 2 + 1;
    speech::dsp::STFT st(expo, Window_Hann, hop);
    auto spec = st.stft(sig);

    auto hz2mel = [](float hz) { return 2595.0f * std::log10(1.0f + hz / 700.0f); };
    auto mel2hz = [](float mel) { return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f); };
    std::vector<int> pts(numMel + 2);
    for (int i = 0; i < numMel + 2; ++i) {
        const float mel = hz2mel(0.0f) + (hz2mel(sampleRate / 2.0f) - hz2mel(0.0f)) * i / (numMel + 1);
        pts[i] = std::clamp(static_cast<int>(std::floor((fftLength + 1) * mel2hz(mel) / sampleRate)),
                            0, numBins - 1);
    }
    std::vector<std::vector<float>> bank(numMel, std::vector<float>(numBins, 0.0f));
    for (int m = 0; m < numMel; ++m) {
        const int l = pts[m], c = pts[m + 1], r = pts[m + 2];
        for (int k = l; k < c; ++k) bank[m][k] = static_cast<float>(k - l) / (c - l);
        for (int k = c; k < r; ++k) bank[m][k] = static_cast<float>(r - k) / (r - c);
    }
    std::vector<std::vector<float>> out;
    for (size_t f = 0; f < spec.first.size(); ++f) {
        std::vector<float> power(numBins), logMel(numMel);
        for (int k = 0; k < numBins; ++k) {
            power[k] = spec.first[f][k] * spec.first[f][k] + spec.second[f][k] * spec.second[f][k];
        }
        for (int m = 0; m < numMel; ++m) {
            float e = 0.0f;
            for (int k = 0; k < numBins; ++k) e += bank[m][k] * power[k];
            logMel[m] = std::log(e + 1e-10f);
        }
        out.push_back(speech::dsp::dctII(logMel, numCoef, true));
    }
    return out;
}

speech::dsp::MFCC::Params mfccParams() {
    speech::dsp::MFCC::Params p;
    p.sampleRate = 16000;
    p.numMelFilters = 26;
    p.numCoefficients = 13;
    p.radix2Exp = 9;
    p.slideLength = 128;
    return p;
}
}  // namespace

UTEST(FastPaths, MfccMatchesOriginalAlgorithm) {
    // 200 frames: 3 full 64-frame blocks plus a partial one.
    auto sig = speechLike(512 + 128 * 199, 16000);
    speech::dsp::MFCC mfcc(mfccParams());
    auto got = mfcc.computeFlat(sig.data(), sig.size());
    auto ref = referenceMfcc(sig, 16000);

    ASSERT_EQ(static_cast<size_t>(got.numFrames), ref.size());
    ASSERT_EQ(got.numCoefficients, 13);
    for (int f = 0; f < got.numFrames; ++f) {
        for (int c = 0; c < 13; ++c) {
            const float a = got.data[static_cast<size_t>(f) * 13 + c];
            ASSERT_TRUE(std::fabs(a - ref[f][c]) < 1e-3f * std::max(1.0f, std::fabs(ref[f][c])));
        }
    }
}

UTEST(FastPaths, MfccIsIndependentOfBlockBoundaries) {
    // Frame f of a long signal must equal frame f of any prefix containing it,
    // whichever 64-frame block it lands in.
    auto sig = speechLike(512 + 128 * 149, 16000);  // 150 frames
    speech::dsp::MFCC mfcc(mfccParams());
    auto full = mfcc.computeFlat(sig.data(), sig.size());
    for (int frames : {1, 2, 63, 64, 65, 127, 128, 129}) {
        const size_t len = 512 + 128 * static_cast<size_t>(frames - 1);
        auto part = mfcc.computeFlat(sig.data(), len);
        ASSERT_EQ(part.numFrames, frames);
        for (int f = 0; f < frames; ++f) {
            for (int c = 0; c < 13; ++c) {
                ASSERT_EQ(part.data[static_cast<size_t>(f) * 13 + c],
                          full.data[static_cast<size_t>(f) * 13 + c]);
            }
        }
    }
}

UTEST(FastPaths, MfccReuseIsDeterministicAndShortInputIsEmpty) {
    auto sig = speechLike(16000, 16000);
    speech::dsp::MFCC mfcc(mfccParams());
    auto a = mfcc.computeFlat(sig.data(), sig.size());
    auto b = mfcc.computeFlat(sig.data(), sig.size());
    ASSERT_EQ(a.data.size(), b.data.size());
    for (size_t i = 0; i < a.data.size(); ++i) ASSERT_EQ(a.data[i], b.data[i]);

    auto none = mfcc.computeFlat(sig.data(), 100);
    ASSERT_EQ(none.numFrames, 0);
    ASSERT_TRUE(none.data.empty());
}

// ---- Resample ---------------------------------------------------------

UTEST(FastPaths, PolyphaseResampleMatchesDirectKernel) {
    const int src = 44100, dst = 16000, n = 44100;
    std::vector<float> x(n);
    for (int i = 0; i < n; ++i) {
        x[i] = static_cast<float>(0.4 * std::sin(2.0 * M_PI * 300.0 * i / src) +
                                  0.1 * std::sin(2.0 * M_PI * 5000.0 * i / src));
    }
    speech::dsp::Resample fast(src, dst);   // polyphase bank built (integer rates)
    speech::dsp::Resample direct(src, dst);
    direct.enableContinuous(false);          // drops the bank -> original strided kernel

    auto yf = fast.resample(x);
    auto yd = direct.resample(x);
    ASSERT_EQ(yf.size(), yd.size());

    double maxDiff = 0.0, maxErr = 0.0;
    for (size_t i = 0; i < yf.size(); ++i) {
        ASSERT_TRUE(std::isfinite(yf[i]));
        maxDiff = std::max(maxDiff, static_cast<double>(std::fabs(yf[i] - yd[i])));
        if (i > 1000 && i + 1000 < yf.size()) {  // interior: compare with the analytic signal
            const double t = static_cast<double>(i) / dst;
            const double ref = 0.4 * std::sin(2.0 * M_PI * 300.0 * t) + 0.1 * std::sin(2.0 * M_PI * 5000.0 * t);
            maxErr = std::max(maxErr, std::fabs(yf[i] - ref));
        }
    }
    ASSERT_TRUE(maxDiff < 5e-3);  // the two kernels differ only in table-interpolation rounding
    ASSERT_TRUE(maxErr < 5e-3);   // and the fast one is a correct resampler
}

UTEST(FastPaths, PolyphaseMatchesDirectKernelAcrossLengthsAndRates) {
    // Lengths straddle the interior/edge split (input windows that fit inside the
    // signal are read in place, the rest go through a small zero-padded buffer), and
    // the rate pairs cover down-, up- and near-unity ratios with different phase counts.
    struct Pair { int src, dst; };
    for (Pair rates : {Pair{44100, 16000}, Pair{48000, 16000}, Pair{22050, 16000}, Pair{16000, 48000},
                       Pair{48000, 44100}}) {
        for (int n : {50, 400, 1000, 3000, 5000, 12345, 44100, 100003}) {
            std::vector<float> x(n);
            for (int i = 0; i < n; ++i) {
                x[i] = static_cast<float>(0.4 * std::sin(2.0 * M_PI * 300.0 * i / rates.src) +
                                          0.1 * std::sin(2.0 * M_PI * 2000.0 * i / rates.src));
            }
            speech::dsp::Resample fast(rates.src, rates.dst);
            speech::dsp::Resample direct(rates.src, rates.dst);
            direct.enableContinuous(false);  // drops the polyphase bank -> original kernel

            std::vector<float> yf, yd;
            try {
                yf = fast.resample(x);
            } catch (const std::runtime_error&) {
                bool alsoThrows = false;  // zero-length output: both kernels must agree
                try { direct.resample(x); } catch (const std::runtime_error&) { alsoThrows = true; }
                ASSERT_TRUE(alsoThrows);
                continue;
            }
            yd = direct.resample(x);
            ASSERT_EQ(yf.size(), yd.size());
            double maxDiff = 0.0;
            for (size_t i = 0; i < yf.size(); ++i) {
                ASSERT_TRUE(std::isfinite(yf[i]));
                maxDiff = std::max(maxDiff, static_cast<double>(std::fabs(yf[i] - yd[i])));
            }
            ASSERT_TRUE(maxDiff < 6e-3);
        }
    }
}

namespace {
double toneAt(int i, int sampleRate) {
    const double t = static_cast<double>(i) / sampleRate;
    return 0.1 * std::sin(2.0 * M_PI * 300.0 * t) + 0.1 * std::sin(2.0 * M_PI * 1200.0 * t + 0.7) +
           0.1 * std::sin(2.0 * M_PI * 2500.0 * t + 1.4);
}
}  // namespace

UTEST(FastPaths, ExactKaiserDesignIsAccurateAgainstTheAnalyticSignal) {
    // The polyphase bank is the closed-form windowed sinc, not a lookup table: in-band tones
    // come out within float rounding of the true signal (the old table-driven kernel was
    // only good to ~1e-3 at large down-ratios).
    struct Pair { int src, dst; };
    for (Pair r : {Pair{44100, 16000}, Pair{48000, 16000}, Pair{44100, 22050}, Pair{48000, 44100},
                   Pair{16000, 48000}, Pair{22050, 16000}}) {
        const int n = r.src;  // 1 s
        std::vector<float> x(n);
        for (int i = 0; i < n; ++i) x[i] = static_cast<float>(toneAt(i, r.src));
        speech::dsp::Resample rs(r.src, r.dst);
        auto y = rs.resample(x);
        const int edge = r.dst / 50;
        double maxErr = 0.0;
        for (size_t i = edge; i + edge < y.size(); ++i) {
            maxErr = std::max(maxErr, std::fabs(y[i] - toneAt(static_cast<int>(i), r.dst)));
        }
        ASSERT_TRUE(maxErr < 2e-6);
    }
}

UTEST(FastPaths, ResampleLongInputRunsEveryChunkAndMatchesTheDirectKernel) {
    // Long enough to be split into parallel chunks (>= 4 chunks of 128 groups of 8 phases-worth).
    const int src = 44100, dst = 16000, n = src * 12;
    std::vector<float> x(n);
    for (int i = 0; i < n; ++i) x[i] = static_cast<float>(toneAt(i, src));
    speech::dsp::Resample fast(src, dst);
    speech::dsp::Resample direct(src, dst);
    direct.enableContinuous(false);
    auto yf = fast.resample(x);
    auto yd = direct.resample(x);
    ASSERT_EQ(yf.size(), yd.size());
    ASSERT_EQ(yf.size(), static_cast<size_t>(dst) * 12);
    double maxDiff = 0.0, maxErr = 0.0;
    for (size_t i = 0; i < yf.size(); ++i) {
        maxDiff = std::max(maxDiff, static_cast<double>(std::fabs(yf[i] - yd[i])));
        if (i > 1000 && i + 1000 < yf.size()) {
            maxErr = std::max(maxErr, std::fabs(yf[i] - toneAt(static_cast<int>(i), dst)));
        }
    }
    ASSERT_TRUE(maxDiff < 3e-3);  // the old kernel's own error bound
    ASSERT_TRUE(maxErr < 2e-6);   // chunk seams included
}

UTEST(FastPaths, CachedBanksAreSharedCorrectlyAndKeyedByEveryParameter) {
    auto x = speechLike(30000, 44100);
    speech::dsp::Resample a(44100, 16000);
    speech::dsp::Resample b(44100, 16000);  // same key: served from the cache
    auto ya = a.resample(x);
    auto yb = b.resample(x);
    ASSERT_EQ(ya.size(), yb.size());
    for (size_t i = 0; i < ya.size(); ++i) ASSERT_EQ(ya[i], yb[i]);

    // Different rates, same source: a distinct bank, not the cached one.
    speech::dsp::Resample c(44100, 22050);
    ASSERT_EQ(c.resample(x).size(), static_cast<size_t>(std::floor(x.size() * 22050.0 / 44100.0)));

    // Same rates, different Kaiser beta: must NOT reuse the default-quality bank.
    speech::dsp::Resample d(44100, 16000, 64, 9, Window_Kaiser, 4.0f, 0.9475937f, false, false);
    auto yd = d.resample(x);
    ASSERT_EQ(yd.size(), ya.size());
    double maxDiff = 0.0;
    for (size_t i = 0; i < ya.size(); ++i) maxDiff = std::max(maxDiff, static_cast<double>(std::fabs(ya[i] - yd[i])));
    ASSERT_TRUE(maxDiff > 1e-4);  // a lower beta has a visibly different filter
}

UTEST(FastPaths, NonKaiserCustomWindowStillResamplesViaAudioFluxTables) {
    auto x = speechLike(20000, 44100);
    speech::dsp::Resample hann(44100, 16000, 32, 9, Window_Hann, 0.0f, 0.945f, false, false);
    speech::dsp::Resample hannDirect(44100, 16000, 32, 9, Window_Hann, 0.0f, 0.945f, false, false);
    hannDirect.enableContinuous(false);
    auto y = hann.resample(x);
    auto yd = hannDirect.resample(x);
    ASSERT_EQ(y.size(), yd.size());
    for (size_t i = 0; i < y.size(); ++i) {
        ASSERT_TRUE(std::isfinite(y[i]));
        ASSERT_TRUE(std::fabs(y[i] - yd[i]) < 6e-3f);
    }
}

UTEST(FastPaths, ResampleFlatEqualsResampleAndIsRepeatable) {
    auto x = speechLike(30000, 44100);
    speech::dsp::Resample r(44100, 16000);
    auto a = r.resample(x);
    auto b = r.resampleFlat(x.data(), x.size());
    auto c = r.resampleFlat(x.data(), x.size());
    ASSERT_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        ASSERT_EQ(a[i], b[i]);
        ASSERT_EQ(b[i], c[i]);
    }
}

UTEST(FastPaths, ResampleHandlesInputShorterThanTheFilter) {
    // Fewer samples than the filter spans: every output touches the zero-padded edges.
    for (int n : {3, 10, 100, 500}) {
        std::vector<float> x(n, 0.25f);
        speech::dsp::Resample r(44100, 16000);
        auto y = r.resample(x);
        ASSERT_EQ(y.size(), static_cast<size_t>(std::floor(n * 16000.0 / 44100.0)));
        for (float v : y) ASSERT_TRUE(std::isfinite(v));
    }
}

UTEST(FastPaths, ResampleThatYieldsNoOutputThrowsOnBothKernels) {
    // 1 input sample at 44.1k -> 16k rounds down to 0 output samples. The original
    // kernel throws here; the polyphase path must keep that contract.
    std::vector<float> x(1, 0.25f);
    for (bool polyphase : {true, false}) {
        speech::dsp::Resample r(44100, 16000);
        if (!polyphase) r.enableContinuous(false);
        bool threw = false;
        try {
            r.resample(x);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        ASSERT_TRUE(threw);
    }
}

UTEST(FastPaths, ResampleIdentityAndUpsample) {
    auto x = speechLike(8000, 16000);
    speech::dsp::Resample same(16000, 16000);
    auto y = same.resample(x);
    ASSERT_EQ(y.size(), x.size());
    for (size_t i = 0; i < x.size(); ++i) ASSERT_EQ(y[i], x[i]);

    speech::dsp::Resample up(16000, 48000);  // ratio > 1 exercises the other table branch
    auto z = up.resample(x);
    ASSERT_EQ(z.size(), x.size() * 3);
    for (float v : z) ASSERT_TRUE(std::isfinite(v));
}

// ---- SIMD backend selection ---------------------------------------------

namespace {
// Restores the global SIMD switch even if an assertion returns early.
struct SimdGuard {
    bool saved = speech::dsp::simd::isEnabled();
    ~SimdGuard() { speech::dsp::simd::setEnabled(saved); }
};
}  // namespace

UTEST(FastPaths, SimdBackendNameAndRuntimeSwitch) {
    SimdGuard guard;
    const std::string name = speech::dsp::simd::backendName();
    ASSERT_TRUE(name == "generic" || name == "avx2+fma" || name == "neon");

    speech::dsp::simd::setEnabled(false);
    ASSERT_FALSE(speech::dsp::simd::isEnabled());
    ASSERT_TRUE(speech::dsp::simd::activeBackend() == speech::dsp::simd::Backend::Generic);
    ASSERT_STREQ("generic", speech::dsp::simd::backendName());

    speech::dsp::simd::setEnabled(true);
    ASSERT_STREQ(name.c_str(), speech::dsp::simd::backendName());
}

UTEST(FastPaths, SimdAndGenericKernelsAgree) {
    // Whatever vectorized backend this binary/CPU selects must match the portable
    // kernels to float rounding (different summation order only).
    SimdGuard guard;
    auto sig16 = speechLike(24000, 16000);
    auto sig44 = speechLike(60000, 44100);
    // A noiseless synthetic tone leaves the upper mel bands at the float rounding
    // floor (~1e-11 power), where MFCC's log() turns rounding-order differences
    // between any two correct FFTs into O(0.1) errors. A 1% noise floor keeps every
    // band well above that, so the comparison measures the kernels, not log(noise).
    unsigned lcg = 12345u;
    for (float& v : sig16) {
        lcg = lcg * 1664525u + 1013904223u;
        v += 0.01f * (static_cast<float>(lcg >> 8) / 16777216.0f - 0.5f);
    }

    auto run = [&](bool simdOn, std::vector<float>& stft, std::vector<float>& mfcc,
                   std::vector<float>& rs) {
        speech::dsp::simd::setEnabled(simdOn);  // objects below pick their kernel at construction
        speech::dsp::STFT st(9, Window_Hann, 128);
        auto sp = st.spectrogram(sig16.data(), sig16.size(), true);
        stft.assign(sp.real.begin(), sp.real.end());
        stft.insert(stft.end(), sp.imag.begin(), sp.imag.end());

        speech::dsp::MFCC m(mfccParams());
        auto mm = m.computeFlat(sig16.data(), sig16.size());
        mfcc.assign(mm.data.begin(), mm.data.end());

        speech::dsp::Resample r(44100, 16000);
        auto y = r.resampleFlat(sig44.data(), sig44.size());
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
