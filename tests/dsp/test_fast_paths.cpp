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
#include <vector>

#include "libspeech/dsp/dct.h"
#include "libspeech/dsp/mfcc.h"
#include "libspeech/dsp/resample.h"
#include "libspeech/dsp/stft.h"

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
