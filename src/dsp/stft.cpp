#include "libspeech/dsp/stft.h"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>

#include "aixlog.hpp"
#include "libspeech/detail/log.h"
#include "dsp/flux_window.h"  // Vendored AudioFlux: window_calFFTWindow()
#include "real_fft.h"
#include "stft_algorithm.h"  // Vendored AudioFlux C header (src/third_party/audioflux)

namespace speech::dsp {

namespace {
constexpr const char* kTag = "speech::dsp::STFT";
}

STFT::STFT(int radix2Exp, WindowType windowType, int slideLength)
    : stftObj_(nullptr), fftLength_(0), slideLength_(0) {
    int isContinue = 0;
    int* slideLenPtr = (slideLength > 0) ? &slideLength : nullptr;

    if (stftObj_new(&stftObj_, radix2Exp, &windowType, slideLenPtr, &isContinue) != 0) {
        SPEECH_LOG(ERROR) << TAG(kTag) << "Failed to create STFT object (radix2Exp=" << radix2Exp
                   << "; must be in [1, 30])." << std::endl;
        throw std::invalid_argument("Invalid radix2Exp for STFT (must be in [1, 30]).");
    }

    fftLength_ = 1 << radix2Exp;
    slideLength_ = (slideLength > 0) ? slideLength : fftLength_ / 4;

    if (rfft_impl::RealFrameFFT::supports(fftLength_)) {
        // The real-FFT path must apply the very same analysis window AudioFlux's STFT
        // does. stftObj_new() builds it with window_calFFTWindow(windowType, fftLength),
        // so call that same (unpatched, public) function instead of reaching into the
        // opaque STFT object -- no change to the vendored sources is needed.
        if (float* window = window_calFFTWindow(windowType, fftLength_)) {
            rfft_ = std::make_unique<rfft_impl::RealFrameFFT>(fftLength_, window);  // copies it
            std::free(window);
        }
    }

    SPEECH_LOG(DEBUG) << TAG(kTag) << "Created STFT: fftLength=" << fftLength_
               << ", slideLength=" << slideLength_ << ", windowType="
               << static_cast<int>(windowType) << std::endl;
}

STFT::~STFT() {
    if (stftObj_) {
        stftObj_free(stftObj_);
        stftObj_ = nullptr;
    }
}

int STFT::calTimeLength(int dataLength) const {
    return stftObj_calTimeLength(stftObj_, dataLength);
}

int STFT::calDataLength(int timeLength) const {
    return stftObj_calDataLength(stftObj_, timeLength);
}

namespace {
// Frame loops are embarrassingly parallel; the threshold keeps tiny inputs from
// paying thread start-up cost.
constexpr int kParallelFrameThreshold = 128;
}  // namespace

int STFT::stftInto(const float* data, std::size_t n, float* real, float* imag) {
    const int dataLength = static_cast<int>(n);
    const int timeLength = calTimeLength(dataLength);
    if (timeLength <= 0) {
        SPEECH_LOG(DEBUG) << TAG(kTag) << "stft(): input length " << dataLength
                   << " is shorter than fftLength=" << fftLength_
                   << "; returning zero frames." << std::endl;
        return 0;
    }
    if (!rfft_) {
        // AudioFlux writes frame-major into flat buffers and does not modify the
        // input (the const_cast only satisfies its non-const C signature).
        stftObj_stft(stftObj_, const_cast<float*>(data), dataLength, real, imag);
        return timeLength;
    }

    // Real input: compute the N/2+1 unique bins per frame, then fill the upper half
    // with the conjugate mirror X[N-k] = conj(X[k]).
    const int N = fftLength_;
    const int half = N / 2;
    const int hop = slideLength_;
    const rfft_impl::RealFrameFFT& fft = *rfft_;
#pragma omp parallel for schedule(static) if (timeLength >= kParallelFrameThreshold)
    for (int f = 0; f < timeLength; ++f) {
        float* re = real + static_cast<std::size_t>(f) * N;
        float* im = imag + static_cast<std::size_t>(f) * N;
        fft.spectrum(data + static_cast<std::size_t>(f) * hop, re, im);
        for (int k = 1; k < half; ++k) {
            re[N - k] = re[k];
            im[N - k] = -im[k];
        }
    }
    return timeLength;
}

int STFT::stftOnesidedInto(const float* data, std::size_t n, float* real, float* imag) {
    const int timeLength = calTimeLength(static_cast<int>(n));
    if (timeLength <= 0) {
        return 0;
    }
    const int bins = fftLength_ / 2 + 1;
    if (!rfft_) {
        // Unsupported length: full complex STFT, then keep the first `bins` columns.
        std::vector<float> fullRe(static_cast<std::size_t>(timeLength) * fftLength_);
        std::vector<float> fullIm(fullRe.size());
        stftInto(data, n, fullRe.data(), fullIm.data());
        for (int f = 0; f < timeLength; ++f) {
            std::copy_n(fullRe.begin() + static_cast<std::size_t>(f) * fftLength_, bins,
                        real + static_cast<std::size_t>(f) * bins);
            std::copy_n(fullIm.begin() + static_cast<std::size_t>(f) * fftLength_, bins,
                        imag + static_cast<std::size_t>(f) * bins);
        }
        return timeLength;
    }
    const int hop = slideLength_;
    const rfft_impl::RealFrameFFT& fft = *rfft_;
#pragma omp parallel for schedule(static) if (timeLength >= kParallelFrameThreshold)
    for (int f = 0; f < timeLength; ++f) {
        fft.spectrum(data + static_cast<std::size_t>(f) * hop,
                     real + static_cast<std::size_t>(f) * bins,
                     imag + static_cast<std::size_t>(f) * bins);
    }
    return timeLength;
}

void STFT::powerSpectra(const float* data, std::size_t n, int firstFrame, int numFrames,
                        float* power) {
    const int bins = fftLength_ / 2 + 1;
    const int hop = slideLength_;
    if (numFrames <= 0) {
        return;
    }
    if (rfft_) {
        const rfft_impl::RealFrameFFT& fft = *rfft_;
        for (int f = 0; f < numFrames; ++f) {
            fft.power(data + static_cast<std::size_t>(firstFrame + f) * hop,
                      power + static_cast<std::size_t>(f) * bins);
        }
        return;
    }
    // Fallback for unsupported lengths: complex STFT of just these frames' samples.
    const std::size_t start = static_cast<std::size_t>(firstFrame) * hop;
    const std::size_t len = static_cast<std::size_t>(numFrames - 1) * hop + fftLength_;
    (void)n;
    std::vector<float> re(static_cast<std::size_t>(numFrames) * fftLength_);
    std::vector<float> im(re.size());
    stftInto(data + start, len, re.data(), im.data());
    for (int f = 0; f < numFrames; ++f) {
        const float* r = re.data() + static_cast<std::size_t>(f) * fftLength_;
        const float* i = im.data() + static_cast<std::size_t>(f) * fftLength_;
        float* p = power + static_cast<std::size_t>(f) * bins;
        for (int k = 0; k < bins; ++k) p[k] = r[k] * r[k] + i[k] * i[k];
    }
}

Spectrogram STFT::spectrogram(const float* data, std::size_t n, bool onesided) {
    Spectrogram out;
    out.fftLength = fftLength_;
    out.numBins = numBins(onesided);
    const int timeLength = calTimeLength(static_cast<int>(n));
    if (timeLength <= 0) {
        return out;
    }
    const std::size_t total = static_cast<std::size_t>(timeLength) * out.numBins;
    out.real.resize(total);  // uninitialized: fully overwritten below
    out.imag.resize(total);
    out.numFrames = onesided ? stftOnesidedInto(data, n, out.real.data(), out.imag.data())
                             : stftInto(data, n, out.real.data(), out.imag.data());
    return out;
}

std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> STFT::stft(
    const std::vector<float>& data) {
    // Legacy nested-vector API (kept for source compatibility): one pass of
    // per-frame copies out of the flat spectrogram. Prefer spectrogram().
    Spectrogram sp = spectrogram(data.data(), data.size());
    if (sp.numFrames <= 0) {
        return {};
    }
    std::vector<std::vector<float>> real(sp.numFrames);
    std::vector<std::vector<float>> imag(sp.numFrames);
    for (int i = 0; i < sp.numFrames; ++i) {
        const std::size_t off = static_cast<std::size_t>(i) * fftLength_;
        real[i].assign(sp.real.begin() + off, sp.real.begin() + off + fftLength_);
        imag[i].assign(sp.imag.begin() + off, sp.imag.begin() + off + fftLength_);
    }
    SPEECH_LOG(DEBUG) << TAG(kTag) << "Computed STFT: " << data.size() << " samples -> "
               << sp.numFrames << " frames." << std::endl;
    return {std::move(real), std::move(imag)};
}

void STFT::istftInto(const float* real, const float* imag, int numFrames, bool onesided,
                     int methodType, float* out) {
    if (numFrames <= 0 || real == nullptr || imag == nullptr || out == nullptr) {
        SPEECH_LOG(ERROR) << TAG(kTag) << "istft() requires at least one frame and non-null buffers."
                   << std::endl;
        throw std::invalid_argument("istft() requires non-empty real/imag inputs.");
    }
    if (methodType != 0 && methodType != 1) {
        SPEECH_LOG(ERROR) << TAG(kTag) << "istft(): methodType must be 0 or 1, got " << methodType
                   << "." << std::endl;
        throw std::invalid_argument("istft(): methodType must be 0 (weighted) or 1 (plain).");
    }

    const int N = fftLength_;
    const int half = N / 2;
    const int hop = slideLength_;
    const int bins = numBins(onesided);
    const std::size_t dataLength = static_cast<std::size_t>(calDataLength(numFrames));

    if (!rfft_) {
        // Unsupported length: AudioFlux's complex kernel, fed the full spectrum (mirroring the
        // one-sided input if that is what we got). It accumulates into `out`, so zero it first.
        const float* fullRe = real;
        const float* fullIm = imag;
        std::vector<float> mirrorRe, mirrorIm;
        if (onesided) {
            mirrorRe.resize(static_cast<std::size_t>(numFrames) * N);
            mirrorIm.resize(mirrorRe.size());
            for (int f = 0; f < numFrames; ++f) {
                const float* re = real + static_cast<std::size_t>(f) * bins;
                const float* im = imag + static_cast<std::size_t>(f) * bins;
                float* fr = mirrorRe.data() + static_cast<std::size_t>(f) * N;
                float* fi = mirrorIm.data() + static_cast<std::size_t>(f) * N;
                for (int k = 0; k < bins; ++k) {
                    fr[k] = re[k];
                    fi[k] = (k == 0 || k == half) ? 0.0f : im[k];  // DC/Nyquist have no imaginary part
                }
                for (int k = 1; k < half; ++k) {
                    fr[N - k] = re[k];
                    fi[N - k] = -im[k];
                }
            }
            fullRe = mirrorRe.data();
            fullIm = mirrorIm.data();
        }
        std::fill(out, out + dataLength, 0.0f);
        stftObj_istft(stftObj_, const_cast<float*>(fullRe), const_cast<float*>(fullIm), numFrames,
                      methodType, out);
        return;
    }

    // Real-FFT path. Per method, frame k is multiplied by winA[k] before overlap-add and winB[k]
    // is accumulated into the normalizer (weighted: w and w^2; plain: 1 and w) -- exactly what
    // AudioFlux's stftObj_istft() does.
    const rfft_impl::RealFrameFFT& fft = *rfft_;
    const float* w = fft.window();
    std::vector<float> winA(static_cast<std::size_t>(N)), winB(static_cast<std::size_t>(N));
    for (int k = 0; k < N; ++k) {
        winA[k] = (methodType == 0) ? w[k] : 1.0f;
        winB[k] = (methodType == 0) ? w[k] * w[k] : w[k];
    }

    std::fill(out, out + dataLength, 0.0f);
    std::vector<float> norm(dataLength, 0.0f);

    // Frames are inverse-transformed a block at a time (in parallel: they are independent), then
    // overlap-added serially in frame order, so the sum is accumulated in the same order as the
    // sequential reference regardless of the thread count.
    const int blockFrames = std::max(16, (1 << 17) / N);
    detail::UninitVector<float> block(static_cast<std::size_t>(blockFrames) * N);
    const bool parallel = numFrames >= kParallelFrameThreshold;
    for (int base = 0; base < numFrames; base += blockFrames) {
        const int count = std::min(blockFrames, numFrames - base);
#pragma omp parallel for schedule(static) if (parallel)
        for (int i = 0; i < count; ++i) {
            const std::size_t f = static_cast<std::size_t>(base + i);
            const float* re = real + f * bins;
            const float* im = imag + f * bins;
            float* x = block.data() + static_cast<std::size_t>(i) * N;
            if (onesided) {
                fft.inverse(re, im, x);
            } else {
                // Full spectrum: take the real part of its inverse transform, which equals the
                // inverse transform of its conjugate-symmetric part (X[k] + conj(X[N-k])) / 2.
                float sre[rfft_impl::RealFrameFFT::kMaxLength / 2 + 1];
                float sim[rfft_impl::RealFrameFFT::kMaxLength / 2 + 1];
                sre[0] = re[0];
                sim[0] = 0.0f;
                for (int k = 1; k < half; ++k) {
                    sre[k] = 0.5f * (re[k] + re[N - k]);
                    sim[k] = 0.5f * (im[k] - im[N - k]);
                }
                sre[half] = re[half];
                sim[half] = 0.0f;
                fft.inverse(sre, sim, x);
            }
        }
        for (int i = 0; i < count; ++i) {
            const float* x = block.data() + static_cast<std::size_t>(i) * N;
            const std::size_t start = static_cast<std::size_t>(base + i) * hop;
            float* o = out + start;
            float* nr = norm.data() + start;
            for (int k = 0; k < N; ++k) {
                o[k] += x[k] * winA[k];
                nr[k] += winB[k];
            }
        }
    }

    for (std::size_t j = 0; j < dataLength; ++j) {
        out[j] /= (norm[j] < 1e-6f) ? 1.0f : norm[j];
    }
}

detail::UninitVector<float> STFT::istft(const float* real, const float* imag, int numFrames,
                                         bool onesided, int methodType) {
    detail::UninitVector<float> out;
    if (numFrames > 0) {
        out.resize(static_cast<std::size_t>(calDataLength(numFrames)));  // fully written below
    }
    istftInto(real, imag, numFrames, onesided, methodType, out.data());  // validates the arguments
    SPEECH_LOG(DEBUG) << TAG(kTag) << "Computed ISTFT: " << numFrames << " frames -> " << out.size()
               << " samples." << std::endl;
    return out;
}

detail::UninitVector<float> STFT::istft(const Spectrogram& spec, int methodType) {
    if (spec.numBins != fftLength_ && spec.numBins != fftLength_ / 2 + 1) {
        SPEECH_LOG(ERROR) << TAG(kTag) << "istft(): spectrogram has " << spec.numBins
                   << " bins per frame, expected " << fftLength_ << " or " << fftLength_ / 2 + 1 << "."
                   << std::endl;
        throw std::invalid_argument("istft(): the spectrogram's bin count does not match fftLength().");
    }
    return istft(spec.real.data(), spec.imag.data(), spec.numFrames, spec.numBins != fftLength_, methodType);
}

std::vector<float> STFT::istft(const std::vector<std::vector<float>>& real,
                                const std::vector<std::vector<float>>& imag, int methodType) {
    const int timeLength = static_cast<int>(real.size());
    if (timeLength == 0 || imag.size() != real.size()) {
        SPEECH_LOG(ERROR) << TAG(kTag)
                   << "istft() requires non-empty, equal-length real/imag frame lists."
                   << std::endl;
        throw std::invalid_argument("istft() requires non-empty, equal-length real/imag inputs.");
    }
    for (int i = 0; i < timeLength; ++i) {
        if (static_cast<int>(real[i].size()) != fftLength_ ||
            static_cast<int>(imag[i].size()) != fftLength_) {
            SPEECH_LOG(ERROR) << TAG(kTag) << "istft(): frame " << i << " has the wrong length."
                       << std::endl;
            throw std::invalid_argument("istft(): every frame must have length fftLength().");
        }
    }

    // Flatten into the frame-major layout of the flat API.
    detail::UninitVector<float> flatReal(static_cast<std::size_t>(timeLength) * fftLength_);
    detail::UninitVector<float> flatImag(flatReal.size());
    for (int i = 0; i < timeLength; ++i) {
        std::copy(real[i].begin(), real[i].end(), flatReal.begin() + static_cast<long>(i) * fftLength_);
        std::copy(imag[i].begin(), imag[i].end(), flatImag.begin() + static_cast<long>(i) * fftLength_);
    }
    const detail::UninitVector<float> out = istft(flatReal.data(), flatImag.data(), timeLength, false, methodType);
    return std::vector<float>(out.begin(), out.end());
}

}  // namespace speech::dsp
