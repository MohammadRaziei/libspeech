#include "libspeech/dsp/stft.h"

#include <algorithm>
#include <stdexcept>

#include "aixlog.hpp"
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
        LOG(ERROR) << TAG(kTag) << "Failed to create STFT object (radix2Exp=" << radix2Exp
                   << "; must be in [1, 30])." << std::endl;
        throw std::invalid_argument("Invalid radix2Exp for STFT (must be in [1, 30]).");
    }

    fftLength_ = 1 << radix2Exp;
    slideLength_ = (slideLength > 0) ? slideLength : fftLength_ / 4;

    if (rfft_impl::RealFrameFFT::supports(fftLength_)) {
        if (const float* window = stftObj_windowData(stftObj_)) {
            rfft_ = std::make_unique<rfft_impl::RealFrameFFT>(fftLength_, window);
        }
    }

    LOG(DEBUG) << TAG(kTag) << "Created STFT: fftLength=" << fftLength_
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
        LOG(DEBUG) << TAG(kTag) << "stft(): input length " << dataLength
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
    LOG(DEBUG) << TAG(kTag) << "Computed STFT: " << data.size() << " samples -> "
               << sp.numFrames << " frames." << std::endl;
    return {std::move(real), std::move(imag)};
}

std::vector<float> STFT::istft(const std::vector<std::vector<float>>& real,
                                const std::vector<std::vector<float>>& imag, int methodType) {
    const int timeLength = static_cast<int>(real.size());
    if (timeLength == 0 || imag.size() != real.size()) {
        LOG(ERROR) << TAG(kTag)
                   << "istft() requires non-empty, equal-length real/imag frame lists."
                   << std::endl;
        throw std::invalid_argument("istft() requires non-empty, equal-length real/imag inputs.");
    }
    for (int i = 0; i < timeLength; ++i) {
        if (static_cast<int>(real[i].size()) != fftLength_ ||
            static_cast<int>(imag[i].size()) != fftLength_) {
            LOG(ERROR) << TAG(kTag) << "istft(): frame " << i << " has the wrong length."
                       << std::endl;
            throw std::invalid_argument("istft(): every frame must have length fftLength().");
        }
    }

    // Flatten into the frame-major layout AudioFlux expects.
    std::vector<float> flatReal(static_cast<size_t>(timeLength) * fftLength_);
    std::vector<float> flatImag(static_cast<size_t>(timeLength) * fftLength_);
    for (int i = 0; i < timeLength; ++i) {
        std::copy(real[i].begin(), real[i].end(), flatReal.begin() + static_cast<long>(i) * fftLength_);
        std::copy(imag[i].begin(), imag[i].end(), flatImag.begin() + static_cast<long>(i) * fftLength_);
    }

    const int dataLength = calDataLength(timeLength);
    // stftObj_istft() *accumulates* into dataArr (dataArr[j] += ...), so the
    // output buffer must start zeroed -- std::vector's value-init already
    // guarantees that, but it's worth calling out since it's easy to get
    // wrong if this code is ever changed to reuse a buffer.
    std::vector<float> output(dataLength, 0.0f);

    stftObj_istft(stftObj_, flatReal.data(), flatImag.data(), timeLength, methodType,
                  output.data());

    LOG(DEBUG) << TAG(kTag) << "Computed ISTFT: " << timeLength << " frames -> " << dataLength
               << " samples." << std::endl;
    return output;
}

}  // namespace speech::dsp
