#include "libspeech/dsp/stft.h"

#include <algorithm>
#include <stdexcept>

#include "aixlog.hpp"
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

int STFT::stftInto(const float* data, std::size_t n, float* real, float* imag) {
    const int dataLength = static_cast<int>(n);
    const int timeLength = calTimeLength(dataLength);
    if (timeLength <= 0) {
        LOG(DEBUG) << TAG(kTag) << "stft(): input length " << dataLength
                   << " is shorter than fftLength=" << fftLength_
                   << "; returning zero frames." << std::endl;
        return 0;
    }
    // AudioFlux writes frame-major into flat buffers and does not modify the
    // input (the const_cast only satisfies its non-const C signature).
    stftObj_stft(stftObj_, const_cast<float*>(data), dataLength, real, imag);
    return timeLength;
}

Spectrogram STFT::spectrogram(const float* data, std::size_t n) {
    Spectrogram out;
    out.fftLength = fftLength_;
    const int timeLength = calTimeLength(static_cast<int>(n));
    if (timeLength <= 0) {
        return out;
    }
    const std::size_t total = static_cast<std::size_t>(timeLength) * fftLength_;
    out.real.resize(total);  // uninitialized: fully overwritten below
    out.imag.resize(total);
    out.numFrames = stftInto(data, n, out.real.data(), out.imag.data());
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
