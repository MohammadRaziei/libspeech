#include "libspeech/dsp/stft.h"

#include <algorithm>
#include <stdexcept>

#include "aixlog.hpp"
#include "stft_algorithm.h"  // Vendored AudioFlux C header (src/vendor/audioflux)

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

std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> STFT::stft(
    const std::vector<float>& data) {
    const int dataLength = static_cast<int>(data.size());
    const int timeLength = calTimeLength(dataLength);

    if (timeLength <= 0) {
        LOG(DEBUG) << TAG(kTag) << "stft(): input length " << dataLength
                   << " is shorter than fftLength=" << fftLength_
                   << "; returning zero frames." << std::endl;
        return {};
    }

    // AudioFlux writes into one flat buffer per matrix, frame-major
    // (frame i occupies [i*fftLength, (i+1)*fftLength)).
    std::vector<float> flatReal(static_cast<size_t>(timeLength) * fftLength_, 0.0f);
    std::vector<float> flatImag(static_cast<size_t>(timeLength) * fftLength_, 0.0f);

    stftObj_stft(stftObj_, const_cast<float*>(data.data()), dataLength, flatReal.data(),
                 flatImag.data());

    std::vector<std::vector<float>> real(timeLength);
    std::vector<std::vector<float>> imag(timeLength);
    for (int i = 0; i < timeLength; ++i) {
        real[i].assign(flatReal.begin() + static_cast<long>(i) * fftLength_,
                        flatReal.begin() + static_cast<long>(i + 1) * fftLength_);
        imag[i].assign(flatImag.begin() + static_cast<long>(i) * fftLength_,
                        flatImag.begin() + static_cast<long>(i + 1) * fftLength_);
    }

    LOG(DEBUG) << TAG(kTag) << "Computed STFT: " << dataLength << " samples -> " << timeLength
               << " frames." << std::endl;
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
