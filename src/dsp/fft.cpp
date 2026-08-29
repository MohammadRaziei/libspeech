#include "libspeech/dsp/fft.h"

#include <stdexcept>
#include <string>

#include "aixlog.hpp"
#include "dsp/fft_algorithm.h"  // Vendored AudioFlux C header (src/vendor/audioflux)

namespace speech::dsp {

namespace {
constexpr const char* kTag = "speech::dsp::FFT";

void checkLength(const std::vector<float>& v, int expected, const char* argName) {
    if (!v.empty() && static_cast<int>(v.size()) != expected) {
        LOG(ERROR) << TAG(kTag) << argName << " has length " << v.size() << ", expected "
                   << expected << " (or empty)." << std::endl;
        throw std::invalid_argument(std::string(argName) + " has the wrong length.");
    }
}
}  // namespace

FFT::FFT(int radix2Exp) : fftObj_(nullptr), length_(0) {
    if (fftObj_new(&fftObj_, radix2Exp) != 0) {
        LOG(ERROR) << TAG(kTag) << "Failed to create FFT object (radix2Exp=" << radix2Exp
                   << "; must be in [1, 30])." << std::endl;
        throw std::invalid_argument("Invalid radix2Exp for FFT (must be in [1, 30]).");
    }
    length_ = fftObj_getFFTLength(fftObj_);
    LOG(DEBUG) << TAG(kTag) << "Created FFT: radix2Exp=" << radix2Exp << ", length=" << length_
               << std::endl;
}

FFT::~FFT() {
    if (fftObj_) {
        fftObj_free(fftObj_);
        fftObj_ = nullptr;
    }
}

std::pair<std::vector<float>, std::vector<float>> FFT::forward(const std::vector<float>& real,
                                                                 const std::vector<float>& imag) {
    checkLength(real, length_, "real");
    checkLength(imag, length_, "imag");

    std::vector<float> outReal(length_, 0.0f);
    std::vector<float> outImag(length_, 0.0f);

    fftObj_fft(fftObj_, real.empty() ? nullptr : const_cast<float*>(real.data()),
               imag.empty() ? nullptr : const_cast<float*>(imag.data()), outReal.data(),
               outImag.data());

    return {std::move(outReal), std::move(outImag)};
}

std::pair<std::vector<float>, std::vector<float>> FFT::inverse(const std::vector<float>& real,
                                                                 const std::vector<float>& imag) {
    if (real.empty() || imag.empty()) {
        LOG(ERROR) << TAG(kTag) << "inverse() requires non-empty real and imag inputs."
                   << std::endl;
        throw std::invalid_argument("inverse() requires non-empty real and imag inputs.");
    }
    checkLength(real, length_, "real");
    checkLength(imag, length_, "imag");

    std::vector<float> outReal(length_, 0.0f);
    std::vector<float> outImag(length_, 0.0f);

    fftObj_ifft(fftObj_, const_cast<float*>(real.data()), const_cast<float*>(imag.data()),
                outReal.data(), outImag.data());

    return {std::move(outReal), std::move(outImag)};
}

std::vector<float> FFT::dct(const std::vector<float>& data, bool isNorm) {
    if (static_cast<int>(data.size()) != length_) {
        LOG(ERROR) << TAG(kTag) << "dct() input has length " << data.size() << ", expected "
                   << length_ << "." << std::endl;
        throw std::invalid_argument("dct() input has the wrong length.");
    }

    std::vector<float> output(length_, 0.0f);
    fftObj_dct(fftObj_, const_cast<float*>(data.data()), output.data(), isNorm ? 1 : 0);
    return output;
}

std::vector<float> FFT::idct(const std::vector<float>& data, bool isNorm) {
    if (static_cast<int>(data.size()) != length_) {
        LOG(ERROR) << TAG(kTag) << "idct() input has length " << data.size() << ", expected "
                   << length_ << "." << std::endl;
        throw std::invalid_argument("idct() input has the wrong length.");
    }

    // AudioFlux's fftObj_idct() mutates dataArr1 (its first argument) in
    // place -- see audioflux_issues.md, Issue 3. We pass it a private copy
    // so callers of speech::dsp::FFT::idct() never see their input vector
    // silently modified.
    std::vector<float> inputCopy = data;
    std::vector<float> output(length_, 0.0f);
    fftObj_idct(fftObj_, inputCopy.data(), output.data(), isNorm ? 1 : 0);
    return output;
}

}  // namespace speech::dsp
