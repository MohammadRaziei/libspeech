#include "libspeech/dsp/fft.h"

#include <stdexcept>
#include <string>

#include "aixlog.hpp"
#include "libspeech/detail/log.h"
#include "libspeech/detail/uninit_vector.h"
#include "dsp/fft_algorithm.h"  // Vendored AudioFlux C header (src/third_party/audioflux)
#include "real_fft.h"

namespace speech::dsp {

namespace {
constexpr const char* kTag = "speech::dsp::FFT";

void checkLength(const std::vector<float>& v, int expected, const char* argName) {
    if (!v.empty() && static_cast<int>(v.size()) != expected) {
        SPEECH_LOG(ERROR) << TAG(kTag) << argName << " has length " << v.size() << ", expected "
                   << expected << " (or empty)." << std::endl;
        throw std::invalid_argument(std::string(argName) + " has the wrong length.");
    }
}
}  // namespace

FFT::FFT(int radix2Exp) : fftObj_(nullptr), length_(0) {
    if (fftObj_new(&fftObj_, radix2Exp) != 0) {
        SPEECH_LOG(ERROR) << TAG(kTag) << "Failed to create FFT object (radix2Exp=" << radix2Exp
                   << "; must be in [1, 30])." << std::endl;
        throw std::invalid_argument("Invalid radix2Exp for FFT (must be in [1, 30]).");
    }
    length_ = fftObj_getFFTLength(fftObj_);
    if (rfft_impl::RealFrameFFT::supportsLarge(length_)) {
        const std::vector<float> rectangular(static_cast<std::size_t>(length_), 1.0f);  // copied by the engine
        rfft_ = std::make_unique<rfft_impl::RealFrameFFT>(length_, rectangular.data());
    }
    SPEECH_LOG(DEBUG) << TAG(kTag) << "Created FFT: radix2Exp=" << radix2Exp << ", length=" << length_
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
    forwardInto(real.empty() ? nullptr : real.data(), imag.empty() ? nullptr : imag.data(),
                outReal.data(), outImag.data());
    return {std::move(outReal), std::move(outImag)};
}

std::pair<std::vector<float>, std::vector<float>> FFT::inverse(const std::vector<float>& real,
                                                                 const std::vector<float>& imag) {
    if (real.empty() || imag.empty()) {
        SPEECH_LOG(ERROR) << TAG(kTag) << "inverse() requires non-empty real and imag inputs."
                   << std::endl;
        throw std::invalid_argument("inverse() requires non-empty real and imag inputs.");
    }
    checkLength(real, length_, "real");
    checkLength(imag, length_, "imag");

    std::vector<float> outReal(length_, 0.0f);
    std::vector<float> outImag(length_, 0.0f);
    inverseInto(real.data(), imag.data(), outReal.data(), outImag.data());
    return {std::move(outReal), std::move(outImag)};
}

std::vector<float> FFT::dct(const std::vector<float>& data, bool isNorm) {
    if (static_cast<int>(data.size()) != length_) {
        SPEECH_LOG(ERROR) << TAG(kTag) << "dct() input has length " << data.size() << ", expected "
                   << length_ << "." << std::endl;
        throw std::invalid_argument("dct() input has the wrong length.");
    }

    std::vector<float> output(length_, 0.0f);
    dctInto(data.data(), output.data(), isNorm);
    return output;
}

std::vector<float> FFT::idct(const std::vector<float>& data, bool isNorm) {
    if (static_cast<int>(data.size()) != length_) {
        SPEECH_LOG(ERROR) << TAG(kTag) << "idct() input has length " << data.size() << ", expected "
                   << length_ << "." << std::endl;
        throw std::invalid_argument("idct() input has the wrong length.");
    }

    std::vector<float> output(length_, 0.0f);
    idctInto(data.data(), output.data(), isNorm);
    return output;
}

namespace {
void requireBuffers(const void* a, const void* b, const char* what) {
    if (a == nullptr || b == nullptr) {
        SPEECH_LOG(ERROR) << TAG(kTag) << what << " needs non-null buffers." << std::endl;
        throw std::invalid_argument(std::string(what) + " needs non-null buffers.");
    }
}
}  // namespace

void FFT::forwardInto(const float* real, const float* imag, float* outReal, float* outImag) {
    requireBuffers(outReal, outImag, "forwardInto()");
    if (rfft_ && real != nullptr && imag == nullptr) {
        // Real input: N/2+1 unique bins from one N/2-point complex FFT, then the conjugate mirror
        // X[N-k] = conj(X[k]) for the rest.
        rfft_->spectrum(real, outReal, outImag);
        const int N = length_;
        for (int k = 1; k < N / 2; ++k) {
            outReal[N - k] = outReal[k];
            outImag[N - k] = -outImag[k];
        }
        return;
    }
    // AudioFlux reads its inputs only (the const_cast satisfies its non-const C signature).
    fftObj_fft(fftObj_, const_cast<float*>(real), const_cast<float*>(imag), outReal, outImag);
}

void FFT::inverseInto(const float* real, const float* imag, float* outReal, float* outImag) {
    requireBuffers(real, imag, "inverseInto()");
    requireBuffers(outReal, outImag, "inverseInto()");
    fftObj_ifft(fftObj_, const_cast<float*>(real), const_cast<float*>(imag), outReal, outImag);
}

void FFT::dctInto(const float* data, float* out, bool isNorm) {
    requireBuffers(data, out, "dctInto()");
    fftObj_dct(fftObj_, const_cast<float*>(data), out, isNorm ? 1 : 0);
}

void FFT::idctInto(const float* data, float* out, bool isNorm) {
    requireBuffers(data, out, "idctInto()");
    // AudioFlux's fftObj_idct() mutates dataArr1 (its first argument) in place -- see
    // audioflux_issues.md, Issue 3. It gets a private copy so callers never see their input
    // silently modified.
    detail::UninitVector<float> inputCopy(data, data + length_);
    fftObj_idct(fftObj_, inputCopy.data(), out, isNorm ? 1 : 0);
}

}  // namespace speech::dsp
