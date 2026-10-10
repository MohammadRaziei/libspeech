// RealFrameFFT: windowed real-input FFT of one frame, producing the one-sided
// spectrum (N/2 + 1 bins). Private to speech::dsp -- STFT and MFCC use it.
//
// Why it exists: AudioFlux's STFT runs a full N-point COMPLEX FFT on real data
// and writes all N bins (real + imaginary), i.e. twice the FFT work and twice
// the output of what a real signal needs. Here:
//   1. the frame is windowed and loaded, in bit-reversed order, as N/2 complex
//      samples z[n] = x[2n] + i*x[2n+1]  (window multiply and permutation fused
//      into the one pass that has to read the frame anyway),
//   2. one N/2-point complex FFT runs in place on split re/im arrays,
//   3. a split pass recovers the N/2+1 bins of the real signal's spectrum.
// All twiddles are computed in double precision and rounded once to float.
#ifndef LIBSPEECH_SRC_DSP_REAL_FFT_H
#define LIBSPEECH_SRC_DSP_REAL_FFT_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace speech::dsp::rfft_impl {

class RealFrameFFT {
   public:
    // Frame lengths used by STFT/MFCC: powers of two in [kMinLength, kMaxLength]. Their work
    // arrays live on the stack (and so does the one in power()), which is what keeps the
    // per-frame hot path allocation-free and safe on threads with small stacks.
    static constexpr int kMinLength = 16;
    static constexpr int kMaxLength = 8192;
    static bool supports(int fftLength) {
        return fftLength >= kMinLength && fftLength <= kMaxLength &&
               (fftLength & (fftLength - 1)) == 0;
    }

    // Wider range for callers that transform one big frame at a time (speech::dsp::FFT): above
    // kMaxLength the work arrays come from the heap, per call. power() is not available there.
    static constexpr int kMaxLargeLength = 1 << 17;  // the bit-reversal table holds 16-bit indices
    static bool supportsLarge(int fftLength) {
        return fftLength >= kMinLength && fftLength <= kMaxLargeLength &&
               (fftLength & (fftLength - 1)) == 0;
    }

    // `window` holds fftLength weights (copied).
    RealFrameFFT(int fftLength, const float* window);

    [[nodiscard]] int length() const { return n_; }
    [[nodiscard]] int numBins() const { return n_ / 2 + 1; }

    // frame: fftLength samples. re/im: numBins() floats each, fully overwritten.
    void spectrum(const float* frame, float* re, float* im) const;

    // |X[k]|^2 for k in [0, numBins()). power: numBins() floats, fully overwritten.
    void power(const float* frame, float* power) const;

    // Inverse of spectrum() WITHOUT the window: rebuilds the real frame x[0, fftLength) whose
    // one-sided spectrum is (re, im) -- numBins() floats each -- including the 1/fftLength
    // scaling, so inverse(spectrum(x)) returns x when the window is all ones. The imaginary parts of the
    // DC and Nyquist bins belong to no real signal and are ignored. `frame` is fully
    // overwritten and must not alias re/im. Same structure as the forward kernel: one
    // fftLength/2-point complex FFT instead of an fftLength-point one.
    void inverse(const float* re, const float* im, float* frame) const;

    // The fftLength analysis-window weights this engine was built with.
    [[nodiscard]] const float* window() const { return window_.data(); }

   private:
    int n_;
    int m_;  // n_/2: size of the complex FFT

    std::vector<float> window_;
    std::vector<std::uint16_t> rev_;  // bit reversal of [0, m_)
    // Stage twiddles for butterfly lengths 8, 16, ..., m_: stage s has len/2
    // entries starting at stageOffset_[s].
    std::vector<float> twr_, twi_;
    std::vector<int> stageOffset_;
    std::vector<float> postR_, postI_;  // exp(-2*pi*i*k/n_), k in [0, m_/2]

    friend struct KernelAccess;
};

}  // namespace speech::dsp::rfft_impl

#endif  // LIBSPEECH_SRC_DSP_REAL_FFT_H
