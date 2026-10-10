//
// speech::dsp::FFT -- power-of-2 FFT/IFFT/DCT/IDCT.
//
// A concrete (non-virtual) RAII class, not an interface: there is exactly
// one FFT backend here (AudioFlux's radix-2 implementation), so there is
// nothing to make polymorphic. See checklist.md for the project's rationale
// on when to use interfaces (models with competing backends) vs. plain
// classes (DSP with one implementation).
//

#ifndef LIBSPEECH_DSP_FFT_H
#define LIBSPEECH_DSP_FFT_H

#include <memory>
#include <utility>
#include <vector>

#include "libspeech/export.h"

// Opaque handle to AudioFlux's underlying C FFT object (declared in the
// global namespace because AudioFlux's own `FFTObj` typedef points here).
struct OpaqueFFT;

namespace speech::dsp {

namespace rfft_impl {
class RealFrameFFT;  // private real-input FFT engine (src/dsp/real_fft.h)
}

/**
 * FFT: fixed-size power-of-2 FFT/IFFT, plus DCT-II/IDCT-III sharing the same
 * underlying transform (AudioFlux implements DCT via FFT internally).
 *
 * Size is fixed at construction (`length = 2^radix2Exp`) and every input to
 * fft()/ifft()/dct()/idct() must have exactly that length -- this mirrors
 * AudioFlux's own object model and avoids the cost of re-deriving the
 * transform's internal tables (twiddle factors, etc.) on every call.
 */
class SPEECH_API FFT {
   public:
    // radix2Exp in [1, 30] -> length = 2^radix2Exp (e.g. radix2Exp=10 -> 1024).
    explicit FFT(int radix2Exp);
    ~FFT();

    FFT(const FFT&) = delete;
    FFT& operator=(const FFT&) = delete;

    // Number of samples this FFT operates on (2^radix2Exp).
    [[nodiscard]] int size() const { return length_; }

    // Forward FFT. `imag` may be empty for a real-only input. Both inputs
    // (when non-empty) must have length size(). Returns {real, imag}, each
    // of length size().
    std::pair<std::vector<float>, std::vector<float>> forward(
        const std::vector<float>& real, const std::vector<float>& imag = {});

    // Inverse FFT. `real`/`imag` must have length size(). Returns {real, imag}.
    std::pair<std::vector<float>, std::vector<float>> inverse(const std::vector<float>& real,
                                                                const std::vector<float>& imag);

    // DCT-II of a real signal of length size(). isNorm applies AudioFlux's
    // orthonormal scaling (matches scipy's `norm='ortho'`); false gives the
    // unnormalized DCT-II. Returns a vector of length size().
    std::vector<float> dct(const std::vector<float>& data, bool isNorm = true);

    // Inverse of dct(). Does NOT mutate the caller's `data` (see
    // audioflux_issues.md, Issue 3 -- the underlying AudioFlux call mutates
    // its input in place, so we pass it a copy internally).
    std::vector<float> idct(const std::vector<float>& data, bool isNorm = true);

    // ---- Copy-free forms -------------------------------------------------
    // Raw-buffer versions of the four transforms above: every input holds size() floats, every
    // output buffer holds size() floats and is fully overwritten (so uninitialized memory is
    // fine). Inputs and outputs must not overlap. Like the vector forms, a call is not
    // thread-safe on one FFT object: use one object per thread.

    // `imag` may be nullptr (real-only input). A real-only input of 16..131072 samples runs on
    // the real-input FFT engine (half the work of a complex FFT; the upper half of the spectrum
    // is the exact conjugate mirror of the lower half); anything else goes through AudioFlux.
    void forwardInto(const float* real, const float* imag, float* outReal, float* outImag);

    // Full complex inverse FFT.
    void inverseInto(const float* real, const float* imag, float* outReal, float* outImag);

    void dctInto(const float* data, float* out, bool isNorm = true);

    // Does not modify `data` (a scratch copy goes to AudioFlux, see idct()).
    void idctInto(const float* data, float* out, bool isNorm = true);

   private:
    ::OpaqueFFT* fftObj_;
    int length_;
    // Real-input engine (rectangular window) for real-only forward transforms; null when the
    // length is outside its range, in which case AudioFlux's complex FFT does the work.
    std::unique_ptr<rfft_impl::RealFrameFFT> rfft_;
};

}  // namespace speech::dsp

#endif  // LIBSPEECH_DSP_FFT_H
