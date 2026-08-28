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

#include <utility>
#include <vector>

// Opaque handle to AudioFlux's underlying C FFT object (declared in the
// global namespace because AudioFlux's own `FFTObj` typedef points here).
struct OpaqueFFT;

namespace speech::dsp {

/**
 * FFT: fixed-size power-of-2 FFT/IFFT, plus DCT-II/IDCT-III sharing the same
 * underlying transform (AudioFlux implements DCT via FFT internally).
 *
 * Size is fixed at construction (`length = 2^radix2Exp`) and every input to
 * fft()/ifft()/dct()/idct() must have exactly that length -- this mirrors
 * AudioFlux's own object model and avoids the cost of re-deriving the
 * transform's internal tables (twiddle factors, etc.) on every call.
 */
class FFT {
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

   private:
    ::OpaqueFFT* fftObj_;
    int length_;
};

}  // namespace speech::dsp

#endif  // LIBSPEECH_DSP_FFT_H
