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

#include <complex>
#include <memory>
#include <utility>
#include <vector>

#include "libspeech/detail/uninit_vector.h"
#include "libspeech/export.h"
#include "libspeech/span.h"

// Opaque handle to AudioFlux's underlying C FFT object (declared in the
// global namespace because AudioFlux's own `FFTObj` typedef points here).
struct OpaqueFFT;

namespace speech::dsp {

namespace rfft_impl {
class real_frame_fft;  // private real-input FFT engine (src/dsp/real_fft.h)
}

/**
 * FFT: fixed-size power-of-2 FFT/IFFT, plus DCT-II/IDCT-III sharing the same
 * underlying transform (AudioFlux implements DCT via FFT internally).
 *
 * Size is fixed at construction (`length = 2^radix2_exp`) and every input to
 * fft()/ifft()/dct()/idct() must have exactly that length -- this mirrors
 * AudioFlux's own object model and avoids the cost of re-deriving the
 * transform's internal tables (twiddle factors, etc.) on every call.
 */
class SPEECH_API fft {
   public:
    // radix2_exp in [1, 30] -> length = 2^radix2_exp (e.g. radix2_exp=10 -> 1024).
    explicit fft(int radix2_exp);
    ~fft();

    fft(const fft&) = delete;
    fft& operator=(const fft&) = delete;

    // Number of samples this FFT operates on (2^radix2_exp).
    [[nodiscard]] int size() const { return length_; }

    // Forward FFT. `imag` may be empty for a real-only input. Both inputs
    // (when non-empty) must have length size(). Returns {real, imag}, each
    // of length size().
    std::pair<std::vector<float>, std::vector<float>> forward(
        const std::vector<float>& real, const std::vector<float>& imag = {});

    // Inverse FFT. `real`/`imag` must have length size(). Returns {real, imag}.
    std::pair<std::vector<float>, std::vector<float>> inverse(const std::vector<float>& real,
                                                                const std::vector<float>& imag);

    // DCT-II of a real signal of length size(). is_norm applies AudioFlux's
    // orthonormal scaling (matches scipy's `norm='ortho'`); false gives the
    // unnormalized DCT-II. Returns a vector of length size().
    std::vector<float> dct(const std::vector<float>& data, bool is_norm = true);

    // Inverse of dct(). Does NOT mutate the caller's `data` (see
    // audioflux_issues.md, Issue 3 -- the underlying AudioFlux call mutates
    // its input in place, so we pass it a copy internally).
    std::vector<float> idct(const std::vector<float>& data, bool is_norm = true);

    // ---- Copy-free forms -------------------------------------------------
    // Raw-buffer versions of the four transforms above: every input holds size() floats, every
    // output buffer holds size() floats and is fully overwritten (so uninitialized memory is
    // fine). Inputs and outputs must not overlap. Like the vector forms, a call is not
    // thread-safe on one FFT object: use one object per thread.

    // `imag` may be nullptr (real-only input). A real-only input of 16..131072 samples runs on
    // the real-input FFT engine (half the work of a complex FFT; the upper half of the spectrum
    // is the exact conjugate mirror of the lower half); anything else goes through AudioFlux.
    void forward_into(const float* real, const float* imag, float* out_real, float* out_imag);

    // Full complex inverse FFT.
    void inverse_into(const float* real, const float* imag, float* out_real, float* out_imag);

    void dct_into(const float* data, float* out, bool is_norm = true);

    // Does not modify `data` (a scratch copy goes to AudioFlux, see idct()).
    void idct_into(const float* data, float* out, bool is_norm = true);

    // View forms of the four calls above (the pointer forms stay). Every non-empty input and every
    // output must hold exactly size() samples; an empty `real` or `imag` means "absent" (real-only
    // or imag-only input to forward_into). std::vector converts to Span automatically:
    //     fft.forward_into(signal, {}, out_real, out_imag);
    // @throws std::invalid_argument on a size mismatch.
    void forward_into(span<const float> real, span<const float> imag, span<float> out_real, span<float> out_imag);
    void inverse_into(span<const float> real, span<const float> imag, span<float> out_real, span<float> out_imag);
    void dct_into(span<const float> data, span<float> out, bool is_norm = true);
    void idct_into(span<const float> data, span<float> out, bool is_norm = true);

    // Interleaved-complex forms (std::complex<float> / NumPy complex64): the full size() -bin
    // spectrum, e.g. directly into a complex64 array.
    //  * real in -> complex out takes the real-FFT engine and writes the result in place: no
    //    allocation, no intermediate planes.
    //  * complex in -> complex out goes through AudioFlux's split-plane kernel; the planes live in
    //    a scratch buffer owned by this object that is allocated on the first call and reused.
    // `in` and `out` may be the same memory for the complex-in forms.
    void forward_into(span<const float> real, span<std::complex<float>> out);
    void forward_into(span<const std::complex<float>> in, span<std::complex<float>> out);
    void inverse_into(span<const std::complex<float>> in, span<std::complex<float>> out);

   private:
    ::OpaqueFFT* fft_obj_;
    int length_;
    // Real-input engine (rectangular window) for real-only forward transforms; null when the
    // length is outside its range, in which case AudioFlux's complex FFT does the work.
    std::unique_ptr<rfft_impl::real_frame_fft> rfft_;
    // Split-plane work space for the complex-in / complex-out forms (4 * length_ floats).
    detail::uninit_vector<float> scratch_;

    void complex_via_planes(bool inverse, span<const std::complex<float>> in, span<std::complex<float>> out);
};

}  // namespace speech::dsp

#endif  // LIBSPEECH_DSP_FFT_H
