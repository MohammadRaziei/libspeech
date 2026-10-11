//
// speech::dsp::STFT -- short-time Fourier transform / inverse STFT.
//
// Concrete (non-virtual) class, same rationale as FFT: one backend, no
// polymorphism needed. Composes speech::dsp::window's window types and
// AudioFlux's FFT engine under the hood.
//

#ifndef LIBSPEECH_DSP_STFT_H
#define LIBSPEECH_DSP_STFT_H

#include <complex>
#include <utility>
#include <vector>

#include <cstddef>
#include <memory>

#include "libspeech/dsp/window_type.h"
#include "libspeech/detail/uninit_vector.h"
#include "libspeech/export.h"
#include "libspeech/span.h"

// Opaque handle to AudioFlux's underlying C STFT object (declared in the
// global namespace because AudioFlux's own `STFTObj` typedef points here).
struct OpaqueSTFT;

namespace speech::dsp {

/**
 * STFT: frames a signal, applies a window, and FFTs each frame (and the
 * reverse: overlap-add reconstruction from a real/imag spectrogram).
 *
 * Each output frame is the *full* fft_length complex spectrum (not folded to
 * the non-redundant half), matching speech::dsp::FFT's convention. The
 * spectrogram is represented as a matrix: outer vector = time frames, inner
 * vector = fft_length frequency-domain samples per frame.
 */
// Flat, frame-major spectrogram: frame i occupies [i*num_bins, (i+1)*num_bins)
// in both `real` and `imag`. One allocation per plane, no per-frame vectors.
struct complex_spectrogram {
    int num_frames = 0;
    int fft_length = 0;
    int num_bins = 0;  // columns per frame: fft_length (full) or fft_length/2 + 1 (one-sided)
    detail::uninit_vector<float> real;
    detail::uninit_vector<float> imag;

    // The planes as num_frames rows of num_bins samples: iterate them frame by frame
    //   for (speech::span<const float> frame : spec.real_frames()) { ... }
    [[nodiscard]] frame_view<float> real_frames() {
        return frame_view<float>(real.data(), static_cast<std::size_t>(num_frames), static_cast<std::size_t>(num_bins));
    }
    [[nodiscard]] frame_view<const float> real_frames() const {
        return frame_view<const float>(real.data(), static_cast<std::size_t>(num_frames), static_cast<std::size_t>(num_bins));
    }
    [[nodiscard]] frame_view<float> imag_frames() {
        return frame_view<float>(imag.data(), static_cast<std::size_t>(num_frames), static_cast<std::size_t>(num_bins));
    }
    [[nodiscard]] frame_view<const float> imag_frames() const {
        return frame_view<const float>(imag.data(), static_cast<std::size_t>(num_frames), static_cast<std::size_t>(num_bins));
    }

    // Both planes at once, frame by frame and bin by bin, as complex numbers (no allocation):
    //   for (auto frame : spec.frames()) for (std::complex<float> bin : frame) { ... }
    [[nodiscard]] split_complex_frames<float> frames() { return split_complex_frames<float>(real_frames(), imag_frames()); }
    [[nodiscard]] split_complex_frames<const float> frames() const {
        return split_complex_frames<const float>(real_frames(), imag_frames());
    }
};

namespace rfft_impl {
class real_frame_fft;  // private real-input FFT engine (src/dsp/real_fft.h)
}

class SPEECH_API stft {
   public:
    // radix2_exp: frame length = 2^radix2_exp.
    // type: analysis window applied to each frame (default: Hann,
    //   which -- combined with the default slide_length below -- satisfies
    //   the constant-overlap-add condition needed for clean reconstruction).
    // slide_length: hop size between frames, in samples. 0 (default) uses
    //   AudioFlux's own default of fft_length/4 (75% overlap).
    explicit stft(int radix2_exp, window_type type = window_type::hann, int slide_length = 0);
    ~stft();

    stft(const stft&) = delete;
    stft& operator=(const stft&) = delete;

    [[nodiscard]] int fft_length() const { return fft_length_; }
    [[nodiscard]] int slide_length() const { return slide_length_; }

    // Number of frames stft() would produce for a signal of this length
    // (with no padding: signals shorter than fft_length produce 0 frames).
    [[nodiscard]] int cal_time_length(int data_length) const;

    // Signal length istft() needs to reconstruct, for a given frame count.
    [[nodiscard]] int cal_data_length(int time_length) const;

    // Copy-free core: frames/windows/FFTs `data[0..n)` straight into two flat
    // buffers (each num_frames*fft_length floats, frame-major). The buffers are
    // fully overwritten, so callers may pass uninitialized memory. Returns the
    // frame count (0 if the signal is shorter than fft_length).
    int stft_into(const float* data, std::size_t n, float* real, float* imag);

    // One-sided variant: only bins [0, fft_length/2] of each frame, which fully
    // describe a real signal's spectrum (the rest is its conjugate mirror).
    // Half the FFT work and half the output of stft_into(); each of `real` and
    // `imag` must hold num_frames * (fft_length/2 + 1) floats.
    int stft_onesided_into(const float* data, std::size_t n, float* real, float* imag);

    // View forms of the two calls above (the pointer forms stay). `real` and `imag` must be
    // contiguous rows of fft_length() (full) or fft_length()/2 + 1 (one-sided) samples, with at least
    // cal_time_length(data.size()) frames; the first frames are written. Returns the frame count.
    // @throws std::invalid_argument if the shapes do not fit.
    int stft_into(span<const float> data, frame_view<float> real, frame_view<float> imag);
    int stft_onesided_into(span<const float> data, frame_view<float> real, frame_view<float> imag);

    // Interleaved-complex output, e.g. straight into a NumPy complex64 array or a
    // std::complex<float> buffer: `out` holds contiguous rows of fft_length() (full) or
    // fft_length()/2 + 1 (one-sided) std::complex<float>, at least cal_time_length(data.size())
    // frames; the first frames are written, no intermediate buffers. Returns the frame count.
    // @throws std::invalid_argument if the view does not fit.
    int stft_into(span<const float> data, frame_view<std::complex<float>> out);
    int stft_onesided_into(span<const float> data, frame_view<std::complex<float>> out);

    // The signal cut into the frames this STFT analyses (no copy): frame i is data[i*hop, i*hop + N).
    [[nodiscard]] frame_view<const float> frames(span<const float> data) const {
        return frame_view<const float>::sliding(data, static_cast<std::size_t>(fft_length_),
                                                static_cast<std::size_t>(slide_length_));
    }

    // Same, allocating the (uninitialized) output planes. onesided=true gives
    // num_bins = fft_length/2 + 1 columns instead of fft_length.
    complex_spectrogram spectrogram(const float* data, std::size_t n, bool onesided = false);

    // Number of columns per frame: fft_length/2 + 1 if onesided, else fft_length.
    [[nodiscard]] int num_bins(bool onesided) const { return onesided ? fft_length_ / 2 + 1 : fft_length_; }

    // Power spectra |X[k]|^2 (one-sided, fft_length/2 + 1 bins per frame) for
    // frames [first_frame, first_frame + num_frames) of `data`, written row-major
    // into `power` (num_frames * (fft_length/2 + 1) floats). The caller guarantees
    // those frames exist (first_frame + num_frames <= cal_time_length(n)).
    void power_spectra(const float* data, std::size_t n, int first_frame, int num_frames, float* power);

    // Frames `data`, windows each frame, and FFTs it. Returns {real, imag},
    // each a [time_length][fft_length] matrix (time_length = cal_time_length(data.size())).
    std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> compute(
        const std::vector<float>& data);

    // Inverse of stft_into()/stft_onesided_into(): reconstructs a signal from flat, frame-major
    // real/imag planes by overlap-add of the inverse-transformed frames.
    //   method_type 0 (default): weighted overlap-add, each frame multiplied by the window again and
    //                           the sum divided by the summed squared window (AudioFlux's default);
    //   method_type 1: plain overlap-add, divided by the summed window.
    //   Any other value throws std::invalid_argument.
    // onesided=false: num_frames * fft_length bins per plane (the full spectrum, as stft() returns);
    // onesided=true:  num_frames * (fft_length/2 + 1) bins, treated as a real signal's spectrum.
    // Like AudioFlux's kernel this returns the REAL PART of the inverse transform, so a full
    // spectrum that is not conjugate-symmetric (e.g. one edited bin by bin) is accepted too.
    // `out` must hold cal_data_length(num_frames) floats and is fully overwritten.
    // NOTE: reconstruction is only accurate away from the very first/last fft_length samples --
    // the analysis window tapers to (near) zero at the signal boundaries, which no normalization
    // can fully undo. This is a property of windowed STFT in general, not an AudioFlux issue.
    void istft_into(const float* real, const float* imag, int num_frames, bool onesided,
                   int method_type, float* out);

    // View form of istft_into(): `real`/`imag` are num_frames rows of fft_length() (full) or
    // fft_length()/2 + 1 (one-sided) samples -- which one is inferred from the row length -- and
    // `out` must hold at least cal_data_length(num_frames) samples (the first ones are written).
    // @throws std::invalid_argument if the shapes do not fit.
    void istft_into(frame_view<const float> real, frame_view<const float> imag, int method_type,
                   span<float> out);

    // The same from interleaved complex rows (e.g. a NumPy complex64 array): full or one-sided
    // inferred from the row length, no deinterleaving pass. `out` needs cal_data_length(frames).
    void istft_into(frame_view<const std::complex<float>> spec, int method_type, span<float> out);
    detail::uninit_vector<float> istft(frame_view<const std::complex<float>> spec, int method_type = 0);

    // Same, allocating the (uninitialized, then fully written) output.
    detail::uninit_vector<float> istft(const float* real, const float* imag, int num_frames,
                                       bool onesided, int method_type = 0);
    detail::uninit_vector<float> istft(frame_view<const float> real, frame_view<const float> imag,
                                       int method_type = 0);

    // Same, from a ComplexSpectrogram (as returned by spectrogram()); onesided is inferred from
    // spec.num_bins.
    detail::uninit_vector<float> istft(const complex_spectrogram& spec, int method_type = 0);

    // Legacy nested-vector form (full spectrum only, source-compatible): the flat overloads above
    // avoid its per-frame copies. Output length is cal_data_length(real.size()).
    std::vector<float> istft(const std::vector<std::vector<float>>& real,
                              const std::vector<std::vector<float>>& imag, int method_type = 0);

   private:
    int stft_complex_impl(span<const float> data, frame_view<std::complex<float>> out, bool onesided);
    ::OpaqueSTFT* stft_obj_;
    int fft_length_;
    int slide_length_;
    // Real-input FFT engine; null for lengths it does not support (then every
    // path falls back to AudioFlux's complex STFT).
    std::unique_ptr<rfft_impl::real_frame_fft> rfft_;
};

}  // namespace speech::dsp

#endif  // LIBSPEECH_DSP_STFT_H
