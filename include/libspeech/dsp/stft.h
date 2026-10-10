//
// speech::dsp::STFT -- short-time Fourier transform / inverse STFT.
//
// Concrete (non-virtual) class, same rationale as FFT: one backend, no
// polymorphism needed. Composes speech::dsp::window's window types and
// AudioFlux's FFT engine under the hood.
//

#ifndef LIBSPEECH_DSP_STFT_H
#define LIBSPEECH_DSP_STFT_H

#include <utility>
#include <vector>

#include <cstddef>
#include <memory>

#include "flux_base.h"  // WindowType enum
#include "libspeech/detail/uninit_vector.h"
#include "libspeech/export.h"

// Opaque handle to AudioFlux's underlying C STFT object (declared in the
// global namespace because AudioFlux's own `STFTObj` typedef points here).
struct OpaqueSTFT;

namespace speech::dsp {

/**
 * STFT: frames a signal, applies a window, and FFTs each frame (and the
 * reverse: overlap-add reconstruction from a real/imag spectrogram).
 *
 * Each output frame is the *full* fftLength complex spectrum (not folded to
 * the non-redundant half), matching speech::dsp::FFT's convention. The
 * spectrogram is represented as a matrix: outer vector = time frames, inner
 * vector = fftLength frequency-domain samples per frame.
 */
// Flat, frame-major spectrogram: frame i occupies [i*numBins, (i+1)*numBins)
// in both `real` and `imag`. One allocation per plane, no per-frame vectors.
struct Spectrogram {
    int numFrames = 0;
    int fftLength = 0;
    int numBins = 0;  // columns per frame: fftLength (full) or fftLength/2 + 1 (one-sided)
    detail::UninitVector<float> real;
    detail::UninitVector<float> imag;
};

namespace rfft_impl {
class RealFrameFFT;  // private real-input FFT engine (src/dsp/real_fft.h)
}

class SPEECH_API STFT {
   public:
    // radix2Exp: frame length = 2^radix2Exp.
    // windowType: analysis window applied to each frame (default: Hann,
    //   which -- combined with the default slideLength below -- satisfies
    //   the constant-overlap-add condition needed for clean reconstruction).
    // slideLength: hop size between frames, in samples. 0 (default) uses
    //   AudioFlux's own default of fftLength/4 (75% overlap).
    explicit STFT(int radix2Exp, WindowType windowType = Window_Hann, int slideLength = 0);
    ~STFT();

    STFT(const STFT&) = delete;
    STFT& operator=(const STFT&) = delete;

    [[nodiscard]] int fftLength() const { return fftLength_; }
    [[nodiscard]] int slideLength() const { return slideLength_; }

    // Number of frames stft() would produce for a signal of this length
    // (with no padding: signals shorter than fftLength produce 0 frames).
    [[nodiscard]] int calTimeLength(int dataLength) const;

    // Signal length istft() needs to reconstruct, for a given frame count.
    [[nodiscard]] int calDataLength(int timeLength) const;

    // Copy-free core: frames/windows/FFTs `data[0..n)` straight into two flat
    // buffers (each numFrames*fftLength floats, frame-major). The buffers are
    // fully overwritten, so callers may pass uninitialized memory. Returns the
    // frame count (0 if the signal is shorter than fftLength).
    int stftInto(const float* data, std::size_t n, float* real, float* imag);

    // One-sided variant: only bins [0, fftLength/2] of each frame, which fully
    // describe a real signal's spectrum (the rest is its conjugate mirror).
    // Half the FFT work and half the output of stftInto(); each of `real` and
    // `imag` must hold numFrames * (fftLength/2 + 1) floats.
    int stftOnesidedInto(const float* data, std::size_t n, float* real, float* imag);

    // Same, allocating the (uninitialized) output planes. onesided=true gives
    // numBins = fftLength/2 + 1 columns instead of fftLength.
    Spectrogram spectrogram(const float* data, std::size_t n, bool onesided = false);

    // Number of columns per frame: fftLength/2 + 1 if onesided, else fftLength.
    [[nodiscard]] int numBins(bool onesided) const { return onesided ? fftLength_ / 2 + 1 : fftLength_; }

    // Power spectra |X[k]|^2 (one-sided, fftLength/2 + 1 bins per frame) for
    // frames [firstFrame, firstFrame + numFrames) of `data`, written row-major
    // into `power` (numFrames * (fftLength/2 + 1) floats). The caller guarantees
    // those frames exist (firstFrame + numFrames <= calTimeLength(n)).
    void powerSpectra(const float* data, std::size_t n, int firstFrame, int numFrames, float* power);

    // Frames `data`, windows each frame, and FFTs it. Returns {real, imag},
    // each a [timeLength][fftLength] matrix (timeLength = calTimeLength(data.size())).
    std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> stft(
        const std::vector<float>& data);

    // Inverse of stftInto()/stftOnesidedInto(): reconstructs a signal from flat, frame-major
    // real/imag planes by overlap-add of the inverse-transformed frames.
    //   methodType 0 (default): weighted overlap-add, each frame multiplied by the window again and
    //                           the sum divided by the summed squared window (AudioFlux's default);
    //   methodType 1: plain overlap-add, divided by the summed window.
    //   Any other value throws std::invalid_argument.
    // onesided=false: numFrames * fftLength bins per plane (the full spectrum, as stft() returns);
    // onesided=true:  numFrames * (fftLength/2 + 1) bins, treated as a real signal's spectrum.
    // Like AudioFlux's kernel this returns the REAL PART of the inverse transform, so a full
    // spectrum that is not conjugate-symmetric (e.g. one edited bin by bin) is accepted too.
    // `out` must hold calDataLength(numFrames) floats and is fully overwritten.
    // NOTE: reconstruction is only accurate away from the very first/last fftLength samples --
    // the analysis window tapers to (near) zero at the signal boundaries, which no normalization
    // can fully undo. This is a property of windowed STFT in general, not an AudioFlux issue.
    void istftInto(const float* real, const float* imag, int numFrames, bool onesided,
                   int methodType, float* out);

    // Same, allocating the (uninitialized, then fully written) output.
    detail::UninitVector<float> istft(const float* real, const float* imag, int numFrames,
                                       bool onesided, int methodType = 0);

    // Same, from a Spectrogram (as returned by spectrogram()); onesided is inferred from
    // spec.numBins.
    detail::UninitVector<float> istft(const Spectrogram& spec, int methodType = 0);

    // Legacy nested-vector form (full spectrum only, source-compatible): the flat overloads above
    // avoid its per-frame copies. Output length is calDataLength(real.size()).
    std::vector<float> istft(const std::vector<std::vector<float>>& real,
                              const std::vector<std::vector<float>>& imag, int methodType = 0);

   private:
    ::OpaqueSTFT* stftObj_;
    int fftLength_;
    int slideLength_;
    // Real-input FFT engine; null for lengths it does not support (then every
    // path falls back to AudioFlux's complex STFT).
    std::unique_ptr<rfft_impl::RealFrameFFT> rfft_;
};

}  // namespace speech::dsp

#endif  // LIBSPEECH_DSP_STFT_H
