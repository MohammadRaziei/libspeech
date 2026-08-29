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

#include "flux_base.h"  // WindowType enum

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
class STFT {
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

    // Frames `data`, windows each frame, and FFTs it. Returns {real, imag},
    // each a [timeLength][fftLength] matrix (timeLength = calTimeLength(data.size())).
    std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> stft(
        const std::vector<float>& data);

    // Reconstructs a signal from a real/imag spectrogram via weighted
    // overlap-add (methodType=0, AudioFlux's default) or plain overlap-add
    // (methodType=1). Output length is calDataLength(real.size()).
    // NOTE: reconstruction is only accurate away from the very first/last
    // fftLength samples -- the analysis window necessarily tapers to (near)
    // zero at signal boundaries, which no amount of normalization can fully
    // undo. This is a property of windowed STFT in general, not an
    // AudioFlux-specific issue.
    std::vector<float> istft(const std::vector<std::vector<float>>& real,
                              const std::vector<std::vector<float>>& imag, int methodType = 0);

   private:
    ::OpaqueSTFT* stftObj_;
    int fftLength_;
    int slideLength_;
};

}  // namespace speech::dsp

#endif  // LIBSPEECH_DSP_STFT_H
