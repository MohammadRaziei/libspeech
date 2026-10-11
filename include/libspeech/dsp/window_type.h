//
// The analysis-window shapes libspeech can generate.
//

#ifndef LIBSPEECH_DSP_WINDOW_TYPE_H
#define LIBSPEECH_DSP_WINDOW_TYPE_H

namespace speech::dsp {

/**
 * Window shape for STFT, MFCC, resampling and window::generate(). The enumerators mirror the
 * windows AudioFlux implements (the library the DSP kernels build on), but this type is libspeech's
 * own, so that AudioFlux's headers do not leak into the public API.
 */
enum class window_type : int {
    rect = 0,
    hann,
    hamming,
    blackman,
    kaiser,
    bartlett,
    triang,
    flattop,
    gauss,
    blackman_harris,
    blackman_nuttall,
    bartlett_hann,
    bohman,
    tukey,  // tapered cosine
};

}  // namespace speech::dsp

#endif  // LIBSPEECH_DSP_WINDOW_TYPE_H
