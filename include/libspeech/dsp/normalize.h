//
// speech::dsp RMS normalization: scale a signal so that its RMS level hits a target, without clipping.
//
// The gain is one number for the whole signal (the signal's shape is not changed, only its level):
//
//     gain = target_rms / rms(signal)
//
// and if that gain would push the largest sample above `peak_ceiling`, the gain is lowered to
// peak_ceiling / peak instead (the peak ceiling wins over the RMS target, so the result never clips; its RMS is then
// lower than the target and `peak_limited` is set). A signal that is silent (all zeros) or contains a NaN/inf is
// left unchanged: there is no level to scale.
//
// Levels are in dBFS: 0 dBFS is a full-scale sine's peak (amplitude 1.0), so a full-scale sine has an RMS of
// -3.01 dBFS and speech is usually normalized to about -20 dBFS (the default). `peak_ceiling` is a linear amplitude
// (0.99 by default), the same ceiling the SpeechBrain denoiser's level matching uses.
//
// Multi-channel audio: pass the whole planar block (channel after channel) in one call to scale all channels by
// the same gain, which keeps the balance between them.
//

#ifndef LIBSPEECH_DSP_NORMALIZE_H
#define LIBSPEECH_DSP_NORMALIZE_H

#include <cstddef>

#include "libspeech/export.h"

namespace speech::dsp {

// What rms_normalize() / scale_to_rms() did.
struct rms_normalize_result {
    double gain = 1.0;                // factor every sample was multiplied by (1.0 when the signal was left alone)
    double input_rms_dbfs = 0.0;      // RMS level before, in dBFS (-infinity for a silent signal)
    double output_rms_dbfs = 0.0;     // RMS level after, in dBFS (input_rms_dbfs + 20 * log10(gain))
    bool peak_limited = false;        // the peak ceiling, not the RMS target, decided the gain
};

// Root mean square of x[0..n) (accumulated in double). 0 for n == 0.
SPEECH_API double rms(const float* x, std::size_t n);

// RMS level of x[0..n) in dBFS: 20 * log10(rms). -infinity for an empty or silent signal.
SPEECH_API double rms_dbfs(const float* x, std::size_t n);

// Scales x[0..n) in place to a linear RMS of `target_rms`, then lowers the gain if the peak would exceed
// `peak_ceiling` (see the header comment). Throws std::invalid_argument if target_rms is not a finite number > 0,
// peak_ceiling is not a finite number > 0, or x is null while n > 0.
SPEECH_API rms_normalize_result scale_to_rms(float* x, std::size_t n, double target_rms, double peak_ceiling = 0.99);

// The same with the target given in dBFS (default -20 dBFS). Throws std::invalid_argument if target_dbfs is not
// finite, peak_ceiling is not a finite number > 0, or x is null while n > 0.
SPEECH_API rms_normalize_result rms_normalize(float* x, std::size_t n, double target_dbfs = -20.0,
                                              double peak_ceiling = 0.99);

}  // namespace speech::dsp

#endif  // LIBSPEECH_DSP_NORMALIZE_H
