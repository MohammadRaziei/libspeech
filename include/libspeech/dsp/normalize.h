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
// Speech-weighted level: the RMS above counts every sample, so a recording with long pauses reads quieter than its
// speech is and the gain comes out too high. rms_normalize_weighted() measures the level only where a per-window
// weight says there is speech (weights come from a voice-activity detector's speech probabilities, turned into
// weights by speech_weights(): a hard threshold or a soft one). The gain is still one number for the whole signal
// and the peak ceiling still wins; only what is measured changes.
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

// How a speech probability becomes a weight in [0, 1].
enum class speech_gate {
    hard,  // weight 1 when probability >= threshold, else 0 (the rule the VAD itself uses to open a segment)
    soft,  // weight (probability - threshold) / (1 - threshold) clamped to [0, 1]: 0 up to the threshold, then a
           // linear ramp to 1 at probability 1 (soft thresholding: a window just above the threshold counts little)
};

// Writes n weights to `out` from n speech probabilities `prob` (each must be in [0, 1]). `threshold` must satisfy
// 0 < threshold <= 1 for hard and 0 < threshold < 1 for soft. Throws std::invalid_argument on a bad threshold, a
// probability outside [0, 1] or NaN, or a null pointer while n > 0.
SPEECH_API void speech_weights(const float* prob, std::size_t n, speech_gate mode, double threshold, float* out);

// RMS of x[0..n) where window k (samples [k * window, (k + 1) * window), cut at n) counts with weight weights[k]:
//
//     sqrt( sum_k w_k * energy_k / sum_k w_k * len_k )
//
// Samples after the last window (the VAD drops a trailing partial window) have weight 0. 0 when the total weight is 0.
// Throws std::invalid_argument if window == 0, a weight is outside [0, 1] or NaN, or a pointer is null while its size
// is > 0.
SPEECH_API double weighted_rms(const float* x, std::size_t n, const float* weights, std::size_t n_weights,
                               std::size_t window);

// Like rms_normalize(), but the level that is brought to `target_dbfs` is the weighted RMS above (the gain applies to
// every sample, the peak ceiling is checked on the whole signal). When the total weight is 0 (no speech found) or the
// signal has a NaN/inf sample, x is left unchanged, gain 1, and input_rms_dbfs / output_rms_dbfs are -infinity for
// no weight. Same exceptions as rms_normalize() and weighted_rms().
SPEECH_API rms_normalize_result rms_normalize_weighted(float* x, std::size_t n, const float* weights,
                                                       std::size_t n_weights, std::size_t window,
                                                       double target_dbfs = -20.0, double peak_ceiling = 0.99);

}  // namespace speech::dsp

#endif  // LIBSPEECH_DSP_NORMALIZE_H
