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
#include <limits>

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
//
// `clip_percentile` (0 < p <= 100, default 100 = off) makes the level robust against a few very loud windows (plosives,
// clicks): the mean square of every window above the weighted p-th percentile of the window mean squares is replaced by
// that percentile, then the sum above is taken. The result is still an RMS (close to the plain one when the speech is
// even), not a percentile of levels. Throws std::invalid_argument if window == 0, clip_percentile is not in (0, 100],
// a weight is outside [0, 1] or NaN, or a pointer is null while its size is > 0.
SPEECH_API double weighted_rms(const float* x, std::size_t n, const float* weights, std::size_t n_weights,
                               std::size_t window, double clip_percentile = 100.0);

// Soft peak limiter, in place: afterwards no |x[i]| is above `ceiling` (> 0). Instead of one gain for the whole signal
// it lowers the gain only around a peak: the needed gain per sample is min(1, ceiling / |x[i]|), its minimum over the
// next `lookahead` samples is taken, and that is averaged over `lookahead` samples, so the gain falls before the peak,
// never jumps, and is exactly 1 more than 2 * lookahead - 1 samples away from any peak. lookahead 0 is treated as 1
// (an instant, hard gain). Returns true if any sample was lowered. Throws std::invalid_argument if ceiling is not a finite
// number > 0, a sample is NaN/inf, or x is null while n > 0.
SPEECH_API bool limit_peaks(float* x, std::size_t n, double ceiling, std::size_t lookahead);

// Like rms_normalize(), but the level that is brought to `target_dbfs` is the weighted RMS above (the gain applies to
// every sample). `clip_percentile` is the one of weighted_rms(). Peaks: with limiter_lookahead == 0 (default) the peak
// ceiling is checked on the whole signal and wins over the target (the gain is lowered for every sample, peak_limited
// is set); with limiter_lookahead > 0 the gain is the one for the target and limit_peaks(x, n, peak_ceiling,
// limiter_lookahead) is applied afterwards, so the speech keeps the target level and only the peaks are lowered
// (peak_limited is set if it lowered any; output_rms_dbfs is the level before the limiter). When the total weight is 0
// (no speech found) or the signal has a NaN/inf sample, x is left unchanged, gain 1, and the levels are -infinity for no
// weight. Same exceptions as rms_normalize() and weighted_rms().
SPEECH_API rms_normalize_result rms_normalize_weighted(float* x, std::size_t n, const float* weights,
                                                       std::size_t n_weights, std::size_t window,
                                                       double target_dbfs = -20.0, double peak_ceiling = 0.99,
                                                       double clip_percentile = 100.0,
                                                       std::size_t limiter_lookahead = 0);

// Weighted cepstral mean (and variance) normalization of a feature matrix, in place. `features` is row-major
// (frames, dim) (the layout of MFCC / log-mel output); frame t counts with weights[t] in [0, 1] (for example the VAD
// weights mapped to frames). Per dimension d, with the weighted mean m_d and variance v_d over the frames:
//
//     features[t][d] = (features[t][d] - m_d) / sqrt(v_d + eps)        (normalize_variance)
//     features[t][d] =  features[t][d] - m_d                            (otherwise)
//
// All frames are transformed (also the ones with weight 0), but the statistics come from the speech frames only. When
// the total weight is 0 the matrix is left unchanged. Throws std::invalid_argument if dim == 0, eps is not a finite
// number >= 0, a weight is outside [0, 1] or NaN, or a pointer is null while its size is > 0.
SPEECH_API void weighted_cmvn(float* features, std::size_t frames, std::size_t dim, const float* weights,
                              bool normalize_variance = true, double eps = 1e-8);

// --- Integrated loudness (LUFS, ITU-R BS.1770-4) --------------------------------------------------------------------
//
// Loudness as defined for broadcast: the signal goes through the K-weighting filter (a high shelf of about +4 dB above
// 1.7 kHz, then a 38 Hz high-pass), the mean square is taken over 400 ms blocks every 100 ms, and the blocks are gated:
// blocks below -70 LUFS are dropped, then blocks more than 10 LU below the mean loudness of the remaining ones. The
// loudness is -0.691 + 10 log10(mean square of the blocks that are left). A full-scale 997 Hz sine reads -3.01 LUFS.
// This silence-aware gating is why the result follows how loud the programme is, not how much silence it contains.
// Mono only (one channel with weight 1); the filter coefficients are computed for any sample rate from the
// analogue prototype (bilinear transform), so they match the standard's 48 kHz values at 48 kHz.

// Integrated loudness of x[0..n) in LUFS. -infinity for an empty signal, or when no block passes the gates (silence).
// A signal shorter than one block (400 ms) is measured as a single block. Throws std::invalid_argument if sample_rate
// is not a finite number >= 8000, a sample is NaN/inf, or x is null while n > 0.
SPEECH_API double integrated_loudness(const float* x, std::size_t n, double sample_rate);

struct loudness_normalize_result {
    double gain = 1.0;                                            // linear factor applied to every sample
    double input_lufs = -std::numeric_limits<double>::infinity();   // integrated loudness before
    double output_lufs = -std::numeric_limits<double>::infinity();  // loudness the gain gives (before a limiter)
    bool peak_limited = false;                                    // the peak ceiling lowered the gain / the limiter acted
};

// Scales x so that its integrated loudness is `target_lufs` (default -23, the EBU R128 target; podcasts use -16).
// Peaks are handled as in rms_normalize_weighted(): with limiter_lookahead == 0 the peak ceiling lowers the gain of the
// whole signal; with limiter_lookahead > 0 the gain is the one for the target and limit_peaks() lowers only the peaks.
// A silent signal (loudness -infinity) or one with a NaN/inf sample is left unchanged (gain 1). Throws
// std::invalid_argument for a bad sample_rate, a non-finite target_lufs, a peak_ceiling that is not a finite number
// > 0, or a null pointer while n > 0.
SPEECH_API loudness_normalize_result loudness_normalize(float* x, std::size_t n, double sample_rate,
                                                        double target_lufs = -23.0, double peak_ceiling = 0.99,
                                                        std::size_t limiter_lookahead = 0);

}  // namespace speech::dsp

#endif  // LIBSPEECH_DSP_NORMALIZE_H
