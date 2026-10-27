//
// speech::dsp speech AGC: automatic gain control that follows the level of the speech over time.
//
// The one-gain normalizers (normalize.h) fix the level of a whole recording. A speaker who moves away from the
// microphone, or two speakers at different distances, still come out at different levels inside it. speech_agc keeps
// the *speech* level at a target while the audio streams through it:
//
//   - the level estimate is an exponential average of the mean square of the samples, weighted per window by speech
//     weights (voice-activity probabilities turned into weights by speech_weights()), so it only moves while someone
//     speaks and stays frozen during pauses and noise: the gain does not pump up and amplify the noise floor;
//   - the gain is target / level, limited to [min_gain_db, max_gain_db], and smoothed per sample: it falls with the
//     attack time constant (loud speech: fast) and rises with the release time constant (quiet speech: slow);
//   - there is no look-ahead (no latency): a sample that would still be above `peak_ceiling` after the gain is scaled
//     down to the ceiling on its own (a one-sample hard limit). For a soft limit over a whole buffer use limit_peaks().
//
// Streaming: process() can be called on consecutive chunks of one stream; the state continues from chunk to chunk.
// Cut the stream into chunks that are a multiple of the weight window (the 512 samples of the Silero VAD) so that every
// window falls inside one call; then the result is bit-identical for any such chunking.
//
// Not thread-safe: use one instance per stream.
//

#ifndef LIBSPEECH_DSP_AGC_H
#define LIBSPEECH_DSP_AGC_H

#include <cstddef>

#include "libspeech/export.h"

namespace speech::dsp {

struct speech_agc_params {
    double target_dbfs = -20.0;  // RMS level the speech is brought to (dBFS)
    double level_ms = 1000.0;    // time constant of the speech level estimate, counted in speech time
    double attack_ms = 20.0;     // time constant of a falling gain (speech got louder)
    double release_ms = 400.0;   // time constant of a rising gain (speech got quieter)
    double max_gain_db = 24.0;   // the gain never goes above this ...
    double min_gain_db = -24.0;  // ... or below this
    double peak_ceiling = 0.99;  // linear; no output sample is above it
    double warmup_ms = 250.0;    // speech time needed before the level is trusted; the gain is 1 until then
};

class SPEECH_API speech_agc {
public:
    // Throws std::invalid_argument if sample_rate is not a finite number > 0, a time constant is not a finite number > 0
    // (warmup_ms >= 0), a level is not finite, max_gain_db < min_gain_db, peak_ceiling is not a finite number > 0, or
    // level_ms is shorter than one sample.
    explicit speech_agc(double sample_rate, const speech_agc_params& params = speech_agc_params{});

    // In place. Window k of this call (samples [k * window, (k + 1) * window)) has weight weights[k] in [0, 1]; samples
    // after the last window have weight 0 (the level and the gain target are frozen there). Throws
    // std::invalid_argument, before anything is changed, if window == 0, a weight is outside [0, 1] or NaN, a sample is
    // NaN/inf, or a pointer is null while its size is > 0.
    void process(float* x, std::size_t n, const float* weights, std::size_t n_weights, std::size_t window);

    // Back to the state of a new instance.
    void reset() noexcept;

    double gain() const noexcept { return gain_; }  // current smoothed gain (linear)
    // Speech level estimate before the gain (dBFS); -infinity until any speech was seen.
    double level_dbfs() const noexcept;
    double sample_rate() const noexcept { return sample_rate_; }

private:
    double sample_rate_;
    speech_agc_params params_;
    double target_amplitude_, min_gain_, max_gain_;
    double level_samples_, warmup_samples_, attack_coeff_, release_coeff_;
    double mean_square_ = 0.0, mass_ = 0.0, gain_ = 1.0;
};

}  // namespace speech::dsp

#endif  // LIBSPEECH_DSP_AGC_H
