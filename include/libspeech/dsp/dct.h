//
// speech::dsp::dct_ii -- direct (O(N^2)) DCT-II for arbitrary N.
//
// speech::dsp::FFT::dct() computes DCT-II via an FFT trick, which requires
// a power-of-2 length. MFCC's final cepstral step needs a DCT over the
// number of mel filters (commonly 20-40, rarely a power of 2), so this is a
// separate, simpler direct-sum implementation -- correct for any N, and
// fast enough because N is always small in this use case (tens, not
// thousands, of samples). Forward-only: MFCC feature extraction has no use
// for the inverse, so it isn't implemented here (YAGNI).
//

#ifndef LIBSPEECH_DSP_DCT_H
#define LIBSPEECH_DSP_DCT_H

#include <cstddef>
#include <vector>

#include "libspeech/export.h"

namespace speech::dsp {

/**
 * Direct DCT-II of `input` (length N), matching the same convention as
 * scipy.fft.dct(type=2, norm='ortho' if orthonormal else None) and
 * speech::dsp::FFT::dct()'s is_norm flag.
 *
 * @param input       Input signal, any length N >= 1.
 * @param num_outputs  Number of leading coefficients to return (-1 = all N).
 *                    Must be in [1, N] if given explicitly.
 * @param orthonormal If true, apply orthonormal scaling (energy-preserving,
 *                     matches norm='ortho'); if false, the unnormalized
 *                     DCT-II (matches norm=None).
 */
SPEECH_API std::vector<float> dct_ii(const std::vector<float>& input, int num_outputs = -1,
                          bool orthonormal = true);

/**
 * DctIIPlan: DCT-II with the cosine basis precomputed once, for applying the same
 * size transform to many vectors (e.g. one per MFCC frame). dct_ii() above
 * re-evaluates std::cos for every (output, input) pair on every call, which
 * dominates MFCC time; this class pays that cost once at construction.
 * Results match dct_ii() to float precision.
 */
class SPEECH_API dct_ii_plan {
   public:
    // n: input length; num_outputs in [1, n]; orthonormal: same scaling as dct_ii().
    dct_ii_plan(int n, int num_outputs, bool orthonormal = true);

    [[nodiscard]] int input_size() const { return n_; }
    [[nodiscard]] int output_size() const { return k_; }

    // out[0..num_outputs) = DCT-II of in[0..n). No allocation.
    void apply(const float* in, float* out) const;

   private:
    int n_;
    int k_;
    std::vector<float> basis_;  // [k][n], row-major, scale folded in
};

}  // namespace speech::dsp

#endif  // LIBSPEECH_DSP_DCT_H
