//
// speech::dsp::delta -- delta ("velocity") and delta-delta ("acceleration") features.
//
// For a frame-major feature matrix c (MFCC, log-mel, ...), the delta at frame t is the least-squares slope of
// each feature column over a (2*half_window + 1)-frame window centred on t:
//
//     d[t] = sum_{n=1..N} n * (c[t+n] - c[t-n]) / (2 * sum_{n=1..N} n^2),      N = half_window
//
// This is the regression formula used by HTK and Kaldi (half_window 2 is their default, a 5-frame window).
// Delta-delta is the same operation applied to the delta; callers just call delta() twice.
//
// Edge handling (the first/last N frames, where the window runs off the matrix):
//   - delta_edge::replicate: the first/last frame is repeated beyond the edge (HTK, Kaldi, torchaudio).
//   - delta_edge::interp:    the slope of the straight line fitted to the first/last (2N+1) frames is used for
//                            every edge frame (what librosa.feature.delta does with mode="interp",
//                            width = 2N+1). Needs at least 2N+1 frames.
// Away from the edges the two modes give identical results.
//

#ifndef LIBSPEECH_DSP_DELTA_H
#define LIBSPEECH_DSP_DELTA_H

#include "libspeech/dsp/mfcc.h"  // mfcc_matrix: the generic frame-major float matrix
#include "libspeech/export.h"

namespace speech::dsp {

enum class delta_edge { replicate, interp };

// Writes the delta of the frame-major matrix `in` (num_frames rows of num_features) into `out`, which must hold
// num_frames * num_features floats and must not overlap `in`.
// Throws std::invalid_argument for half_window < 1, num_features < 1, a negative frame count, or (interp only)
// fewer than 2*half_window + 1 frames.
SPEECH_API void delta_into(const float* in, int num_frames, int num_features, int half_window, delta_edge edge,
                           float* out);

// Allocating form of delta_into(): returns a matrix with the same shape as the input.
SPEECH_API mfcc_matrix delta(const float* in, int num_frames, int num_features, int half_window = 2,
                             delta_edge edge = delta_edge::replicate);

}  // namespace speech::dsp

#endif  // LIBSPEECH_DSP_DELTA_H
