//
// speech::dsp::window -- FFT/STFT analysis window generation.
//
// Flat free-functions, not a class: window generation is a stateless
// "give me N floats" operation with no lifecycle to manage, so a class
// would add ceremony without buying anything (no RAII resource, nothing to
// configure once and reuse). Kept in the `window` sub-namespace instead of
// polluting `speech::dsp` directly.
//

#ifndef LIBSPEECH_DSP_WINDOW_H
#define LIBSPEECH_DSP_WINDOW_H

#include <vector>

#include "libspeech/dsp/window_type.h"
#include "libspeech/export.h"

namespace speech::dsp::window {

/**
 * Generates a periodic analysis window of the given type and length,
 * suitable for use before an FFT/STFT (periodic windows avoid double-
 * counting the first/last sample across frame boundaries).
 *
 * @param type   The window shape (window_type::hann, hamming, blackman, rect, ...).
 * @param length Number of samples in the window. Must be > 0.
 * @throws std::invalid_argument if length <= 0.
 */
SPEECH_API std::vector<float> generate(window_type type, int length);

}  // namespace speech::dsp::window

#endif  // LIBSPEECH_DSP_WINDOW_H
