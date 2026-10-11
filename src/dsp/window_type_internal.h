// Internal: window_type <-> the vendored AudioFlux enum, for the calls that cross into AudioFlux.
#ifndef LIBSPEECH_WINDOW_TYPE_INTERNAL_H
#define LIBSPEECH_WINDOW_TYPE_INTERNAL_H

#include "flux_base.h"  // Vendored AudioFlux header (src/third_party/audioflux): ::WindowType
#include "libspeech/dsp/window_type.h"

namespace speech::dsp {

// The two enums list the same windows in the same order; check it instead of trusting it.
#define LS_CHECK_WINDOW(ours, theirs) \
    static_assert(static_cast<int>(window_type::ours) == static_cast<int>(theirs), #ours " does not line up with AudioFlux")
LS_CHECK_WINDOW(rect, Window_Rect);
LS_CHECK_WINDOW(hann, Window_Hann);
LS_CHECK_WINDOW(hamming, Window_Hamm);
LS_CHECK_WINDOW(blackman, Window_Blackman);
LS_CHECK_WINDOW(kaiser, Window_Kaiser);
LS_CHECK_WINDOW(bartlett, Window_Bartlett);
LS_CHECK_WINDOW(triang, Window_Triang);
LS_CHECK_WINDOW(flattop, Window_Flattop);
LS_CHECK_WINDOW(gauss, Window_Gauss);
LS_CHECK_WINDOW(blackman_harris, Window_Blackman_Harris);
LS_CHECK_WINDOW(blackman_nuttall, Window_Blackman_Nuttall);
LS_CHECK_WINDOW(bartlett_hann, Window_Bartlett_Hann);
LS_CHECK_WINDOW(bohman, Window_Bohman);
LS_CHECK_WINDOW(tukey, Window_Tukey);
#undef LS_CHECK_WINDOW

inline ::WindowType to_audioflux(window_type type) noexcept {
    return static_cast<::WindowType>(static_cast<int>(type));
}
inline window_type from_audioflux(::WindowType type) noexcept {
    return static_cast<window_type>(static_cast<int>(type));
}

}  // namespace speech::dsp

#endif  // LIBSPEECH_WINDOW_TYPE_INTERNAL_H
