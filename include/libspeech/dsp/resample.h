//
// Created by mohammad on 3/19/25.
// Refactored under speech::dsp/ namespace.
//

#ifndef LIBSPEECH_DSP_RESAMPLE_H
#define LIBSPEECH_DSP_RESAMPLE_H

#include <cstddef>
#include <memory>
#include <vector>

#include "libspeech/dsp/window_type.h"
#include "libspeech/detail/uninit_vector.h"
#include "libspeech/export.h"

// Opaque handle to AudioFlux's underlying C resample object (declared in the
// global namespace because AudioFlux's own `ResampleObj` typedef points here).
struct OpaqueResample;

namespace speech::dsp {

/**
 * Resample: sample-rate conversion (e.g. 44100Hz -> 16000Hz) backed by
 * AudioFlux's polyphase/bandlimited resampler.
 */
namespace resample_impl {
struct polyphase;  // polyphase filter bank (defined in resample.cpp)
}

class SPEECH_API resample {
   public:
    // Creates a resampler with AudioFlux's default (best-quality) settings.
    // Amplitude-preserving: does not apply AudioFlux's optional sqrt(ratio)
    // energy-domain scaling (see resample.cpp for why).
    resample(int source_rate, int target_rate);

    // Creates a resampler with explicit window/quality settings.
    resample(int source_rate, int target_rate, int zero_num, int nbit, window_type win_type,
              float value, float roll_off, bool is_scale, bool is_continue);

    ~resample();

    resample(const resample&) = delete;
    resample& operator=(const resample&) = delete;

    // Sets the sample rate ratio manually (overrides the ratio derived from source/target rates).
    void set_sample_rate_ratio(float ratio);

    // Enables or disables continuous (streaming) resampling across successive calls.
    void enable_continuous(bool flag);

    // Resamples input_data and returns the result. Throws std::runtime_error on failure.
    std::vector<float> process(const std::vector<float>& input_data);

    // Copy-free variant: reads `n` samples from `data`, returns the output in
    // a buffer that is allocated once, uninitialized, and trimmed in place.
    detail::uninit_vector<float> resample_flat(const float* data, std::size_t n);

   private:
    // AudioFlux's resample object. Created on demand (ensure_audio_flux()): the default
    // Kaiser "Best/Mid/Fast" resamplers run entirely on the polyphase filter bank below
    // and never need it; it backs continuous (streaming) mode, is_scale, custom ratios and
    // non-Kaiser windows.
    ::OpaqueResample* resample_obj;

    // Rational polyphase filter bank: one contiguous weight row per output phase.
    //  * Kaiser windows: designed directly in double precision from the closed-form
    //    windowed sinc (no lookup table, no interpolation) -- see build_polyphase_exact().
    //  * Other windows: derived from AudioFlux's own interpolation tables.
    // Null for continuous mode / is_scale / arbitrary float ratios, which use AudioFlux's
    // direct kernel instead.
    // Shared and immutable: designed banks are cached process-wide (see resample.cpp), so
    // building many resamplers for the same rates -- a loop over files, one per channel --
    // designs the filter once.
    std::shared_ptr<const resample_impl::polyphase> poly_;
    bool build_polyphase_exact();
    void build_polyphase_from_tables();
    detail::uninit_vector<float> resample_polyphase(const float* data, std::size_t n);

    // Constructor arguments, normalized the way AudioFlux normalizes them, kept so the
    // AudioFlux object can be created lazily and the exact design can use them.
    struct params {
        int source_rate = 0;
        int target_rate = 0;
        int zero_num = 64;
        int nbit = 9;
        window_type win_type = window_type::kaiser;
        float value = 14.7696565f;  // Kaiser beta
        float roll_off = 0.9475937f;
        bool is_scale = false;
        bool is_continue = false;
        bool custom_window = false;  // false: AudioFlux "Best" preset
    } params_;
    void ensure_audio_flux();

    // True when source_rate == target_rate. AudioFlux's resampleObj_setSamplate()
    // silently no-ops (leaving its internal ratio at a hardcoded default of 0.5)
    // when the two rates are equal, which would corrupt data if we called into
    // it anyway. We bypass AudioFlux entirely in that case instead.
    bool is_identity;

    // Frees and clears the underlying AudioFlux resample object.
    void reset();
};

}  // namespace speech::dsp

#endif  // LIBSPEECH_DSP_RESAMPLE_H
