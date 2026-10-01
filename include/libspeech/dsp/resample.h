//
// Created by mohammad on 3/19/25.
// Refactored under speech::dsp/ namespace.
//

#ifndef LIBSPEECH_DSP_RESAMPLE_H
#define LIBSPEECH_DSP_RESAMPLE_H

#include <cstddef>
#include <memory>
#include <vector>

#include "flux_base.h"  // Vendored AudioFlux header (src/third_party/audioflux): WindowType enum.
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
class SPEECH_API Resample {
   public:
    // Creates a resampler with AudioFlux's default (best-quality) settings.
    // Amplitude-preserving: does not apply AudioFlux's optional sqrt(ratio)
    // energy-domain scaling (see resample.cpp for why).
    Resample(int sourceRate, int targetRate);

    // Creates a resampler with explicit window/quality settings.
    Resample(int sourceRate, int targetRate, int zeroNum, int nbit, WindowType winType,
              float value, float rollOff, bool isScale, bool isContinue);

    ~Resample();

    Resample(const Resample&) = delete;
    Resample& operator=(const Resample&) = delete;

    // Sets the sample rate ratio manually (overrides the ratio derived from source/target rates).
    void setSampleRateRatio(float ratio);

    // Enables or disables continuous (streaming) resampling across successive calls.
    void enableContinuous(bool flag);

    // Resamples inputData and returns the result. Throws std::runtime_error on failure.
    std::vector<float> resample(const std::vector<float>& inputData);

    // Copy-free variant: reads `n` samples from `data`, returns the output in
    // a buffer that is allocated once, uninitialized, and trimmed in place.
    detail::UninitVector<float> resampleFlat(const float* data, std::size_t n);

   private:
    ::OpaqueResample* resampleObj;

    // Exact rational polyphase filter bank (one contiguous weight row per
    // output phase), built from AudioFlux's own interpolation tables when the
    // rates are integers. Replaces the per-sample strided table gather, which
    // was cache-bound. Null for continuous mode / isScale / custom ratios.
    struct Polyphase;
    std::unique_ptr<Polyphase> poly_;
    void buildPolyphase();
    detail::UninitVector<float> resamplePolyphase(const float* data, std::size_t n);

    // True when sourceRate == targetRate. AudioFlux's resampleObj_setSamplate()
    // silently no-ops (leaving its internal ratio at a hardcoded default of 0.5)
    // when the two rates are equal, which would corrupt data if we called into
    // it anyway. We bypass AudioFlux entirely in that case instead.
    bool isIdentity;

    // Frees and clears the underlying AudioFlux resample object.
    void reset();
};

}  // namespace speech::dsp

#endif  // LIBSPEECH_DSP_RESAMPLE_H
