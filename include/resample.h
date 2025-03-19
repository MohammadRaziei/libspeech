//
// Created by mohammad on 3/19/25.
//

#ifndef LIBSPEECH_RESAMPLE_H
#define LIBSPEECH_RESAMPLE_H

#include <vector>
#include <stdexcept>

#include "flux_base.h" // Include the C header for resample functions


// Forward declaration of ResampleObj
struct OpaqueResample;

class Resample {
   public:
    // Constructor: Initializes the resample object with default parameters.
    Resample(int sourceRate, int targetRate);

    // Constructor: Initializes the resample object with custom quality and window settings.
    Resample(int sourceRate, int targetRate, int zeroNum, int nbit, WindowType winType, float value, float rollOff, bool isScale, bool isContinue);

    // Destructor: Frees the resample object.
    ~Resample();

    // Sets the sample rate ratio manually.
    void setSampleRateRatio(float ratio);

    // Enables or disables continuous resampling.
    void enableContinuous(bool flag);

    // Resamples the input data and returns the resampled data.
    std::vector<float> resample(const std::vector<float>& inputData);

   private:
    OpaqueResample* resampleObj; // Pointer to the underlying C resample object.

    // Resets internal state.
    void reset();
};

#endif  // LIBSPEECH_RESAMPLE_H
