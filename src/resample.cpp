#include "resample.h"
#include <stdexcept>
#include <vector>

#include "dsp/resample_algorithm.h" // Include the C header for resample functions


// Constructor: Initializes the resample object with default parameters.
Resample::Resample(int sourceRate, int targetRate) : resampleObj(nullptr) {
    ResampleQualityType qualType = ResampleQuality_Best;
    int isScale = 1;    // Enable scaling by default
    int isContinue = 0; // Disable continuous mode by default

    if (resampleObj_new(&resampleObj, &qualType, &isScale, &isContinue) != 0) {
        throw std::runtime_error("Failed to create resample object.");
    }

    resampleObj_setSamplate(resampleObj, sourceRate, targetRate);
}

// Constructor: Initializes the resample object with custom quality and window settings.
Resample::Resample(int sourceRate, int targetRate, int zeroNum, int nbit, WindowType winType, float value, float rollOff, bool isScale, bool isContinue)
    : resampleObj(nullptr) {
    int isScaleInt = isScale ? 1 : 0;
    int isContinueInt = isContinue ? 1 : 0;

    if (resampleObj_newWithWindow(&resampleObj, &zeroNum, &nbit, &winType, &value, &rollOff, &isScaleInt, &isContinueInt) != 0) {
        throw std::runtime_error("Failed to create resample object with custom settings.");
    }

    resampleObj_setSamplate(resampleObj, sourceRate, targetRate);
}

// Destructor: Frees the resample object.
Resample::~Resample() {
    if (resampleObj) {
        resampleObj_free(resampleObj);
    }
}

// Sets the sample rate ratio manually.
void Resample::setSampleRateRatio(float ratio) {
    resampleObj_setSamplateRatio(resampleObj, ratio);
}

// Enables or disables continuous resampling.
void Resample::enableContinuous(bool flag) {
    resampleObj_enableContinue(resampleObj, flag ? 1 : 0);
}

// Resamples the input data and returns the resampled data.
std::vector<float> Resample::resample(const std::vector<float>& inputData) {
    // Step 1: Calculate the input length
    int inputLength = static_cast<int>(inputData.size());

    // Step 2: Calculate the output length using resampleObj_calDataLength
    int outputLength = resampleObj_calDataLength(resampleObj, inputLength);

    // Step 3: Allocate the output buffer with the calculated length
    std::vector<float> outputData(outputLength, 0.0f);

    // Step 4: Perform resampling
    int actualOutputLength = resampleObj_resample(
        resampleObj,
        const_cast<float*>(inputData.data()), // Cast away const (ensure safety)
        inputLength,
        outputData.data()
    );

    // Step 5: Check for errors in resampling
    if (actualOutputLength <= 0) {
        throw std::runtime_error("Resampling failed or produced no output.");
    }

    // Step 6: Resize the output vector to match the actual output length
    outputData.resize(actualOutputLength);

    return outputData;
}

// Resets internal state.
void Resample::reset() {
    if (resampleObj) {
        resampleObj_free(resampleObj);
        resampleObj = nullptr;
    }
}