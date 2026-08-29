#include "libspeech/dsp/resample.h"

#include <stdexcept>

#include "aixlog.hpp"
#include "dsp/resample_algorithm.h"  // Vendored AudioFlux C header (src/vendor/audioflux)

namespace speech::dsp {

namespace {
constexpr const char* kTag = "speech::dsp::Resample";
}

Resample::Resample(int sourceRate, int targetRate)
    : resampleObj(nullptr), isIdentity(sourceRate == targetRate) {
    if (isIdentity) {
        LOG(DEBUG) << TAG(kTag) << "Source/target rate both " << sourceRate
                   << "Hz; using identity passthrough (skipping AudioFlux)." << std::endl;
        return;
    }

    ResampleQualityType qualType = ResampleQuality_Best;
    // isScale=0: AudioFlux's isScale=1 divides the output by sqrt(ratio), an
    // energy-domain normalization (see resampleObj_resample()). For a
    // downsample from 44100->16000 that means dividing by ~0.60, i.e.
    // amplifying the waveform by ~1.66x -- which can push an already
    // near-full-scale signal well past +/-1.0. We want amplitude-preserving
    // resampling by default for a general audio pipeline (feeding VAD,
    // denoisers, playback, etc.); advanced users who want the energy-scaled
    // behavior can use the extended constructor with isScale=true.
    // Full write-up: /audioflux_issues.md (Issue 2).
    int isScale = 0;
    int isContinue = 0;   // Disable continuous mode by default

    if (resampleObj_new(&resampleObj, &qualType, &isScale, &isContinue) != 0) {
        LOG(ERROR) << TAG(kTag) << "Failed to create resample object (" << sourceRate
                   << "Hz -> " << targetRate << "Hz)." << std::endl;
        throw std::runtime_error("Failed to create resample object.");
    }

    resampleObj_setSamplate(resampleObj, sourceRate, targetRate);
    LOG(DEBUG) << TAG(kTag) << "Created resampler: " << sourceRate << "Hz -> " << targetRate
               << "Hz." << std::endl;
}

Resample::Resample(int sourceRate, int targetRate, int zeroNum, int nbit, WindowType winType,
                     float value, float rollOff, bool isScale, bool isContinue)
    : resampleObj(nullptr), isIdentity(sourceRate == targetRate) {
    if (isIdentity) {
        LOG(DEBUG) << TAG(kTag) << "Source/target rate both " << sourceRate
                   << "Hz; using identity passthrough (skipping AudioFlux)." << std::endl;
        return;
    }

    int isScaleInt = isScale ? 1 : 0;
    int isContinueInt = isContinue ? 1 : 0;

    if (resampleObj_newWithWindow(&resampleObj, &zeroNum, &nbit, &winType, &value, &rollOff,
                                   &isScaleInt, &isContinueInt) != 0) {
        LOG(ERROR) << TAG(kTag) << "Failed to create resample object with custom window settings."
                   << std::endl;
        throw std::runtime_error("Failed to create resample object with custom settings.");
    }

    resampleObj_setSamplate(resampleObj, sourceRate, targetRate);
    LOG(DEBUG) << TAG(kTag) << "Created resampler (custom window): " << sourceRate << "Hz -> "
               << targetRate << "Hz." << std::endl;
}

Resample::~Resample() {
    reset();
}

void Resample::setSampleRateRatio(float ratio) {
    if (isIdentity) {
        LOG(DEBUG) << TAG(kTag)
                   << "setSampleRateRatio() ignored: resampler is in identity passthrough mode."
                   << std::endl;
        return;
    }
    resampleObj_setSamplateRatio(resampleObj, ratio);
    LOG(DEBUG) << TAG(kTag) << "Sample rate ratio set to " << ratio << std::endl;
}

void Resample::enableContinuous(bool flag) {
    if (isIdentity) {
        return;
    }
    resampleObj_enableContinue(resampleObj, flag ? 1 : 0);
}

std::vector<float> Resample::resample(const std::vector<float>& inputData) {
    if (isIdentity) {
        return inputData;
    }

    if (inputData.empty()) {
        LOG(DEBUG) << TAG(kTag) << "resample() called with empty input; returning empty output."
                   << std::endl;
        return {};
    }

    const int inputLength = static_cast<int>(inputData.size());

    const int outputLength = resampleObj_calDataLength(resampleObj, inputLength);
    // TRACE, not DEBUG: this is the exact internal computation that exposed
    // both bugs in audioflux_issues.md (Issue 1 and 2) -- worth keeping at a
    // verbosity level someone can turn on when output length/amplitude looks
    // wrong, without it cluttering normal DEBUG output on every call.
    LOG(TRACE) << TAG(kTag) << "calDataLength(" << inputLength << ") -> expected output length "
               << outputLength << std::endl;
    std::vector<float> outputData(outputLength, 0.0f);

    const int actualOutputLength = resampleObj_resample(
        resampleObj, const_cast<float*>(inputData.data()), inputLength, outputData.data());

    if (actualOutputLength <= 0) {
        LOG(ERROR) << TAG(kTag) << "Resampling produced no output (input length=" << inputLength
                   << ")." << std::endl;
        throw std::runtime_error("Resampling failed or produced no output.");
    }

    outputData.resize(actualOutputLength);
    LOG(DEBUG) << TAG(kTag) << "Resampled " << inputLength << " -> " << actualOutputLength
               << " samples." << std::endl;
    return outputData;
}

void Resample::reset() {
    if (resampleObj) {
        resampleObj_free(resampleObj);
        resampleObj = nullptr;
    }
}

}  // namespace speech::dsp
