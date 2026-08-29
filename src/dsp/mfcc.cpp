#include "libspeech/dsp/mfcc.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "aixlog.hpp"
#include "libspeech/dsp/dct.h"
#include "libspeech/dsp/stft.h"

namespace speech::dsp {

namespace {
constexpr const char* kTag = "speech::dsp::MFCC";

// HTK-style conversion, matching AudioFlux's auditory_freToMel/auditory_melToFre.
float hzToMel(float hz) { return 2595.0f * std::log10(1.0f + hz / 700.0f); }
float melToHz(float mel) { return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f); }

}  // namespace

MFCC::MFCC(Params params) : params_(params) {
    if (params_.numMelFilters < 2) {
        LOG(ERROR) << TAG(kTag) << "numMelFilters must be >= 2 (got " << params_.numMelFilters
                   << ")." << std::endl;
        throw std::invalid_argument("numMelFilters must be >= 2.");
    }
    if (params_.numCoefficients < 1 || params_.numCoefficients > params_.numMelFilters) {
        LOG(ERROR) << TAG(kTag) << "numCoefficients must be in [1, numMelFilters] (got "
                   << params_.numCoefficients << " with numMelFilters=" << params_.numMelFilters
                   << ")." << std::endl;
        throw std::invalid_argument("numCoefficients must be in [1, numMelFilters].");
    }

    stft_ = std::make_unique<STFT>(params_.radix2Exp, Window_Hann, params_.slideLength);

    const int fftLength = stft_->fftLength();
    const float highFreq =
        (params_.highFreqHz > 0.0f) ? params_.highFreqHz : params_.sampleRate / 2.0f;

    melFilterBank_ = buildMelFilterBank(params_.numMelFilters, fftLength, params_.sampleRate,
                                         params_.lowFreqHz, highFreq);

    LOG(DEBUG) << TAG(kTag) << "Created MFCC: sampleRate=" << params_.sampleRate
               << ", numMelFilters=" << params_.numMelFilters
               << ", numCoefficients=" << params_.numCoefficients
               << ", fftLength=" << fftLength << std::endl;
}

MFCC::~MFCC() = default;

std::vector<std::vector<float>> MFCC::buildMelFilterBank(int numMelFilters, int fftLength,
                                                           int sampleRate, float lowFreqHz,
                                                           float highFreqHz) {
    const int numFftBins = fftLength / 2 + 1;

    const float melLow = hzToMel(lowFreqHz);
    const float melHigh = hzToMel(highFreqHz);

    // numMelFilters+2 boundary points define numMelFilters triangular filters.
    std::vector<int> binPoints(numMelFilters + 2);
    for (int i = 0; i < numMelFilters + 2; ++i) {
        float mel = melLow + (melHigh - melLow) * i / (numMelFilters + 1);
        float hz = melToHz(mel);
        int bin = static_cast<int>(std::floor((fftLength + 1) * hz / sampleRate));
        binPoints[i] = std::clamp(bin, 0, numFftBins - 1);
    }

    std::vector<std::vector<float>> filterBank(numMelFilters,
                                                std::vector<float>(numFftBins, 0.0f));
    for (int m = 0; m < numMelFilters; ++m) {
        int left = binPoints[m];
        int center = binPoints[m + 1];
        int right = binPoints[m + 2];

        // Degenerate (zero-width) filters can happen when fftLength/sampleRate
        // is too coarse to resolve the requested number of mel filters at the
        // low end of the spectrum -- leave them all-zero rather than dividing
        // by zero. compute() still produces a value for these filters (0,
        // feeding log(epsilon) downstream), it's just not a meaningful one;
        // callers hitting this should use a larger fftLength or fewer filters.
        if (center > left) {
            for (int k = left; k < center; ++k) {
                filterBank[m][k] = static_cast<float>(k - left) / (center - left);
            }
        }
        if (right > center) {
            for (int k = center; k < right; ++k) {
                filterBank[m][k] = static_cast<float>(right - k) / (right - center);
            }
        }
    }

    return filterBank;
}

std::vector<std::vector<float>> MFCC::compute(const std::vector<float>& signal) {
    auto [real, imag] = stft_->stft(signal);
    const int numFrames = static_cast<int>(real.size());
    const int numFftBins = stft_->fftLength() / 2 + 1;
    constexpr float kLogEpsilon = 1e-10f;

    std::vector<std::vector<float>> result(numFrames);

    for (int f = 0; f < numFrames; ++f) {
        std::vector<float> powerSpectrum(numFftBins);
        for (int k = 0; k < numFftBins; ++k) {
            powerSpectrum[k] = real[f][k] * real[f][k] + imag[f][k] * imag[f][k];
        }

        std::vector<float> logMelEnergies(params_.numMelFilters);
        for (int m = 0; m < params_.numMelFilters; ++m) {
            float energy = 0.0f;
            for (int k = 0; k < numFftBins; ++k) {
                energy += melFilterBank_[m][k] * powerSpectrum[k];
            }
            logMelEnergies[m] = std::log(energy + kLogEpsilon);
        }

        result[f] = dctII(logMelEnergies, params_.numCoefficients, /*orthonormal=*/true);
    }

    LOG(DEBUG) << TAG(kTag) << "Computed MFCC: " << signal.size() << " samples -> " << numFrames
               << " frames x " << params_.numCoefficients << " coefficients." << std::endl;
    return result;
}

}  // namespace speech::dsp
