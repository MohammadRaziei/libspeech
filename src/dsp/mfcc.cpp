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
constexpr int kBlockFrames = 64;  // 64 * 512 * 4 B * 2 planes = 256 KiB: stays in L2

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

    // Sparse mel filters: identical weights to the dense bank, minus the zeros.
    const int numFftBins = fftLength / 2 + 1;
    melStart_.assign(params_.numMelFilters, 0);
    melLen_.assign(params_.numMelFilters, 0);
    melOffset_.assign(params_.numMelFilters, 0);
    minBin_ = numFftBins;
    maxBin_ = -1;
    for (int m = 0; m < params_.numMelFilters; ++m) {
        int first = -1;
        int last = -1;
        for (int k = 0; k < numFftBins; ++k) {
            if (melFilterBank_[m][k] != 0.0f) {
                if (first < 0) first = k;
                last = k;
            }
        }
        melOffset_[m] = static_cast<int>(melWeights_.size());
        if (first >= 0) {
            melStart_[m] = first;
            melLen_[m] = last - first + 1;
            melWeights_.insert(melWeights_.end(), melFilterBank_[m].begin() + first,
                               melFilterBank_[m].begin() + last + 1);
            minBin_ = std::min(minBin_, first);
            maxBin_ = std::max(maxBin_, last);
        }
    }
    if (maxBin_ < 0) {
        minBin_ = 0;
        maxBin_ = -1;
    }

    dct_ = std::make_unique<DctII>(params_.numMelFilters, params_.numCoefficients,
                                   /*orthonormal=*/true);

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

MfccMatrix MFCC::computeFlat(const float* signal, std::size_t n) {
    constexpr float kLogEpsilon = 1e-10f;
    const int fftLength = stft_->fftLength();
    const int numBins = fftLength / 2 + 1;
    const int numFrames = stft_->calTimeLength(static_cast<int>(n));
    const int numMel = params_.numMelFilters;
    const int numCoef = params_.numCoefficients;

    MfccMatrix out;
    out.numCoefficients = numCoef;
    if (numFrames <= 0) {
        return out;
    }
    out.numFrames = numFrames;
    out.data.resize(static_cast<std::size_t>(numFrames) * numCoef);  // fully overwritten below

    const int numBlocks = (numFrames + kBlockFrames - 1) / kBlockFrames;
    STFT* stft = stft_.get();
    float* outData = out.data.data();

    // Blocks are independent, so they can run on separate threads; each thread
    // owns its power-spectrum block buffer and per-frame scratch.
#pragma omp parallel if (numFrames >= 256)
    {
        std::vector<float> powerBlock(static_cast<std::size_t>(kBlockFrames) * numBins);
        std::vector<float> melHeap(static_cast<std::size_t>(numMel));
        float* melBuf = melHeap.data();

#pragma omp for schedule(static)
        for (int blk = 0; blk < numBlocks; ++blk) {
            const int f0 = blk * kBlockFrames;
            const int frames = std::min(kBlockFrames, numFrames - f0);
            stft->powerSpectra(signal, n, f0, frames, powerBlock.data());

            for (int fr = 0; fr < frames; ++fr) {
                const float* power = powerBlock.data() + static_cast<std::size_t>(fr) * numBins;
                for (int m = 0; m < numMel; ++m) {
                    const float* w = melWeights_.data() + melOffset_[m];
                    const float* p = power + melStart_[m];
                    float energy = 0.0f;
                    for (int j = 0; j < melLen_[m]; ++j) {
                        energy += w[j] * p[j];
                    }
                    melBuf[m] = std::log(energy + kLogEpsilon);
                }
                dct_->apply(melBuf, outData + static_cast<std::size_t>(f0 + fr) * numCoef);
            }
        }
    }

    LOG(DEBUG) << TAG(kTag) << "Computed MFCC: " << n << " samples -> " << numFrames
               << " frames x " << numCoef << " coefficients." << std::endl;
    return out;
}

std::vector<std::vector<float>> MFCC::compute(const std::vector<float>& signal) {
    // Legacy nested-vector API: one copy out of the flat result.
    MfccMatrix flat = computeFlat(signal.data(), signal.size());
    std::vector<std::vector<float>> result(flat.numFrames);
    for (int f = 0; f < flat.numFrames; ++f) {
        const auto begin = flat.data.begin() + static_cast<std::ptrdiff_t>(f) * flat.numCoefficients;
        result[f].assign(begin, begin + flat.numCoefficients);
    }
    return result;
}

}  // namespace speech::dsp
