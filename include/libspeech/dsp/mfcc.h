//
// speech::dsp::MFCC -- Mel-Frequency Cepstral Coefficients.
//
// Composes speech::dsp::STFT (framing + FFT) with a triangular mel
// filterbank (built here, not vendored from AudioFlux -- see the note
// below) and speech::dsp::dctII (the mel-filter count is rarely a power of
// 2, so FFT::dct's power-of-2 requirement doesn't fit here).
//
// Why the mel filterbank is hand-written instead of vendored: AudioFlux
// builds mel-scale (and bark/erb/log-scale) filterbanks inside a single
// large, feature-rich engine (bft_algorithm.c, ~635 lines) that also
// handles time-frequency reassignment and temporal statistics (energy/
// rms/zero-crossing-rate) libspeech doesn't need. Vendoring all of that to
// reach a fairly standard, well-documented triangular mel filterbank
// (the same construction used by HTK, librosa, and effectively every
// other speech toolkit) would import a lot of surface area -- and risk --
// for a small, specific need. The two-line Hz<->mel conversion below
// matches AudioFlux's own `auditory_freToMel`/`auditory_melToFre` (HTK-
// style: mel = 2595*log10(1+f/700)) so results stay directly comparable.
//

#ifndef LIBSPEECH_DSP_MFCC_H
#define LIBSPEECH_DSP_MFCC_H

#include <cstddef>
#include <memory>
#include <vector>

#include "libspeech/detail/uninit_vector.h"
#include "libspeech/export.h"

namespace speech::dsp {

class STFT;  // fwd-declared to keep this header light; see stft.h
class DctII;

// Flat, frame-major MFCC matrix: frame f, coefficient c at data[f*numCoefficients + c].
struct MfccMatrix {
    int numFrames = 0;
    int numCoefficients = 0;
    detail::UninitVector<float> data;
};

/**
 * MFCC: log-mel-filterbank energies -> DCT-II -> cepstral coefficients,
 * the standard feature used across ASR/speaker/keyword-spotting models.
 */
class SPEECH_API MFCC {
   public:
    struct Params {
        int sampleRate;
        int numMelFilters = 40;    // number of triangular mel filters
        int numCoefficients = 13;  // cepstral coefficients kept per frame
        float lowFreqHz = 0.0f;
        float highFreqHz = 0.0f;   // 0 => sampleRate / 2 (Nyquist)
        int radix2Exp = 10;        // STFT frame length = 2^radix2Exp (1024 default)
        int slideLength = 0;       // 0 => STFT default (fftLength / 4)
    };

    explicit MFCC(Params params);
    ~MFCC();

    MFCC(const MFCC&) = delete;
    MFCC& operator=(const MFCC&) = delete;

    // Computes MFCCs for `signal`. Returns a [numFrames][numCoefficients]
    // matrix (numFrames follows the same framing as speech::dsp::STFT).
    std::vector<std::vector<float>> compute(const std::vector<float>& signal);

    // Copy-free variant: one flat output allocation, no per-frame vectors.
    // Frames are processed in cache-sized blocks (real-input FFT -> power ->
    // sparse mel -> log -> DCT), so no spectrogram is ever materialized.
    MfccMatrix computeFlat(const float* signal, std::size_t n);

   private:
    Params params_;
    std::unique_ptr<STFT> stft_;
    std::vector<std::vector<float>> melFilterBank_;  // [numMelFilters][fftLength/2 + 1] (dense, reference)

    // Sparse view of melFilterBank_: filter m has non-zero weights only on
    // bins [melStart_[m], melStart_[m] + melLen_[m]), stored contiguously
    // from melOffset_[m] in melWeights_.
    std::vector<int> melStart_;
    std::vector<int> melLen_;
    std::vector<int> melOffset_;
    std::vector<float> melWeights_;
    int minBin_ = 0;  // power spectrum only needs bins [minBin_, maxBin_]
    int maxBin_ = 0;

    std::unique_ptr<DctII> dct_;

    static std::vector<std::vector<float>> buildMelFilterBank(int numMelFilters, int fftLength,
                                                                int sampleRate, float lowFreqHz,
                                                                float highFreqHz);
};

}  // namespace speech::dsp

#endif  // LIBSPEECH_DSP_MFCC_H
