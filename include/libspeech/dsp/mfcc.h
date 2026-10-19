//
// speech::dsp::MFCC -- Mel-Frequency Cepstral Coefficients.
//
// Composes speech::dsp::STFT (framing + FFT) with a triangular mel
// filterbank (built here, not vendored from AudioFlux -- see the note
// below) and speech::dsp::dct_ii (the mel-filter count is rarely a power of
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

class stft;  // fwd-declared to keep this header light; see stft.h
class dct_ii_plan;

// Flat, frame-major MFCC matrix: frame f, coefficient c at data[f*num_coefficients + c].
struct mfcc_matrix {
    int num_frames = 0;
    int num_coefficients = 0;
    detail::uninit_vector<float> data;
};

/**
 * MFCC: log-mel-filterbank energies -> DCT-II -> cepstral coefficients,
 * the standard feature used across ASR/speaker/keyword-spotting models.
 */
class SPEECH_API mfcc {
   public:
    struct params {
        int sample_rate;
        int num_mel_filters = 40;    // number of triangular mel filters
        int num_coefficients = 13;  // cepstral coefficients kept per frame
        float low_freq_hz = 0.0f;
        float high_freq_hz = 0.0f;   // 0 => sample_rate / 2 (Nyquist)
        int radix2_exp = 10;        // STFT frame length = 2^radix2_exp (1024 default)
        int slide_length = 0;       // 0 => STFT default (fft_length / 4)
    };

    explicit mfcc(params params);
    ~mfcc();

    mfcc(const mfcc&) = delete;
    mfcc& operator=(const mfcc&) = delete;

    // Computes MFCCs for `signal`. Returns a [num_frames][num_coefficients]
    // matrix (num_frames follows the same framing as speech::dsp::STFT).
    std::vector<std::vector<float>> compute(const std::vector<float>& signal);

    // Copy-free variant: one flat output allocation, no per-frame vectors.
    // Frames are processed in cache-sized blocks (real-input FFT -> power ->
    // sparse mel -> log -> DCT), so no spectrogram is ever materialized.
    mfcc_matrix compute_flat(const float* signal, std::size_t n);

    // Log-mel filterbank energies (the "fbank" feature): the same pipeline as compute_flat() minus the DCT.
    // Returns a [num_frames][num_mel_filters] matrix (mfcc_matrix::num_coefficients == num_mel_filters),
    // values are log(mel_energy + 1e-10); MFCCs are exactly the orthonormal DCT-II of these rows.
    mfcc_matrix compute_log_mel_flat(const float* signal, std::size_t n);

   private:
    mfcc_matrix run(const float* signal, std::size_t n, bool apply_dct);

    params params_;
    std::unique_ptr<stft> stft_;
    std::vector<std::vector<float>> mel_filter_bank_;  // [num_mel_filters][fft_length/2 + 1] (dense, reference)

    // Sparse view of mel_filter_bank_: filter m has non-zero weights only on
    // bins [mel_start_[m], mel_start_[m] + mel_len_[m]), stored contiguously
    // from mel_offset_[m] in mel_weights_.
    std::vector<int> mel_start_;
    std::vector<int> mel_len_;
    std::vector<int> mel_offset_;
    std::vector<float> mel_weights_;
    int min_bin_ = 0;  // power spectrum only needs bins [min_bin_, max_bin_]
    int max_bin_ = 0;

    std::unique_ptr<dct_ii_plan> dct_;

    static std::vector<std::vector<float>> build_mel_filter_bank(int num_mel_filters, int fft_length,
                                                                int sample_rate, float low_freq_hz,
                                                                float high_freq_hz);
};

}  // namespace speech::dsp

#endif  // LIBSPEECH_DSP_MFCC_H
