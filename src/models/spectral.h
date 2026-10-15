// Internal (not installed): the small spectral toolbox the spectrogram-domain denoisers share
// (GTCRN, DPDFNet, ...): a mixed-radix real FFT and a torch.stft(center=True) / torch.istft compatible
// frame loop. Self-contained on purpose: speech::models does not link speech::dsp.
#ifndef LIBSPEECH_MODELS_SPECTRAL_H
#define LIBSPEECH_MODELS_SPECTRAL_H

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace speech::models::detail {

/**
 * Real FFT of a fixed size (any n >= 1; mixed-radix, so 320 = 2^6*5 or 960 = 2^6*3*5 cost
 * O(n * sum of prime factors), and a prime n falls back to a plain O(n^2) DFT).
 * const methods are thread-safe (no shared scratch).
 */
class real_fft {
   public:
    explicit real_fft(std::size_t n) : n_(n), twiddle_(n) {
        if (n < 1) throw std::invalid_argument("real_fft: size must be >= 1");
        const double pi = std::acos(-1.0);
        for (std::size_t k = 0; k < n; ++k) {
            const double angle = -2.0 * pi * static_cast<double>(k) / static_cast<double>(n);
            twiddle_[k] = {std::cos(angle), std::sin(angle)};
        }
        std::size_t rest = n;
        for (std::size_t p = 2; p * p <= rest; ++p) {
            while (rest % p == 0) {
                factors_.push_back(p);
                rest /= p;
            }
        }
        if (rest > 1) factors_.push_back(rest);
    }

    std::size_t size() const { return n_; }
    std::size_t bins() const { return n_ / 2 + 1; }

    /** x: n real samples -> re, im: n/2 + 1 bins each (unnormalized, like torch.fft.rfft). */
    void forward(const float* x, float* re, float* im) const {
        std::vector<cplx> in(n_), out(n_);
        for (std::size_t i = 0; i < n_; ++i) in[i] = x[i];
        transform(in.data(), 1, out.data(), n_, 0);
        for (std::size_t k = 0; k < bins(); ++k) {
            re[k] = static_cast<float>(out[k].real());
            im[k] = static_cast<float>(out[k].imag());
        }
    }

    /** re, im: n/2 + 1 bins -> out: n real samples, scaled by 1/n (like torch.fft.irfft). */
    void inverse(const float* re, const float* im, float* out) const {
        // Hermitian-complete the spectrum, then ifft(X) = conj(fft(conj(X))) / n.
        std::vector<cplx> spectrum(n_), time(n_);
        for (std::size_t k = 0; k < bins(); ++k) {
            // The DC (and, for even n, Nyquist) bin of a real signal is real.
            const bool self_conjugate = k == 0 || (n_ % 2 == 0 && k == n_ / 2);
            const double imag = self_conjugate ? 0.0 : static_cast<double>(im[k]);
            spectrum[k] = std::conj(cplx(static_cast<double>(re[k]), imag));
            if (!self_conjugate) spectrum[n_ - k] = std::conj(cplx(static_cast<double>(re[k]), -imag));
        }
        transform(spectrum.data(), 1, time.data(), n_, 0);
        const double scale = 1.0 / static_cast<double>(n_);
        for (std::size_t i = 0; i < n_; ++i) out[i] = static_cast<float>(time[i].real() * scale);
    }

   private:
    using cplx = std::complex<double>;

    // out[0..n) = DFT of in[0], in[stride], ..., in[(n-1)*stride]; factors_[fi..] factor n.
    void transform(const cplx* in, std::size_t stride, cplx* out, std::size_t n, std::size_t fi) const {
        if (n == 1) {
            out[0] = in[0];
            return;
        }
        const std::size_t p = factors_[fi];
        const std::size_t m = n / p;
        for (std::size_t r = 0; r < p; ++r) transform(in + r * stride, stride * p, out + r * m, m, fi + 1);

        // W_n^j = W_N^(j * step); the radix-p butterfly uses W_p^j = W_N^(j * m * step).
        const std::size_t step = n_ / n;
        std::vector<cplx> result(n), scaled(p);
        for (std::size_t k = 0; k < m; ++k) {
            for (std::size_t r = 0; r < p; ++r) scaled[r] = out[r * m + k] * twiddle_[(r * k * step) % n_];
            for (std::size_t q = 0; q < p; ++q) {
                cplx sum = 0.0;
                for (std::size_t r = 0; r < p; ++r) sum += scaled[r] * twiddle_[(r * q * m * step) % n_];
                result[q * m + k] = sum;
            }
        }
        std::copy(result.begin(), result.end(), out);
    }

    std::size_t n_;
    std::vector<std::complex<double>> twiddle_;  // W_N^k = exp(-2*pi*i*k/N)
    std::vector<std::size_t> factors_;           // prime factors of N, smallest first
};

/** sqrt of the periodic Hann window (torch.hann_window(n).pow(0.5)). */
inline std::vector<float> sqrt_hann_window(std::size_t n) {
    std::vector<float> w(n);
    const double pi = std::acos(-1.0);
    for (std::size_t i = 0; i < n; ++i) {
        w[i] = static_cast<float>(std::sqrt(0.5 - 0.5 * std::cos(2.0 * pi * static_cast<double>(i) / static_cast<double>(n))));
    }
    return w;
}

/** The Vorbis power-complementary window: sin(pi/2 * sin^2(pi * (i + 0.5) / n)). */
inline std::vector<float> vorbis_window(std::size_t n) {
    std::vector<float> w(n);
    const double pi = std::acos(-1.0);
    for (std::size_t i = 0; i < n; ++i) {
        const double s = std::sin(pi * (static_cast<double>(i) + 0.5) / static_cast<double>(n));
        w[i] = static_cast<float>(std::sin(0.5 * pi * s * s));
    }
    return w;
}

/** Index of sample `i` of a signal of length n reflected at both ends (torch pad_mode="reflect"). */
inline std::size_t reflect_index(std::ptrdiff_t i, std::size_t n) {
    if (n <= 1) return 0;
    const std::ptrdiff_t period = 2 * (static_cast<std::ptrdiff_t>(n) - 1);
    i %= period;
    if (i < 0) i += period;
    return static_cast<std::size_t>(i) < n ? static_cast<std::size_t>(i) : static_cast<std::size_t>(period - i);
}

/**
 * Runs `per_frame(re, im)` on every STFT frame of x and overlap-adds the result back, exactly like
 *     torch.istft(f(torch.stft(x, n_fft, hop, n_fft, window, center=True)), ..., center=True)
 * with f applied per frame (re/im are the n_fft/2 + 1 one-sided bins, edited in place).
 * The output has the same length as x; with an identity `per_frame` it reproduces x.
 * `window` has n_fft samples and must satisfy the overlap-add condition for `hop`.
 *
 * `shift` is for models that answer late: a causal network whose output for frame t describes the
 * audio `shift` samples earlier. The output is advanced by that many samples to line up with the
 * input again, and enough extra frames of silence are fed after the end of x so that the last
 * `shift` samples are produced by the model too instead of being lost.
 */
template <typename PerFrame>
std::vector<float> stft_apply_istft(const float* x, std::size_t n, const real_fft& fft,
                                    const std::vector<float>& window, std::size_t hop, PerFrame&& per_frame,
                                    std::size_t shift = 0) {
    const std::size_t n_fft = fft.size();
    const std::size_t pad = n_fft / 2;
    const std::size_t flush = (shift + hop - 1) / hop;
    const std::size_t frames = 1 + n / hop + flush;
    const std::size_t total = (frames - 1) * hop + n_fft;

    std::vector<float> ola(total, 0.0f);  // padded coordinates: index p <-> padded sample p
    std::vector<float> env(total, 0.0f);  // sum of window^2: torch.istft's normalisation
    std::vector<float> frame(n_fft), re(fft.bins()), im(fft.bins());

    for (std::size_t t = 0; t < frames; ++t) {
        const std::ptrdiff_t first = static_cast<std::ptrdiff_t>(t * hop) - static_cast<std::ptrdiff_t>(pad);
        for (std::size_t j = 0; j < n_fft; ++j) {
            const std::ptrdiff_t idx = first + static_cast<std::ptrdiff_t>(j);
            // reflect-padded like torch's center=True, and silence beyond that (only the flush frames
            // of a `shift` > 0 reach there)
            const float sample = idx >= static_cast<std::ptrdiff_t>(n + pad) ? 0.0f : x[reflect_index(idx, n)];
            frame[j] = sample * window[j];
        }
        fft.forward(frame.data(), re.data(), im.data());
        per_frame(re.data(), im.data());
        fft.inverse(re.data(), im.data(), frame.data());
        for (std::size_t j = 0; j < n_fft; ++j) {
            ola[t * hop + j] += frame[j] * window[j];
            env[t * hop + j] += window[j] * window[j];
        }
    }

    std::vector<float> out(n, 0.0f);
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t p = i + pad + shift;
        if (p < total && env[p] > 1e-8f) out[i] = ola[p] / env[p];
    }
    return out;
}

}  // namespace speech::models::detail

#endif  // LIBSPEECH_MODELS_SPECTRAL_H
