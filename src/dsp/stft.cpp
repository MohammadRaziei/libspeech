#include "libspeech/dsp/stft.h"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "aixlog.hpp"
#include "libspeech/detail/log.h"
#include "dsp/flux_window.h"  // Vendored AudioFlux: window_calFFTWindow()
#include "real_fft.h"
#include "window_type_internal.h"
#include "stft_algorithm.h"  // Vendored AudioFlux C header (src/third_party/audioflux)

namespace speech::dsp {

namespace {
constexpr const char* log_tag = "speech::dsp::STFT";
}

stft::stft(int radix2_exp, window_type type, int slide_length)
    : stft_obj_(nullptr), fft_length_(0), slide_length_(0) {
    int is_continue = 0;
    int* slide_len_ptr = (slide_length > 0) ? &slide_length : nullptr;
    ::WindowType audioflux_type = to_audioflux(type);

    if (stftObj_new(&stft_obj_, radix2_exp, &audioflux_type, slide_len_ptr, &is_continue) != 0) {
        SPEECH_LOG(ERROR) << TAG(log_tag) << "Failed to create STFT object (radix2_exp=" << radix2_exp
                   << "; must be in [1, 30])." << std::endl;
        throw std::invalid_argument("Invalid radix2_exp for STFT (must be in [1, 30]).");
    }

    fft_length_ = 1 << radix2_exp;
    slide_length_ = (slide_length > 0) ? slide_length : fft_length_ / 4;

    if (rfft_impl::real_frame_fft::supports(fft_length_)) {
        // The real-FFT path must apply the very same analysis window AudioFlux's STFT
        // does. stftObj_new() builds it with window_calFFTWindow(type, fft_length),
        // so call that same (unpatched, public) function instead of reaching into the
        // opaque STFT object -- no change to the vendored sources is needed.
        if (float* window = window_calFFTWindow(audioflux_type, fft_length_)) {
            rfft_ = std::make_unique<rfft_impl::real_frame_fft>(fft_length_, window);  // copies it
            std::free(window);
        }
    }

    SPEECH_LOG(DEBUG) << TAG(log_tag) << "Created STFT: fft_length=" << fft_length_
               << ", slide_length=" << slide_length_ << ", window_type_="
               << static_cast<int>(type) << std::endl;
}

stft::~stft() {
    if (stft_obj_) {
        stftObj_free(stft_obj_);
        stft_obj_ = nullptr;
    }
}

int stft::cal_time_length(int data_length) const {
    return stftObj_calTimeLength(stft_obj_, data_length);
}

int stft::cal_data_length(int time_length) const {
    return stftObj_calDataLength(stft_obj_, time_length);
}

namespace {
// Frame loops are embarrassingly parallel; the threshold keeps tiny inputs from
// paying thread start-up cost.
constexpr int parallel_frame_threshold = 128;
}  // namespace

namespace {
// Forward STFT of every frame with the real-input engine, written either as split planes (S = 1:
// re0/im0 are the two planes) or as interleaved complex numbers (S = 2: re0 = p, im0 = p + 1).
// `row_bins` is the bins per output row (N full, N/2+1 one-sided); `mirror` fills the upper half
// with the conjugate mirror X[N-k] = conj(X[k]) (full spectrum only).
template <int S>
void stft_frames(const rfft_impl::real_frame_fft& fft, const float* data, int time_length, int N, int hop,
                int row_bins, bool mirror, float* re0, float* im0) {
    const int half = N / 2;
#pragma omp parallel for schedule(static) if (time_length >= parallel_frame_threshold)
    for (int f = 0; f < time_length; ++f) {
        float* re = re0 + static_cast<std::size_t>(f) * row_bins * S;
        float* im = im0 + static_cast<std::size_t>(f) * row_bins * S;
        const float* frame = data + static_cast<std::size_t>(f) * hop;
        if constexpr (S == 1) {
            fft.spectrum(frame, re, im);
        } else {
            fft.spectrum_interleaved(frame, re);
        }
        if (mirror) {
            for (int k = 1; k < half; ++k) {
                re[S * (N - k)] = re[S * k];
                im[S * (N - k)] = -im[S * k];
            }
        }
    }
}
}  // namespace

int stft::stft_into(const float* data, std::size_t n, float* real, float* imag) {
    const int data_length = static_cast<int>(n);
    const int time_length = cal_time_length(data_length);
    if (time_length <= 0) {
        SPEECH_LOG(DEBUG) << TAG(log_tag) << "stft(): input length " << data_length
                   << " is shorter than fft_length=" << fft_length_
                   << "; returning zero frames." << std::endl;
        return 0;
    }
    if (!rfft_) {
        // AudioFlux writes frame-major into flat buffers and does not modify the
        // input (the const_cast only satisfies its non-const C signature).
        stftObj_stft(stft_obj_, const_cast<float*>(data), data_length, real, imag);
        return time_length;
    }

    // Real input: compute the N/2+1 unique bins per frame, then fill the upper half
    // with the conjugate mirror X[N-k] = conj(X[k]).
    stft_frames<1>(*rfft_, data, time_length, fft_length_, slide_length_, fft_length_, true, real, imag);
    return time_length;
}

int stft::stft_onesided_into(const float* data, std::size_t n, float* real, float* imag) {
    const int time_length = cal_time_length(static_cast<int>(n));
    if (time_length <= 0) {
        return 0;
    }
    const int bins = fft_length_ / 2 + 1;
    if (!rfft_) {
        // Unsupported length: full complex STFT, then keep the first `bins` columns.
        std::vector<float> full_re(static_cast<std::size_t>(time_length) * fft_length_);
        std::vector<float> full_im(full_re.size());
        stft_into(data, n, full_re.data(), full_im.data());
        for (int f = 0; f < time_length; ++f) {
            std::copy_n(full_re.begin() + static_cast<std::size_t>(f) * fft_length_, bins,
                        real + static_cast<std::size_t>(f) * bins);
            std::copy_n(full_im.begin() + static_cast<std::size_t>(f) * fft_length_, bins,
                        imag + static_cast<std::size_t>(f) * bins);
        }
        return time_length;
    }
    stft_frames<1>(*rfft_, data, time_length, fft_length_, slide_length_, bins, false, real, imag);
    return time_length;
}

int stft::stft_complex_impl(span<const float> data, frame_view<std::complex<float>> out, bool onesided) {
    const int frames = std::max(0, cal_time_length(static_cast<int>(data.size())));
    const std::size_t bins = static_cast<std::size_t>(num_bins(onesided));
    if (out.frame_length() != bins || !out.contiguous() || out.size() < static_cast<std::size_t>(frames)) {
        SPEECH_LOG(ERROR) << TAG(log_tag) << "stft(): the complex output must be contiguous rows of " << bins
                   << " values with at least " << frames << " frames." << std::endl;
        throw std::invalid_argument("stft_into(): the complex output view does not fit the spectrogram.");
    }
    if (frames == 0) {
        return 0;
    }
    float* base = reinterpret_cast<float*>(out.data());  // std::complex<float> is two consecutive floats
    if (rfft_) {
        stft_frames<2>(*rfft_, data.data(), frames, fft_length_, slide_length_, static_cast<int>(bins),
                      !onesided, base, base + 1);
        return frames;
    }
    // Unsupported length: planar result (AudioFlux) interleaved afterwards. Allocates, but only
    // for lengths outside 16..8192.
    std::vector<float> re(static_cast<std::size_t>(frames) * bins), im(re.size());
    if (onesided) {
        stft_onesided_into(data.data(), data.size(), re.data(), im.data());
    } else {
        stft_into(data.data(), data.size(), re.data(), im.data());
    }
    for (std::size_t i = 0; i < re.size(); ++i) {
        out.data()[i] = std::complex<float>(re[i], im[i]);
    }
    return frames;
}

int stft::stft_into(span<const float> data, frame_view<std::complex<float>> out) {
    return stft_complex_impl(data, out, false);
}

int stft::stft_onesided_into(span<const float> data, frame_view<std::complex<float>> out) {
    return stft_complex_impl(data, out, true);
}

namespace {
// Output planes must be contiguous rows of `bins` samples with room for `frames` frames.
void require_output_rows(const frame_view<float>& real, const frame_view<float>& imag, int bins,
                       int frames, const char* what) {
    const std::size_t b = static_cast<std::size_t>(bins);
    if (real.frame_length() != b || imag.frame_length() != b || !real.contiguous() || !imag.contiguous() ||
        real.size() < static_cast<std::size_t>(frames) || imag.size() < static_cast<std::size_t>(frames)) {
        SPEECH_LOG(ERROR) << TAG(log_tag) << what << ": real/imag must be contiguous rows of " << bins
                   << " samples with at least " << frames << " frames." << std::endl;
        throw std::invalid_argument(std::string(what) + ": the real/imag views do not fit the spectrogram.");
    }
}
}  // namespace

int stft::stft_into(span<const float> data, frame_view<float> real, frame_view<float> imag) {
    const int frames = std::max(0, cal_time_length(static_cast<int>(data.size())));
    require_output_rows(real, imag, fft_length_, frames, "stft_into()");
    return stft_into(data.data(), data.size(), real.data(), imag.data());
}

int stft::stft_onesided_into(span<const float> data, frame_view<float> real, frame_view<float> imag) {
    const int frames = std::max(0, cal_time_length(static_cast<int>(data.size())));
    require_output_rows(real, imag, fft_length_ / 2 + 1, frames, "stft_onesided_into()");
    return stft_onesided_into(data.data(), data.size(), real.data(), imag.data());
}

void stft::power_spectra(const float* data, std::size_t n, int first_frame, int num_frames,
                        float* power) {
    const int bins = fft_length_ / 2 + 1;
    const int hop = slide_length_;
    if (num_frames <= 0) {
        return;
    }
    if (rfft_) {
        const rfft_impl::real_frame_fft& fft = *rfft_;
        for (int f = 0; f < num_frames; ++f) {
            fft.power(data + static_cast<std::size_t>(first_frame + f) * hop,
                      power + static_cast<std::size_t>(f) * bins);
        }
        return;
    }
    // Fallback for unsupported lengths: complex STFT of just these frames' samples.
    const std::size_t start = static_cast<std::size_t>(first_frame) * hop;
    const std::size_t len = static_cast<std::size_t>(num_frames - 1) * hop + fft_length_;
    (void)n;
    std::vector<float> re(static_cast<std::size_t>(num_frames) * fft_length_);
    std::vector<float> im(re.size());
    stft_into(data + start, len, re.data(), im.data());
    for (int f = 0; f < num_frames; ++f) {
        const float* r = re.data() + static_cast<std::size_t>(f) * fft_length_;
        const float* i = im.data() + static_cast<std::size_t>(f) * fft_length_;
        float* p = power + static_cast<std::size_t>(f) * bins;
        for (int k = 0; k < bins; ++k) p[k] = r[k] * r[k] + i[k] * i[k];
    }
}

complex_spectrogram stft::spectrogram(const float* data, std::size_t n, bool onesided) {
    complex_spectrogram out;
    out.fft_length = fft_length_;
    out.num_bins = num_bins(onesided);
    const int time_length = cal_time_length(static_cast<int>(n));
    if (time_length <= 0) {
        return out;
    }
    const std::size_t total = static_cast<std::size_t>(time_length) * out.num_bins;
    out.real.resize(total);  // uninitialized: fully overwritten below
    out.imag.resize(total);
    out.num_frames = onesided ? stft_onesided_into(data, n, out.real.data(), out.imag.data())
                             : stft_into(data, n, out.real.data(), out.imag.data());
    return out;
}

std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> stft::compute(
    const std::vector<float>& data) {
    // Legacy nested-vector API (kept for source compatibility): one pass of
    // per-frame copies out of the flat spectrogram. Prefer spectrogram().
    complex_spectrogram sp = spectrogram(data.data(), data.size());
    if (sp.num_frames <= 0) {
        return {};
    }
    std::vector<std::vector<float>> real(sp.num_frames);
    std::vector<std::vector<float>> imag(sp.num_frames);
    for (int i = 0; i < sp.num_frames; ++i) {
        const std::size_t off = static_cast<std::size_t>(i) * fft_length_;
        real[i].assign(sp.real.begin() + off, sp.real.begin() + off + fft_length_);
        imag[i].assign(sp.imag.begin() + off, sp.imag.begin() + off + fft_length_);
    }
    SPEECH_LOG(DEBUG) << TAG(log_tag) << "Computed STFT: " << data.size() << " samples -> "
               << sp.num_frames << " frames." << std::endl;
    return {std::move(real), std::move(imag)};
}

namespace {
// Weighted / plain overlap-add of the inverse-transformed frames (see STFT::istft_into). The input
// is read as split planes (S = 1) or interleaved complex numbers (S = 2: re0 = p, im0 = p + 1).
template <int S>
void istft_frames(const rfft_impl::real_frame_fft& fft, int N, int hop, const float* re0, const float* im0,
                 int num_frames, bool onesided, int method_type, std::size_t data_length, float* out) {
    const int half = N / 2;
    const int bins = onesided ? half + 1 : N;

    // Per method, frame k is multiplied by win_a[k] before overlap-add and win_b[k] is accumulated
    // into the normalizer (weighted: w and w^2; plain: 1 and w) -- exactly what AudioFlux's
    // stftObj_istft() does.
    const float* w = fft.window();
    std::vector<float> win_a(static_cast<std::size_t>(N)), win_b(static_cast<std::size_t>(N));
    for (int k = 0; k < N; ++k) {
        win_a[k] = (method_type == 0) ? w[k] : 1.0f;
        win_b[k] = (method_type == 0) ? w[k] * w[k] : w[k];
    }

    std::fill(out, out + data_length, 0.0f);
    std::vector<float> norm(data_length, 0.0f);

    // Frames are inverse-transformed a block at a time (in parallel: they are independent), then
    // overlap-added serially in frame order, so the sum is accumulated in the same order as the
    // sequential reference regardless of the thread count.
    const int block_frames = std::max(16, (1 << 17) / N);
    detail::uninit_vector<float> block(static_cast<std::size_t>(block_frames) * N);
    const bool parallel = num_frames >= parallel_frame_threshold;
    for (int base = 0; base < num_frames; base += block_frames) {
        const int count = std::min(block_frames, num_frames - base);
#pragma omp parallel for schedule(static) if (parallel)
        for (int i = 0; i < count; ++i) {
            const std::size_t f = static_cast<std::size_t>(base + i);
            const float* re = re0 + f * bins * S;
            const float* im = im0 + f * bins * S;
            float* x = block.data() + static_cast<std::size_t>(i) * N;
            if (onesided) {
                if constexpr (S == 1) {
                    fft.inverse(re, im, x);
                } else {
                    fft.inverse_interleaved(re, x);
                }
            } else {
                // Full spectrum: take the real part of its inverse transform, which equals the
                // inverse transform of its conjugate-symmetric part (X[k] + conj(X[N-k])) / 2.
                float sre[rfft_impl::real_frame_fft::max_length / 2 + 1];
                float sim[rfft_impl::real_frame_fft::max_length / 2 + 1];
                sre[0] = re[0];
                sim[0] = 0.0f;
                for (int k = 1; k < half; ++k) {
                    sre[k] = 0.5f * (re[S * k] + re[S * (N - k)]);
                    sim[k] = 0.5f * (im[S * k] - im[S * (N - k)]);
                }
                sre[half] = re[S * half];
                sim[half] = 0.0f;
                fft.inverse(sre, sim, x);
            }
        }
        for (int i = 0; i < count; ++i) {
            const float* x = block.data() + static_cast<std::size_t>(i) * N;
            const std::size_t start = static_cast<std::size_t>(base + i) * hop;
            float* o = out + start;
            float* nr = norm.data() + start;
            for (int k = 0; k < N; ++k) {
                o[k] += x[k] * win_a[k];
                nr[k] += win_b[k];
            }
        }
    }

    for (std::size_t j = 0; j < data_length; ++j) {
        out[j] /= (norm[j] < 1e-6f) ? 1.0f : norm[j];
    }
}
}  // namespace

void stft::istft_into(const float* real, const float* imag, int num_frames, bool onesided,
                     int method_type, float* out) {
    if (num_frames <= 0 || real == nullptr || imag == nullptr || out == nullptr) {
        SPEECH_LOG(ERROR) << TAG(log_tag) << "istft() requires at least one frame and non-null buffers."
                   << std::endl;
        throw std::invalid_argument("istft() requires non-empty real/imag inputs.");
    }
    if (method_type != 0 && method_type != 1) {
        SPEECH_LOG(ERROR) << TAG(log_tag) << "istft(): method_type must be 0 or 1, got " << method_type
                   << "." << std::endl;
        throw std::invalid_argument("istft(): method_type must be 0 (weighted) or 1 (plain).");
    }

    const int N = fft_length_;
    const int half = N / 2;
    const int bins = num_bins(onesided);
    const std::size_t data_length = static_cast<std::size_t>(cal_data_length(num_frames));

    if (!rfft_) {
        // Unsupported length: AudioFlux's complex kernel, fed the full spectrum (mirroring the
        // one-sided input if that is what we got). It accumulates into `out`, so zero it first.
        const float* full_re = real;
        const float* full_im = imag;
        std::vector<float> mirror_re, mirror_im;
        if (onesided) {
            mirror_re.resize(static_cast<std::size_t>(num_frames) * N);
            mirror_im.resize(mirror_re.size());
            for (int f = 0; f < num_frames; ++f) {
                const float* re = real + static_cast<std::size_t>(f) * bins;
                const float* im = imag + static_cast<std::size_t>(f) * bins;
                float* fr = mirror_re.data() + static_cast<std::size_t>(f) * N;
                float* fi = mirror_im.data() + static_cast<std::size_t>(f) * N;
                for (int k = 0; k < bins; ++k) {
                    fr[k] = re[k];
                    fi[k] = (k == 0 || k == half) ? 0.0f : im[k];  // DC/Nyquist have no imaginary part
                }
                for (int k = 1; k < half; ++k) {
                    fr[N - k] = re[k];
                    fi[N - k] = -im[k];
                }
            }
            full_re = mirror_re.data();
            full_im = mirror_im.data();
        }
        std::fill(out, out + data_length, 0.0f);
        stftObj_istft(stft_obj_, const_cast<float*>(full_re), const_cast<float*>(full_im), num_frames,
                      method_type, out);
        return;
    }

    istft_frames<1>(*rfft_, N, slide_length_, real, imag, num_frames, onesided, method_type, data_length, out);
}

void stft::istft_into(frame_view<const std::complex<float>> spec, int method_type, span<float> out) {
    const std::size_t N = static_cast<std::size_t>(fft_length_);
    if (!spec.contiguous() || (spec.frame_length() != N && spec.frame_length() != N / 2 + 1)) {
        SPEECH_LOG(ERROR) << TAG(log_tag) << "istft(): the complex spectrum must be contiguous rows of " << N
                   << " (full) or " << N / 2 + 1 << " (one-sided) values." << std::endl;
        throw std::invalid_argument("istft(): the complex spectrum view does not fit this STFT.");
    }
    if (spec.empty()) {
        throw std::invalid_argument("istft() requires non-empty real/imag inputs.");
    }
    if (method_type != 0 && method_type != 1) {
        throw std::invalid_argument("istft(): method_type must be 0 (weighted) or 1 (plain).");
    }
    const int frames = static_cast<int>(spec.size());
    const std::size_t data_length = static_cast<std::size_t>(cal_data_length(frames));
    if (out.size() < data_length) {
        SPEECH_LOG(ERROR) << TAG(log_tag) << "istft(): output holds " << out.size() << " samples, needs "
                   << data_length << "." << std::endl;
        throw std::invalid_argument("istft(): the output span is too small.");
    }
    const bool onesided = spec.frame_length() != N;
    const float* base = reinterpret_cast<const float*>(spec.data());  // two consecutive floats per value
    if (rfft_) {
        istft_frames<2>(*rfft_, fft_length_, slide_length_, base, base + 1, frames, onesided, method_type,
                       data_length, out.data());
        return;
    }
    // Unsupported length: split the planes and use the planar (AudioFlux) path. Allocates, but
    // only for lengths outside 16..8192.
    const std::size_t count = spec.size() * spec.frame_length();
    std::vector<float> re(count), im(count);
    for (std::size_t i = 0; i < count; ++i) {
        re[i] = spec.data()[i].real();
        im[i] = spec.data()[i].imag();
    }
    istft_into(re.data(), im.data(), frames, onesided, method_type, out.data());
}

detail::uninit_vector<float> stft::istft(frame_view<const std::complex<float>> spec, int method_type) {
    detail::uninit_vector<float> out;
    if (!spec.empty()) {
        out.resize(static_cast<std::size_t>(cal_data_length(static_cast<int>(spec.size()))));
    }
    istft_into(spec, method_type, span<float>(out.data(), out.size()));
    return out;
}

void stft::istft_into(frame_view<const float> real, frame_view<const float> imag, int method_type,
                     span<float> out) {
    const std::size_t N = static_cast<std::size_t>(fft_length_);
    const bool shape_ok = real.size() == imag.size() && real.frame_length() == imag.frame_length() &&
                         real.contiguous() && imag.contiguous() &&
                         (real.frame_length() == N || real.frame_length() == N / 2 + 1);
    if (!shape_ok) {
        SPEECH_LOG(ERROR) << TAG(log_tag) << "istft(): real/imag must be equal-sized contiguous rows of "
                   << N << " (full) or " << N / 2 + 1 << " (one-sided) samples." << std::endl;
        throw std::invalid_argument("istft(): the real/imag views do not fit this STFT.");
    }
    if (real.empty()) {
        throw std::invalid_argument("istft() requires non-empty real/imag inputs.");
    }
    const int frames = static_cast<int>(real.size());
    if (out.size() < static_cast<std::size_t>(cal_data_length(frames))) {
        SPEECH_LOG(ERROR) << TAG(log_tag) << "istft(): output holds " << out.size() << " samples, needs "
                   << cal_data_length(frames) << "." << std::endl;
        throw std::invalid_argument("istft(): the output span is too small.");
    }
    istft_into(real.data(), imag.data(), frames, real.frame_length() != N, method_type, out.data());
}

detail::uninit_vector<float> stft::istft(frame_view<const float> real, frame_view<const float> imag,
                                         int method_type) {
    detail::uninit_vector<float> out;
    if (!real.empty()) {
        out.resize(static_cast<std::size_t>(cal_data_length(static_cast<int>(real.size()))));
    }
    istft_into(real, imag, method_type, span<float>(out.data(), out.size()));
    return out;
}

detail::uninit_vector<float> stft::istft(const float* real, const float* imag, int num_frames,
                                         bool onesided, int method_type) {
    detail::uninit_vector<float> out;
    if (num_frames > 0) {
        out.resize(static_cast<std::size_t>(cal_data_length(num_frames)));  // fully written below
    }
    istft_into(real, imag, num_frames, onesided, method_type, out.data());  // validates the arguments
    SPEECH_LOG(DEBUG) << TAG(log_tag) << "Computed ISTFT: " << num_frames << " frames -> " << out.size()
               << " samples." << std::endl;
    return out;
}

detail::uninit_vector<float> stft::istft(const complex_spectrogram& spec, int method_type) {
    if (spec.num_bins != fft_length_ && spec.num_bins != fft_length_ / 2 + 1) {
        SPEECH_LOG(ERROR) << TAG(log_tag) << "istft(): spectrogram has " << spec.num_bins
                   << " bins per frame, expected " << fft_length_ << " or " << fft_length_ / 2 + 1 << "."
                   << std::endl;
        throw std::invalid_argument("istft(): the spectrogram's bin count does not match fft_length().");
    }
    return istft(spec.real.data(), spec.imag.data(), spec.num_frames, spec.num_bins != fft_length_, method_type);
}

std::vector<float> stft::istft(const std::vector<std::vector<float>>& real,
                                const std::vector<std::vector<float>>& imag, int method_type) {
    const int time_length = static_cast<int>(real.size());
    if (time_length == 0 || imag.size() != real.size()) {
        SPEECH_LOG(ERROR) << TAG(log_tag)
                   << "istft() requires non-empty, equal-length real/imag frame lists."
                   << std::endl;
        throw std::invalid_argument("istft() requires non-empty, equal-length real/imag inputs.");
    }
    for (int i = 0; i < time_length; ++i) {
        if (static_cast<int>(real[i].size()) != fft_length_ ||
            static_cast<int>(imag[i].size()) != fft_length_) {
            SPEECH_LOG(ERROR) << TAG(log_tag) << "istft(): frame " << i << " has the wrong length."
                       << std::endl;
            throw std::invalid_argument("istft(): every frame must have length fft_length().");
        }
    }

    // Flatten into the frame-major layout of the flat API.
    detail::uninit_vector<float> flat_real(static_cast<std::size_t>(time_length) * fft_length_);
    detail::uninit_vector<float> flat_imag(flat_real.size());
    for (int i = 0; i < time_length; ++i) {
        std::copy(real[i].begin(), real[i].end(), flat_real.begin() + static_cast<long>(i) * fft_length_);
        std::copy(imag[i].begin(), imag[i].end(), flat_imag.begin() + static_cast<long>(i) * fft_length_);
    }
    const detail::uninit_vector<float> out = istft(flat_real.data(), flat_imag.data(), time_length, false, method_type);
    return std::vector<float>(out.begin(), out.end());
}

}  // namespace speech::dsp
