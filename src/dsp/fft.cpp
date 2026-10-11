#include "libspeech/dsp/fft.h"

#include <stdexcept>
#include <string>

#include "aixlog.hpp"
#include "libspeech/detail/log.h"
#include "libspeech/detail/uninit_vector.h"
#include "dsp/fft_algorithm.h"  // Vendored AudioFlux C header (src/third_party/audioflux)
#include "real_fft.h"

namespace speech::dsp {

namespace {
constexpr const char* log_tag = "speech::dsp::FFT";

void check_length(const std::vector<float>& v, int expected, const char* arg_name) {
    if (!v.empty() && static_cast<int>(v.size()) != expected) {
        SPEECH_LOG(ERROR) << TAG(log_tag) << arg_name << " has length " << v.size() << ", expected "
                   << expected << " (or empty)." << std::endl;
        throw std::invalid_argument(std::string(arg_name) + " has the wrong length.");
    }
}
}  // namespace

fft::fft(int radix2_exp) : fft_obj_(nullptr), length_(0) {
    if (fftObj_new(&fft_obj_, radix2_exp) != 0) {
        SPEECH_LOG(ERROR) << TAG(log_tag) << "Failed to create FFT object (radix2_exp=" << radix2_exp
                   << "; must be in [1, 30])." << std::endl;
        throw std::invalid_argument("Invalid radix2_exp for FFT (must be in [1, 30]).");
    }
    length_ = fftObj_getFFTLength(fft_obj_);
    if (rfft_impl::real_frame_fft::supports_large(length_)) {
        const std::vector<float> rectangular(static_cast<std::size_t>(length_), 1.0f);  // copied by the engine
        rfft_ = std::make_unique<rfft_impl::real_frame_fft>(length_, rectangular.data());
    }
    SPEECH_LOG(DEBUG) << TAG(log_tag) << "Created FFT: radix2_exp=" << radix2_exp << ", length=" << length_
               << std::endl;
}

fft::~fft() {
    if (fft_obj_) {
        fftObj_free(fft_obj_);
        fft_obj_ = nullptr;
    }
}

std::pair<std::vector<float>, std::vector<float>> fft::forward(const std::vector<float>& real,
                                                                 const std::vector<float>& imag) {
    check_length(real, length_, "real");
    check_length(imag, length_, "imag");

    std::vector<float> out_real(length_, 0.0f);
    std::vector<float> out_imag(length_, 0.0f);
    forward_into(real.empty() ? nullptr : real.data(), imag.empty() ? nullptr : imag.data(),
                out_real.data(), out_imag.data());
    return {std::move(out_real), std::move(out_imag)};
}

std::pair<std::vector<float>, std::vector<float>> fft::inverse(const std::vector<float>& real,
                                                                 const std::vector<float>& imag) {
    if (real.empty() || imag.empty()) {
        SPEECH_LOG(ERROR) << TAG(log_tag) << "inverse() requires non-empty real and imag inputs."
                   << std::endl;
        throw std::invalid_argument("inverse() requires non-empty real and imag inputs.");
    }
    check_length(real, length_, "real");
    check_length(imag, length_, "imag");

    std::vector<float> out_real(length_, 0.0f);
    std::vector<float> out_imag(length_, 0.0f);
    inverse_into(real.data(), imag.data(), out_real.data(), out_imag.data());
    return {std::move(out_real), std::move(out_imag)};
}

std::vector<float> fft::dct(const std::vector<float>& data, bool is_norm) {
    if (static_cast<int>(data.size()) != length_) {
        SPEECH_LOG(ERROR) << TAG(log_tag) << "dct() input has length " << data.size() << ", expected "
                   << length_ << "." << std::endl;
        throw std::invalid_argument("dct() input has the wrong length.");
    }

    std::vector<float> output(length_, 0.0f);
    dct_into(data.data(), output.data(), is_norm);
    return output;
}

std::vector<float> fft::idct(const std::vector<float>& data, bool is_norm) {
    if (static_cast<int>(data.size()) != length_) {
        SPEECH_LOG(ERROR) << TAG(log_tag) << "idct() input has length " << data.size() << ", expected "
                   << length_ << "." << std::endl;
        throw std::invalid_argument("idct() input has the wrong length.");
    }

    std::vector<float> output(length_, 0.0f);
    idct_into(data.data(), output.data(), is_norm);
    return output;
}

namespace {
void require_buffers(const void* a, const void* b, const char* what) {
    if (a == nullptr || b == nullptr) {
        SPEECH_LOG(ERROR) << TAG(log_tag) << what << " needs non-null buffers." << std::endl;
        throw std::invalid_argument(std::string(what) + " needs non-null buffers.");
    }
}
}  // namespace

void fft::forward_into(const float* real, const float* imag, float* out_real, float* out_imag) {
    require_buffers(out_real, out_imag, "forward_into()");
    if (rfft_ && real != nullptr && imag == nullptr) {
        // Real input: N/2+1 unique bins from one N/2-point complex FFT, then the conjugate mirror
        // X[N-k] = conj(X[k]) for the rest.
        rfft_->spectrum(real, out_real, out_imag);
        const int N = length_;
        for (int k = 1; k < N / 2; ++k) {
            out_real[N - k] = out_real[k];
            out_imag[N - k] = -out_imag[k];
        }
        return;
    }
    // AudioFlux reads its inputs only (the const_cast satisfies its non-const C signature).
    fftObj_fft(fft_obj_, const_cast<float*>(real), const_cast<float*>(imag), out_real, out_imag);
}

void fft::inverse_into(const float* real, const float* imag, float* out_real, float* out_imag) {
    require_buffers(real, imag, "inverse_into()");
    require_buffers(out_real, out_imag, "inverse_into()");
    fftObj_ifft(fft_obj_, const_cast<float*>(real), const_cast<float*>(imag), out_real, out_imag);
}

void fft::dct_into(const float* data, float* out, bool is_norm) {
    require_buffers(data, out, "dct_into()");
    fftObj_dct(fft_obj_, const_cast<float*>(data), out, is_norm ? 1 : 0);
}

void fft::idct_into(const float* data, float* out, bool is_norm) {
    require_buffers(data, out, "idct_into()");
    // AudioFlux's fftObj_idct() mutates data_arr1 (its first argument) in place -- see
    // audioflux_issues.md, Issue 3. It gets a private copy so callers never see their input
    // silently modified.
    detail::uninit_vector<float> input_copy(data, data + length_);
    fftObj_idct(fft_obj_, input_copy.data(), out, is_norm ? 1 : 0);
}

namespace {
void require_size(std::size_t actual, int expected, const char* what, const char* arg, bool allow_empty = false) {
    if ((allow_empty && actual == 0) || actual == static_cast<std::size_t>(expected)) return;
    SPEECH_LOG(ERROR) << TAG(log_tag) << what << ": " << arg << " has " << actual << " samples, expected "
               << expected << "." << std::endl;
    throw std::invalid_argument(std::string(what) + ": " + arg + " has the wrong size.");
}
}  // namespace

void fft::forward_into(span<const float> real, span<const float> imag, span<float> out_real,
                      span<float> out_imag) {
    require_size(real.size(), length_, "forward_into()", "real", true);
    require_size(imag.size(), length_, "forward_into()", "imag", true);
    require_size(out_real.size(), length_, "forward_into()", "out_real");
    require_size(out_imag.size(), length_, "forward_into()", "out_imag");
    forward_into(real.empty() ? nullptr : real.data(), imag.empty() ? nullptr : imag.data(),
                out_real.data(), out_imag.data());
}

void fft::inverse_into(span<const float> real, span<const float> imag, span<float> out_real,
                      span<float> out_imag) {
    require_size(real.size(), length_, "inverse_into()", "real");
    require_size(imag.size(), length_, "inverse_into()", "imag");
    require_size(out_real.size(), length_, "inverse_into()", "out_real");
    require_size(out_imag.size(), length_, "inverse_into()", "out_imag");
    inverse_into(real.data(), imag.data(), out_real.data(), out_imag.data());
}

void fft::dct_into(span<const float> data, span<float> out, bool is_norm) {
    require_size(data.size(), length_, "dct_into()", "data");
    require_size(out.size(), length_, "dct_into()", "out");
    dct_into(data.data(), out.data(), is_norm);
}

void fft::idct_into(span<const float> data, span<float> out, bool is_norm) {
    require_size(data.size(), length_, "idct_into()", "data");
    require_size(out.size(), length_, "idct_into()", "out");
    idct_into(data.data(), out.data(), is_norm);
}

void fft::forward_into(span<const float> real, span<std::complex<float>> out) {
    require_size(real.size(), length_, "forward_into()", "real");
    require_size(out.size(), length_, "forward_into()", "out");
    if (rfft_) {
        // N/2+1 unique bins straight into the interleaved output, then the conjugate mirror.
        rfft_->spectrum_interleaved(real.data(), reinterpret_cast<float*>(out.data()));
        for (int k = 1; k < length_ / 2; ++k) {
            out[static_cast<std::size_t>(length_ - k)] = std::conj(out[static_cast<std::size_t>(k)]);
        }
        return;
    }
    // Lengths below the real-FFT engine's range (< 16 samples): tiny, so a planar detour is fine.
    std::vector<float> re(static_cast<std::size_t>(length_)), im(re.size());
    forward_into(real.data(), nullptr, re.data(), im.data());
    for (std::size_t i = 0; i < re.size(); ++i) out[i] = std::complex<float>(re[i], im[i]);
}

void fft::complex_via_planes(bool inverse, span<const std::complex<float>> in, span<std::complex<float>> out) {
    require_size(in.size(), length_, inverse ? "inverse_into()" : "forward_into()", "in");
    require_size(out.size(), length_, inverse ? "inverse_into()" : "forward_into()", "out");
    const std::size_t n = static_cast<std::size_t>(length_);
    scratch_.resize(4 * n);  // grows once; later calls reuse it
    float* in_re = scratch_.data();
    float* in_im = in_re + n;
    float* out_re = in_im + n;
    float* out_im = out_re + n;
    for (std::size_t i = 0; i < n; ++i) {  // deinterleave first, so in == out is fine
        in_re[i] = in[i].real();
        in_im[i] = in[i].imag();
    }
    if (inverse) {
        fftObj_ifft(fft_obj_, in_re, in_im, out_re, out_im);
    } else {
        fftObj_fft(fft_obj_, in_re, in_im, out_re, out_im);
    }
    for (std::size_t i = 0; i < n; ++i) out[i] = std::complex<float>(out_re[i], out_im[i]);
}

void fft::forward_into(span<const std::complex<float>> in, span<std::complex<float>> out) {
    complex_via_planes(false, in, out);
}

void fft::inverse_into(span<const std::complex<float>> in, span<std::complex<float>> out) {
    complex_via_planes(true, in, out);
}

}  // namespace speech::dsp
