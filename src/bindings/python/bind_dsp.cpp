#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "libspeech/dsp/dct.h"
#include "libspeech/dsp/fft.h"
#include "libspeech/dsp/mfcc.h"
#include "libspeech/dsp/resample.h"
#include "libspeech/dsp/simd.h"
#include "libspeech/dsp/stft.h"
#include "libspeech/dsp/window.h"
#include <algorithm>
#include <cctype>
#include <complex>
#include <stdexcept>
#include <string>
#include <variant>

#include "ndarray_util.h"

namespace nb = nanobind;

namespace {
// istft's overlap-add flavour: 0 / "weighted" (default) or 1 / "plain".
using istft_method = std::variant<int, std::string>;
int parse_istft_method(const istft_method& method) {
    if (const int* number = std::get_if<int>(&method)) return *number;  // validated by istft() itself
    std::string name = std::get<std::string>(method);
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (name == "weighted") return 0;
    if (name == "plain") return 1;
    throw std::invalid_argument("method must be 'weighted' or 'plain' (or 0 / 1), got '" + name + "'");
}
}  // namespace

NB_MODULE(NB_MODULE_NAME, m) {
    m.def("simd_backend", &speech::dsp::simd::backend_name,
          "Name of the SIMD backend the DSP kernels use right now: 'avx2+fma', 'neon' or "
          "'generic'. Chosen at run time from the CPU; LIBSPEECH_SIMD=off forces 'generic'.");
    m.def("set_simd_enabled", &speech::dsp::simd::set_enabled, nb::arg("enabled"),
          "Run-time switch for the vectorized kernels (default on). Disabling selects the "
          "portable kernels for objects constructed / calls made afterwards.");
    m.def("simd_enabled", &speech::dsp::simd::is_enabled);

    // --- Window ------------------------------------------------------------
    nb::enum_<speech::dsp::window_type>(m, "WindowType")
        .value("rect", speech::dsp::window_type::rect)
        .value("hann", speech::dsp::window_type::hann)
        .value("hamming", speech::dsp::window_type::hamming)
        .value("blackman", speech::dsp::window_type::blackman)
        .value("kaiser", speech::dsp::window_type::kaiser)
        .value("bartlett", speech::dsp::window_type::bartlett)
        .value("triang", speech::dsp::window_type::triang)
        .value("flattop", speech::dsp::window_type::flattop)
        .value("gauss", speech::dsp::window_type::gauss);

    m.def("window", &speech::dsp::window::generate,
          "Generates a periodic analysis window (for use before an FFT/STFT).",
          nb::arg("type"), nb::arg("length"));

    // --- dct_ii ---------------------------------------------------------------
    m.def("dct", &speech::dsp::dct_ii,
          "Direct DCT-II of a signal of any length (see FFT.dct for the "
          "faster power-of-2-only variant).",
          nb::arg("input"), nb::arg("num_outputs") = -1, nb::arg("orthonormal") = true);

    // --- Resample ------------------------------------------------------------
    nb::class_<speech::dsp::resample>(m, "Resample")
        .def(nb::init<int, int>(), nb::arg("source_rate"), nb::arg("target_rate"),
             "Amplitude-preserving resampler between two sample rates (quality \"best\").")
        .def("__init__",
             [](speech::dsp::resample* self, int source_rate, int target_rate, const std::string& quality) {
                 using q = speech::dsp::resample_quality;
                 q level;
                 if (quality == "best") level = q::best;
                 else if (quality == "high") level = q::high;
                 else if (quality == "medium") level = q::medium;
                 else if (quality == "fast") level = q::fast;
                 else throw nb::value_error("quality must be \"best\", \"high\", \"medium\" or \"fast\"");
                 new (self) speech::dsp::resample(source_rate, target_rate, level);
             },
             nb::arg("source_rate"), nb::arg("target_rate"), nb::arg("quality"),
             "Resampler with a speed/accuracy preset: \"best\" (default), \"high\", \"medium\", \"fast\".")
        // NumPy fast path: float32 C-contiguous arrays are read in place (other
        // dtypes/layouts are converted once by nanobind) and the result is a
        // NumPy array that owns the C++ buffer -- no list conversion, no copies.
        .def("resample",
             [](speech::dsp::resample& self, speech::py::in_array_1d x) {
                 speech::detail::uninit_vector<float> out;
                 {
                     nb::gil_scoped_release release;
                     out = self.resample_flat(x.data(), x.size());
                 }
                 const std::size_t n = out.size();
                 return speech::py::wrap<speech::py::out_array_1d>(std::move(out), {n});
             },
             nb::arg("input_data"),
             "Resamples a float32 1-D array; returns a float32 NumPy array.")
        .def("resample", &speech::dsp::resample::process, nb::arg("input_data"),
             "Resamples a list of floats; returns a list of floats.");

    // --- FFT -------------------------------------------------------------
    // The NumPy overloads come first so float32 (or convertible) arrays take the zero-copy path;
    // lists fall through to the original overloads. The GIL is kept: one FFT object holds
    // transform state and must not run on two threads at once.
    auto check_len = [](const speech::py::in_array_1d& a, int expected, const char* name) {
        if (a.size() != static_cast<std::size_t>(expected)) {
            throw std::invalid_argument(std::string(name) + " has length " + std::to_string(a.size()) +
                                        ", expected " + std::to_string(expected) + ".");
        }
    };
    nb::class_<speech::dsp::fft>(m, "FFT")
        .def(nb::init<int>(), nb::arg("radix2_exp"),
             "Power-of-2 FFT of length 2**radix2_exp.")
        .def_prop_ro("size", &speech::dsp::fft::size)
        .def("forward",
             [check_len](speech::dsp::fft& self, speech::py::in_array_1d real, speech::py::in_array_1d imag) {
                 check_len(real, self.size(), "real");
                 if (imag.is_valid()) check_len(imag, self.size(), "imag");
                 const std::size_t n = static_cast<std::size_t>(self.size());
                 speech::py::float_vec re(n), im(n);
                 self.forward_into(real.data(), imag.is_valid() ? imag.data() : nullptr, re.data(), im.data());
                 auto out_re = speech::py::wrap<speech::py::out_array_1d>(std::move(re), {n});
                 auto out_im = speech::py::wrap<speech::py::out_array_1d>(std::move(im), {n});
                 return nb::make_tuple(out_re, out_im);
             },
             nb::arg("real"), nb::arg("imag").none() = nb::none(),
             "Forward FFT of a float32 array of length `size` (imag optional). Returns (real, imag) "
             "float32 arrays of length `size`. A real-only input of 16..131072 samples uses a "
             "real-input FFT (half the work).")
        .def("forward", &speech::dsp::fft::forward,
             nb::arg("real"), nb::arg("imag") = std::vector<float>{},
             "Forward FFT. Returns (real, imag).")
        .def("inverse",
             [check_len](speech::dsp::fft& self, speech::py::in_array_1d real, speech::py::in_array_1d imag) {
                 check_len(real, self.size(), "real");
                 check_len(imag, self.size(), "imag");
                 const std::size_t n = static_cast<std::size_t>(self.size());
                 speech::py::float_vec re(n), im(n);
                 self.inverse_into(real.data(), imag.data(), re.data(), im.data());
                 auto out_re = speech::py::wrap<speech::py::out_array_1d>(std::move(re), {n});
                 auto out_im = speech::py::wrap<speech::py::out_array_1d>(std::move(im), {n});
                 return nb::make_tuple(out_re, out_im);
             },
             nb::arg("real"), nb::arg("imag"),
             "Inverse FFT of float32 arrays of length `size`. Returns (real, imag) float32 arrays.")
        .def("inverse", &speech::dsp::fft::inverse, nb::arg("real"), nb::arg("imag"),
             "Inverse FFT. Returns (real, imag).")
        // complex64 in/out (what np.fft.fft returns): written straight into the result array.
        .def("fft",
             [](speech::dsp::fft& self, speech::py::in_cx_array_1d z) {
                 if (z.size() != static_cast<std::size_t>(self.size())) {
                     throw std::invalid_argument("fft(): input has length " + std::to_string(z.size()) +
                                                 ", expected " + std::to_string(self.size()) + ".");
                 }
                 const std::size_t n = z.size();
                 speech::py::cx_vec out(n);
                 self.forward_into(speech::span<const speech::py::cx>(z.data(), n),
                                  speech::span<speech::py::cx>(out.data(), n));
                 return speech::py::wrap_complex<speech::py::out_cx_array_1d>(std::move(out), {n});
             },
             nb::arg("z"),
             "FFT of a complex64 array of length `size`; returns a complex64 array.")
        .def("fft",
             [check_len](speech::dsp::fft& self, speech::py::in_array_1d x) {
                 check_len(x, self.size(), "x");
                 const std::size_t n = x.size();
                 speech::py::cx_vec out(n);
                 self.forward_into(speech::span<const float>(x.data(), n),
                                  speech::span<speech::py::cx>(out.data(), n));
                 return speech::py::wrap_complex<speech::py::out_cx_array_1d>(std::move(out), {n});
             },
             nb::arg("x"),
             "FFT of a real float32 array of length `size`; returns the full spectrum as a complex64 "
             "array (a real-input FFT does the work, written in place).")
        .def("ifft",
             [](speech::dsp::fft& self, speech::py::in_cx_array_1d z) {
                 if (z.size() != static_cast<std::size_t>(self.size())) {
                     throw std::invalid_argument("ifft(): input has length " + std::to_string(z.size()) +
                                                 ", expected " + std::to_string(self.size()) + ".");
                 }
                 const std::size_t n = z.size();
                 speech::py::cx_vec out(n);
                 self.inverse_into(speech::span<const speech::py::cx>(z.data(), n),
                                  speech::span<speech::py::cx>(out.data(), n));
                 return speech::py::wrap_complex<speech::py::out_cx_array_1d>(std::move(out), {n});
             },
             nb::arg("z"),
             "Inverse FFT of a complex64 array of length `size`; returns a complex64 array.")
        .def("dct",
             [check_len](speech::dsp::fft& self, speech::py::in_array_1d data, bool is_norm) {
                 check_len(data, self.size(), "data");
                 const std::size_t n = static_cast<std::size_t>(self.size());
                 speech::py::float_vec out(n);
                 self.dct_into(data.data(), out.data(), is_norm);
                 return speech::py::wrap<speech::py::out_array_1d>(std::move(out), {n});
             },
             nb::arg("data"), nb::arg("is_norm") = true,
             "DCT-II of a float32 array of length `size`; returns a float32 array.")
        .def("dct", &speech::dsp::fft::dct, nb::arg("data"), nb::arg("is_norm") = true)
        .def("idct",
             [check_len](speech::dsp::fft& self, speech::py::in_array_1d data, bool is_norm) {
                 check_len(data, self.size(), "data");
                 const std::size_t n = static_cast<std::size_t>(self.size());
                 speech::py::float_vec out(n);
                 self.idct_into(data.data(), out.data(), is_norm);
                 return speech::py::wrap<speech::py::out_array_1d>(std::move(out), {n});
             },
             nb::arg("data"), nb::arg("is_norm") = true,
             "Inverse of dct() for a float32 array of length `size`; the input is not modified.")
        .def("idct", &speech::dsp::fft::idct, nb::arg("data"), nb::arg("is_norm") = true);

    // --- STFT ------------------------------------------------------------
    nb::class_<speech::dsp::stft>(m, "STFT")
        .def("__init__",
             [](speech::dsp::stft* self, int radix2_exp, speech::dsp::window_type type, int slide_length) {
                 if (radix2_exp > 30) {
                     throw std::invalid_argument(
                         "radix2_exp is the EXPONENT of the frame length (length = 2 ** radix2_exp), got " +
                         std::to_string(radix2_exp) + "; to give the length in samples use STFT(n_fft=" +
                         std::to_string(radix2_exp) + ").");
                 }
                 new (self) speech::dsp::stft(radix2_exp, type, slide_length);
             },
             nb::arg("radix2_exp"), nb::arg("window_type") = speech::dsp::window_type::hann, nb::arg("slide_length") = 0,
             "Short-time Fourier transform: frames + windows + FFTs a signal. The frame length is "
             "2 ** radix2_exp samples; slide_length is the hop (0 = a quarter frame). "
             "STFT(n_fft=...) takes the length in samples instead.")
        .def("__init__",
             [](speech::dsp::stft* self, int n_fft, speech::dsp::window_type type, int hop_length) {
                 if (n_fft < 2 || (n_fft & (n_fft - 1)) != 0) {
                     throw std::invalid_argument("n_fft must be a power of two, got " + std::to_string(n_fft) + ".");
                 }
                 int radix2_exp = 0;
                 while ((1 << radix2_exp) < n_fft) ++radix2_exp;
                 new (self) speech::dsp::stft(radix2_exp, type, hop_length);
             },
             nb::kw_only(), nb::arg("n_fft"), nb::arg("window_type") = speech::dsp::window_type::hann,
             nb::arg("hop_length") = 0,
             "STFT with the frame length given in samples (a power of two), e.g. STFT(n_fft=512, "
             "hop_length=128); hop_length 0 means a quarter frame.")
        .def_prop_ro("fft_length", &speech::dsp::stft::fft_length)
        .def_prop_ro("slide_length", &speech::dsp::stft::slide_length)
        .def("num_bins", &speech::dsp::stft::num_bins, nb::arg("onesided") = false,
             "Columns per frame: fft_length, or fft_length // 2 + 1 if onesided.")
        .def("cal_time_length", &speech::dsp::stft::cal_time_length, nb::arg("data_length"))
        .def("cal_data_length", &speech::dsp::stft::cal_data_length, nb::arg("time_length"))
        .def("stft",
             [](speech::dsp::stft& self, speech::py::in_array_1d x, bool onesided) {
                 speech::dsp::complex_spectrogram sp;
                 {
                     nb::gil_scoped_release release;
                     sp = self.spectrogram(x.data(), x.size(), onesided);
                 }
                 const std::size_t frames = static_cast<std::size_t>(sp.num_frames);
                 const std::size_t bins = static_cast<std::size_t>(sp.num_bins);
                 auto re = speech::py::wrap<speech::py::out_array_2d>(std::move(sp.real), {frames, bins});
                 auto im = speech::py::wrap<speech::py::out_array_2d>(std::move(sp.imag), {frames, bins});
                 return nb::make_tuple(re, im);
             },
             nb::arg("data"), nb::arg("onesided") = false,
             "Float32 1-D array in; returns (real, imag) float32 NumPy arrays of shape "
             "(num_frames, fft_length), or (num_frames, fft_length // 2 + 1) with "
             "onesided=True (the non-redundant half of a real signal's spectrum; half the "
             "work and half the memory).")
        .def("stft", &speech::dsp::stft::compute, nb::arg("data"),
             "Returns (real, imag), each a [num_frames][fft_length] matrix.")
        // The spectrogram as ONE complex64 array (frames, bins) -- what librosa.stft returns,
        // transposed -- written straight into the result by the kernel (no split planes, no copy).
        .def("stft_complex",
             [](speech::dsp::stft& self, speech::py::in_array_1d data, bool onesided) {
                 const int frames = std::max(0, self.cal_time_length(static_cast<int>(data.size())));
                 const std::size_t bins = static_cast<std::size_t>(self.num_bins(onesided));
                 const std::size_t rows = static_cast<std::size_t>(frames);
                 speech::py::cx_vec out(rows * bins);
                 {
                     nb::gil_scoped_release release;
                     speech::frame_view<speech::py::cx> view(out.data(), rows, bins);
                     if (onesided) {
                         self.stft_onesided_into(speech::span<const float>(data.data(), data.size()), view);
                     } else {
                         self.stft_into(speech::span<const float>(data.data(), data.size()), view);
                     }
                 }
                 return speech::py::wrap_complex<speech::py::out_cx_array_2d>(std::move(out), {rows, bins});
             },
             nb::arg("data"), nb::arg("onesided") = false,
             "STFT of a float32 array as a complex64 array of shape (num_frames, bins): fft_length bins "
             "(the full spectrum) or fft_length // 2 + 1 with onesided=True.")
        .def("istft",
             [](speech::dsp::stft& self, speech::py::in_cx_array_2d spec, const istft_method& method_type) {
                 const std::size_t frames = spec.shape(0);
                 const std::size_t bins = spec.shape(1);
                 speech::frame_view<const speech::py::cx> view(spec.data(), frames, bins);
                 speech::py::float_vec out;
                 {
                     nb::gil_scoped_release release;
                     out = self.istft(view, parse_istft_method(method_type));
                 }
                 const std::size_t n = out.size();
                 return speech::py::wrap<speech::py::out_array_1d>(std::move(out), {n});
             },
             nb::arg("spec"), nb::arg("method_type") = 0,
             "Inverse STFT of a complex64 (num_frames, bins) array, e.g. from stft_complex(); bins is "
             "fft_length or fft_length // 2 + 1. method is 'weighted' (default; also 0) or 'plain' (1). "
             "Returns a float32 1-D array.")
        // NumPy first so arrays take the zero-copy path (float64 arrays are converted to float32).
        .def("istft",
             [](speech::dsp::stft& self, speech::py::in_array_2d real, speech::py::in_array_2d imag,
                const istft_method& method_type) {
                 if (real.shape(0) != imag.shape(0) || real.shape(1) != imag.shape(1)) {
                     throw std::invalid_argument("istft(): real and imag must have the same shape.");
                 }
                 // The views carry the (frames, bins) shape; the STFT infers full vs one-sided from
                 // the row length and rejects anything else with std::invalid_argument (-> ValueError).
                 const std::size_t frames = real.shape(0);
                 const std::size_t bins = real.shape(1);
                 speech::frame_view<const float> re_frames(real.data(), frames, bins);
                 speech::frame_view<const float> im_frames(imag.data(), frames, bins);
                 speech::py::float_vec out;
                 {
                     nb::gil_scoped_release release;
                     out = self.istft(re_frames, im_frames, parse_istft_method(method_type));
                 }
                 const std::size_t n = out.size();
                 return speech::py::wrap<speech::py::out_array_1d>(std::move(out), {n});
             },
             nb::arg("real"), nb::arg("imag"), nb::arg("method_type") = 0,
             "Inverse STFT of float32 (num_frames, bins) NumPy arrays, where bins is fft_length (the "
             "full spectrum, as stft() returns) or fft_length // 2 + 1 (stft(onesided=True)). Returns "
             "a float32 1-D array of cal_data_length(num_frames) samples. method_type: 'weighted' or 0 = weighted "
             "overlap-add (default), 1 = plain overlap-add; anything else raises ValueError. Only the "
             "real part of the inverse transform is returned, as in AudioFlux.")
        .def("istft",
             [](speech::dsp::stft& self, const std::vector<std::vector<float>>& real,
                const std::vector<std::vector<float>>& imag, int method_type) {
                 return self.istft(real, imag, method_type);
             },
             nb::arg("real"), nb::arg("imag"), nb::arg("method_type") = 0,
             "Inverse STFT of [num_frames][fft_length] lists (slower than the NumPy form).");

    // --- MFCC ------------------------------------------------------------
    nb::class_<speech::dsp::mfcc::params>(m, "MFCCParams")
        .def(nb::init<>())
        .def_rw("sample_rate", &speech::dsp::mfcc::params::sample_rate)
        .def_rw("num_mel_filters", &speech::dsp::mfcc::params::num_mel_filters)
        .def_rw("num_coefficients", &speech::dsp::mfcc::params::num_coefficients)
        .def_rw("low_freq_hz", &speech::dsp::mfcc::params::low_freq_hz)
        .def_rw("high_freq_hz", &speech::dsp::mfcc::params::high_freq_hz)
        .def_rw("radix2_exp", &speech::dsp::mfcc::params::radix2_exp)
        .def_rw("slide_length", &speech::dsp::mfcc::params::slide_length);

    nb::class_<speech::dsp::mfcc>(m, "MFCC")
        .def(nb::init<speech::dsp::mfcc::params>(), nb::arg("params"))
        .def("compute",
             [](speech::dsp::mfcc& self, speech::py::in_array_1d x) {
                 speech::dsp::mfcc_matrix m;
                 {
                     nb::gil_scoped_release release;
                     m = self.compute_flat(x.data(), x.size());
                 }
                 const std::size_t frames = static_cast<std::size_t>(m.num_frames);
                 const std::size_t coefs = static_cast<std::size_t>(m.num_coefficients);
                 return speech::py::wrap<speech::py::out_array_2d>(std::move(m.data), {frames, coefs});
             },
             nb::arg("signal"),
             "Float32 1-D array in; returns a float32 NumPy array of shape "
             "(num_frames, num_coefficients).")
        .def("compute", &speech::dsp::mfcc::compute, nb::arg("signal"),
             "Returns a [num_frames][num_coefficients] matrix of MFCCs.");
}
