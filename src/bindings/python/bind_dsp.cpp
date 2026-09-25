#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>

#include "libspeech/dsp/dct.h"
#include "libspeech/dsp/fft.h"
#include "libspeech/dsp/mfcc.h"
#include "libspeech/dsp/resample.h"
#include "libspeech/dsp/stft.h"
#include "libspeech/dsp/window.h"

namespace nb = nanobind;

NB_MODULE(NB_MODULE_NAME, m) {
    // --- Window ------------------------------------------------------------
    nb::enum_<WindowType>(m, "WindowType")
        .value("rect", Window_Rect)
        .value("hann", Window_Hann)
        .value("hamming", Window_Hamm)
        .value("blackman", Window_Blackman)
        .value("kaiser", Window_Kaiser)
        .value("bartlett", Window_Bartlett)
        .value("triang", Window_Triang)
        .value("flattop", Window_Flattop)
        .value("gauss", Window_Gauss);

    m.def("window", &speech::dsp::window::generate,
          "Generates a periodic analysis window (for use before an FFT/STFT).",
          nb::arg("type"), nb::arg("length"));

    // --- dctII ---------------------------------------------------------------
    m.def("dct", &speech::dsp::dctII,
          "Direct DCT-II of a signal of any length (see FFT.dct for the "
          "faster power-of-2-only variant).",
          nb::arg("input"), nb::arg("num_outputs") = -1, nb::arg("orthonormal") = true);

    // --- Resample ------------------------------------------------------------
    nb::class_<speech::dsp::Resample>(m, "Resample")
        .def(nb::init<int, int>(), nb::arg("source_rate"), nb::arg("target_rate"),
             "Amplitude-preserving resampler between two sample rates.")
        .def("resample", &speech::dsp::Resample::resample, nb::arg("input_data"));

    // --- FFT -------------------------------------------------------------
    nb::class_<speech::dsp::FFT>(m, "FFT")
        .def(nb::init<int>(), nb::arg("radix2_exp"),
             "Power-of-2 FFT of length 2**radix2_exp.")
        .def_prop_ro("size", &speech::dsp::FFT::size)
        .def("forward", &speech::dsp::FFT::forward,
             nb::arg("real"), nb::arg("imag") = std::vector<float>{},
             "Forward FFT. Returns (real, imag).")
        .def("inverse", &speech::dsp::FFT::inverse, nb::arg("real"), nb::arg("imag"),
             "Inverse FFT. Returns (real, imag).")
        .def("dct", &speech::dsp::FFT::dct, nb::arg("data"), nb::arg("is_norm") = true)
        .def("idct", &speech::dsp::FFT::idct, nb::arg("data"), nb::arg("is_norm") = true);

    // --- STFT ------------------------------------------------------------
    nb::class_<speech::dsp::STFT>(m, "STFT")
        .def(nb::init<int, WindowType, int>(),
             nb::arg("radix2_exp"), nb::arg("window_type") = Window_Hann, nb::arg("slide_length") = 0,
             "Short-time Fourier transform: frames + windows + FFTs a signal.")
        .def_prop_ro("fft_length", &speech::dsp::STFT::fftLength)
        .def_prop_ro("slide_length", &speech::dsp::STFT::slideLength)
        .def("cal_time_length", &speech::dsp::STFT::calTimeLength, nb::arg("data_length"))
        .def("cal_data_length", &speech::dsp::STFT::calDataLength, nb::arg("time_length"))
        .def("stft", &speech::dsp::STFT::stft, nb::arg("data"),
             "Returns (real, imag), each a [num_frames][fft_length] matrix.")
        .def("istft", &speech::dsp::STFT::istft,
             nb::arg("real"), nb::arg("imag"), nb::arg("method_type") = 0);

    // --- MFCC ------------------------------------------------------------
    nb::class_<speech::dsp::MFCC::Params>(m, "MFCCParams")
        .def(nb::init<>())
        .def_rw("sample_rate", &speech::dsp::MFCC::Params::sampleRate)
        .def_rw("num_mel_filters", &speech::dsp::MFCC::Params::numMelFilters)
        .def_rw("num_coefficients", &speech::dsp::MFCC::Params::numCoefficients)
        .def_rw("low_freq_hz", &speech::dsp::MFCC::Params::lowFreqHz)
        .def_rw("high_freq_hz", &speech::dsp::MFCC::Params::highFreqHz)
        .def_rw("radix2_exp", &speech::dsp::MFCC::Params::radix2Exp)
        .def_rw("slide_length", &speech::dsp::MFCC::Params::slideLength);

    nb::class_<speech::dsp::MFCC>(m, "MFCC")
        .def(nb::init<speech::dsp::MFCC::Params>(), nb::arg("params"))
        .def("compute", &speech::dsp::MFCC::compute, nb::arg("signal"),
             "Returns a [num_frames][num_coefficients] matrix of MFCCs.");
}
