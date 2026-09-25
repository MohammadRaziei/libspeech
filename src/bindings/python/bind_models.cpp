#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <memory>
#include <string>

#include "libspeech/models/denoiser.h"
#include "libspeech/models/silero_vad.h"

namespace nb = nanobind;

NB_MODULE(NB_MODULE_NAME, m) {
    // --- speech::models::Denoiser ---------------------------------------------------------
    // Abstract interface: no public constructor is exposed, matching the
    // C++ API -- the only way to get one is speech::models::Denoiser.create(...).
    nb::class_<speech::models::Denoiser>(m, "Denoiser")
        .def_static("create", &speech::models::Denoiser::Create,
                     "Creates a denoiser backend.",
                     nb::arg("backend"), nb::arg("model_path"), nb::arg("sample_rate") = 16000)
        .def("process", &speech::models::Denoiser::process,
             "Denoises mono audio (a list of [-1, 1]-normalized floats). "
             "Returns denoised audio, the same length as the input.",
             nb::arg("input_audio"));

    // --- speech::models::SileroVadModel -----------------------------------------------------
    nb::class_<speech::models::timestamp_t>(m, "SpeechTimestamp")
        .def(nb::init<int, int, int>(),
             nb::arg("start") = -1, nb::arg("end") = -1, nb::arg("sample_rate") = 16000)
        .def_ro("start", &speech::models::timestamp_t::start, "Start of the speech segment, in samples.")
        .def_ro("end", &speech::models::timestamp_t::end, "End of the speech segment, in samples.")
        .def_prop_ro("start_s", &speech::models::timestamp_t::start_s, "Start of the speech segment, in seconds.")
        .def_prop_ro("end_s", &speech::models::timestamp_t::end_s, "End of the speech segment, in seconds.")
        .def("__repr__", &speech::models::timestamp_t::c_str)
        .def("__eq__", &speech::models::timestamp_t::operator==);

    nb::class_<speech::models::SileroVadModel>(m, "SileroVad")
        .def(nb::init<const std::string&, int, int, float, int, int, int, float>(),
             nb::arg("model_path") = "silero_vad.onnx",
             nb::arg("sample_rate") = 16000,
             nb::arg("window_frame_size") = 32,
             nb::arg("threshold") = 0.5f,
             nb::arg("min_silence_duration_ms") = 100,
             nb::arg("speech_pad_ms") = 30,
             nb::arg("min_speech_duration_ms") = 250,
             nb::arg("max_speech_duration_s") = std::numeric_limits<float>::infinity(),
             "Loads a Silero VAD model. model_path may be a local file path "
             "or a URL (the .onnx weights are downloaded on first use).")
        .def("process", &speech::models::SileroVadModel::processOnVector,
             "Runs voice-activity detection over the full input audio "
             "(mono, [-1, 1]-normalized floats at the model's sample_rate).",
             nb::arg("input_audio"))
        .def("get_speech_timestamps", &speech::models::SileroVadModel::get_speech_timestamps,
             "Returns the detected speech segments (call after process()).")
        .def("reset", &speech::models::SileroVadModel::reset,
             "Resets internal state, so the same model instance can be "
             "reused on a new, unrelated piece of audio.");
}
