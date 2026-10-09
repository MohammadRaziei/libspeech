#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "libspeech/models/denoiser.h"
#include "libspeech/models/silero_vad.h"
#include "ndarray_util.h"

namespace nb = nanobind;

NB_MODULE(NB_MODULE_NAME, m) {
    // --- speech::models::Denoiser ---------------------------------------------------------
    // Abstract interface: no public constructor is exposed, matching the
    // C++ API -- the only way to get one is Denoiser.create(...).
    //
    // `create` hands Python a unique_ptr<Denoiser>. That is only sound because every backend
    // derives from Denoiser alone (single inheritance), so the Denoiser* is the start of the
    // allocation: nanobind runs ~T() and then operator delete() on exactly that pointer. With the
    // old (ONNXModel, Denoiser) multiple inheritance it sat at offset 200 and freeing it aborted
    // with "free(): invalid pointer". test_denoiser_close_and_destroy guards this invariant.
    nb::class_<speech::models::Denoiser>(m, "Denoiser")
        .def_static("create",
                    &speech::models::Denoiser::Create,
                    "Creates a denoiser backend ('facebook' or 'speechbrain'). model_path may be a "
                    "local file or a URL (weights are downloaded on first use). num_threads is the "
                    "ONNX Runtime intra-op thread count: 1 (default), or 0 for one per hardware thread.",
                    nb::arg("backend"), nb::arg("model_path"), nb::arg("sample_rate") = 16000,
                    nb::arg("num_threads") = 1)
        // NumPy first so arrays take the zero-copy path; float64 arrays are converted to float32.
        // The GIL is released while the model runs, so other Python threads keep going.
        .def("process",
             [](speech::models::Denoiser& self, speech::py::InArray1D input_audio) {
                 std::vector<float> out;
                 {
                     nb::gil_scoped_release release;
                     out = self.process(input_audio.data(), input_audio.size());
                 }
                 auto* owner = new std::vector<float>(std::move(out));
                 nb::capsule keep(owner, [](void* p) noexcept { delete static_cast<std::vector<float>*>(p); });
                 return speech::py::OutArray1D(owner->data(), {owner->size()}, keep);
             },
             "Denoises mono audio given as a 1-D NumPy array ([-1, 1]-normalized). Returns a "
             "float32 NumPy array of the same length. Raises RuntimeError if the denoiser is closed.",
             nb::arg("input_audio"))
        .def("process",
             [](speech::models::Denoiser& self, const std::vector<float>& input_audio) {
                 nb::gil_scoped_release release;
                 return self.process(input_audio);
             },
             "Denoises mono audio (a list of [-1, 1]-normalized floats). "
             "Returns denoised audio, the same length as the input.",
             nb::arg("input_audio"))
        .def("close", &speech::models::Denoiser::close,
             "Releases the model and its memory right away. Safe to call more than once; "
             "process() raises RuntimeError afterwards.")
        .def_prop_ro("closed", &speech::models::Denoiser::closed, "True once close() was called.")
        .def_prop_ro("sample_rate", [](const speech::models::Denoiser& d) { return d.sample_rate; },
                     "Sample rate (Hz) the model expects its input at.")
        .def("__enter__", [](nb::handle self) { return self; })
        .def("__exit__", [](speech::models::Denoiser& self, nb::args) { self.close(); });

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
             nb::arg("model_path") = "silero-vad.onnx",
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
