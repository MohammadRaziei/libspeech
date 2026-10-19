#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/filesystem.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "libspeech/models/denoiser.h"
#include "libspeech/models/denoiser_stream.h"
#include "libspeech/models/dpdfnet_denoiser.h"
#include "libspeech/models/gtcrn_denoiser.h"
#include "libspeech/models/model_files.h"
#include "libspeech/models/providers.h"
#include "libspeech/models/silero_vad.h"
#include "libspeech/utils/log.h"
#include "libspeech/utils/utils.h"
#include "ndarray_util.h"

namespace nb = nanobind;

NB_MODULE(NB_MODULE_NAME, m) {
    // --- logging -----------------------------------------------------------------------------
    m.def("set_log_level",
          [](const std::string& level) { speech::utils::set_log_level(speech::utils::parse_log_level(level)); },
          "Sets how much libspeech writes to stderr: 'trace', 'debug', 'info', 'warning' (the default), "
          "'error' or 'off' (case-insensitive). Before the first call the level can also be set with the "
          "LIBSPEECH_LOG environment variable. Raises ValueError for an unknown level.",
          nb::arg("level"));
    m.def("get_log_level",
          [] { return std::string(speech::utils::log_level_name(speech::utils::get_log_level())); },
          "The current log level, e.g. 'warning'.");

    // --- execution providers -----------------------------------------------------------------
    m.def("available_providers", &speech::models::available_providers,
          "ONNX Runtime execution providers of the loaded build, in priority order, e.g. "
          "['CUDAExecutionProvider', 'CPUExecutionProvider']. 'CPUExecutionProvider' is always there.");
    m.def("is_cuda_available", &speech::models::is_cuda_available,
          "True if models can run on CUDA: the ONNX Runtime build has the CUDA provider and it can "
          "actually be created on this machine. False for a CPU-only build. Never raises.");

    // --- model files: checksums, downloading, readme ------------------------------------------
    m.def("sha256",
          [](nb::bytes data) {
              return speech::utils::sha256_hex(data.c_str(), data.size());
          },
          "SHA-256 of a bytes object as 64 lowercase hex characters.", nb::arg("data"));
    m.def("sha256_file",
          [](const std::filesystem::path& path) {
              nb::gil_scoped_release release;
              return speech::utils::sha256_file(path);
          },
          "SHA-256 of a file (read in chunks, any size) as 64 lowercase hex characters. "
          "Raises RuntimeError if the file cannot be read.",
          nb::arg("path"));
    const auto make_fetch_options = [](bool ignore_cache, std::optional<std::filesystem::path> cache_dir) {
        speech::models::fetch_options options;
        options.ignore_cache = ignore_cache;
        if (cache_dir) options.cache_dir = *cache_dir;
        return options;
    };
    m.def("fetch_model",
          [make_fetch_options](const std::string& model, bool ignore_cache,
                               std::optional<std::filesystem::path> cache_dir) {
              nb::gil_scoped_release release;
              return speech::models::fetch_model(model, make_fetch_options(ignore_cache, cache_dir)).model;
          },
          "Makes sure a model is in the cache (default ~/.libspeech) and returns the path of the .onnx file. "
          "`model` is a bare file name from libspeech's models release or a full http(s) URL. The checksum "
          "file (<model>.sha256) is downloaded first and must be one line holding the SHA-256, then the model "
          "is downloaded and checked against it (a mismatch deletes the download and raises RuntimeError), "
          "then <model>.readme. A cached model is checked on every call. ignore_cache=True downloads "
          "everything again.",
          nb::arg("model"), nb::arg("ignore_cache") = false, nb::arg("cache_dir") = nb::none());
    m.def("model_readme",
          [make_fetch_options](const std::string& model, bool ignore_cache,
                               std::optional<std::filesystem::path> cache_dir) {
              nb::gil_scoped_release release;
              return speech::models::model_readme(model, make_fetch_options(ignore_cache, cache_dir));
          },
          "The readme of a model (what it is, who made it, license, provenance), fetched like fetch_model().",
          nb::arg("model"), nb::arg("ignore_cache") = false, nb::arg("cache_dir") = nb::none());

    // --- speech::models::Denoiser ---------------------------------------------------------
    // The C++ type is an abstract interface; Python gets a constructor that picks the backend
    // (a factory via nb::new_), and the same thing as Denoiser.create(...).
    //
    // `create` hands Python a unique_ptr<Denoiser>. That is only sound because every backend
    // derives from Denoiser alone (single inheritance), so the Denoiser* is the start of the
    // allocation: nanobind runs ~T() and then operator delete() on exactly that pointer. With the
    // old (ONNXModel, Denoiser) multiple inheritance it sat at offset 200 and freeing it aborted
    // with "free(): invalid pointer". test_denoiser_close_and_destroy guards this invariant.
    // process() and __call__ are the same function: denoiser("facebook")(x).
    // NumPy first so arrays take the zero-copy path; float64 arrays are converted to float32.
    // The GIL is released while the model runs, so other Python threads keep going.
    const auto process_array = [](speech::models::denoiser& self, speech::py::in_array_1d input_audio) {
        std::vector<float> out;
        {
            nb::gil_scoped_release release;
            out = self.process(input_audio.data(), input_audio.size());
        }
        auto* owner = new std::vector<float>(std::move(out));
        nb::capsule keep(owner, [](void* p) noexcept { delete static_cast<std::vector<float>*>(p); });
        return speech::py::out_array_1d(owner->data(), {owner->size()}, keep);
    };
    const auto process_list = [](speech::models::denoiser& self, const std::vector<float>& input_audio) {
        nb::gil_scoped_release release;
        return self.process(input_audio);
    };
    const char* const array_doc =
        "Denoises mono audio given as a 1-D NumPy array ([-1, 1]-normalized). Returns a float32 "
        "NumPy array of the same length. Raises RuntimeError if the denoiser is closed.";
    const char* const list_doc =
        "Denoises mono audio (a list of [-1, 1]-normalized floats). Returns denoised audio, the "
        "same length as the input.";

    // --- speech::models::DenoiserStream ---------------------------------------------------------
    // vector -> float32 NumPy array without a copy (the vector lives in a capsule owned by the array).
    const auto to_array = [](std::vector<float>&& v) {
        auto* owner = new std::vector<float>(std::move(v));
        nb::capsule keep(owner, [](void* p) noexcept { delete static_cast<std::vector<float>*>(p); });
        return speech::py::out_array_1d(owner->data(), {owner->size()}, keep);
    };
    nb::class_<speech::models::denoiser_stream>(m, "DenoiserStream")
        .def("feed",
             [to_array](speech::models::denoiser_stream& self, speech::py::in_array_1d chunk) {
                 std::vector<float> out;
                 {
                     nb::gil_scoped_release release;
                     out = self.feed(chunk.data(), chunk.size());
                 }
                 return to_array(std::move(out));
             },
             nb::arg("chunk"),
             "Feeds a chunk of mono audio (1-D NumPy array, [-1, 1]-normalized, any length). Returns a float32 "
             "array with the samples that became final -- usually about as long as the chunk, but delayed by up to "
             "latency_samples, so it can be shorter (even empty) or longer than the chunk. Raises RuntimeError after "
             "finish().")
        .def("feed",
             [to_array](speech::models::denoiser_stream& self, const std::vector<float>& chunk) {
                 std::vector<float> out;
                 {
                     nb::gil_scoped_release release;
                     out = self.feed(chunk);
                 }
                 return to_array(std::move(out));
             },
             nb::arg("chunk"), "Same as feed() for a list of floats.")
        .def("finish",
             [to_array](speech::models::denoiser_stream& self) {
                 std::vector<float> out;
                 {
                     nb::gil_scoped_release release;
                     out = self.finish();
                 }
                 return to_array(std::move(out));
             },
             "Ends the stream and returns the remaining samples, so that the total output is exactly as long as the "
             "total input. Safe to call more than once (later calls return an empty array).")
        .def_prop_ro("samples_in", &speech::models::denoiser_stream::samples_in, "Samples fed so far.")
        .def_prop_ro("samples_out", &speech::models::denoiser_stream::samples_out, "Samples returned so far.")
        .def_prop_ro("latency_samples", &speech::models::denoiser_stream::latency_samples,
                     "At most this many samples are fed but not yet returned (GTCRN: 511 = 32 ms at 16 kHz).")
        .def_prop_ro("finished", &speech::models::denoiser_stream::finished, "True once finish() was called.");

    nb::class_<speech::models::denoiser>(m, "Denoiser")
        // Denoiser("facebook") -- the everyday way; `model` defaults to the backend's standard model
        // (downloaded to ~/.libspeech on first use). Denoiser.create(...) is the same thing spelled
        // as a factory.
        .def(nb::new_([](const std::string& backend, std::optional<std::string> model, int sample_rate,
                         int num_threads, bool ignore_cache) {
                 const std::string default_model = backend == "facebook" ? "facebook-denoiser-dns64.onnx"
                                                   : backend == "speechbrain"
                                                       ? "speechbrain-sepformer-wham16k-enhancement.onnx"
                                                   : backend == "gtcrn"
                                                       ? std::string(speech::models::gtcrn_denoiser::default_model)
                                                   : backend == "dpdfnet"
                                                       ? std::string(speech::models::dpdfnet_denoiser::default_model)
                                                       : std::string();
                 // The constructor may download a model (seconds to minutes): let other Python threads run.
                 nb::gil_scoped_release release;
                 return speech::models::denoiser::create(backend, model ? *model : default_model, sample_rate,
                                                         num_threads, ignore_cache);
             }),
             nb::arg("backend"), nb::arg("model") = nb::none(), nb::arg("sample_rate") = 16000,
             nb::arg("num_threads") = 1, nb::arg("ignore_cache") = false,
             "Loads a denoiser: backend is 'facebook' (DNS64), 'speechbrain' (SepFormer), 'gtcrn' (tiny, "
             "16 kHz only) or 'dpdfnet' (variants dpdfnet_baseline/2/4/8, dpdfnet2_48khz_hr; "
             "sample_rate must match the variant). `model` is "
             "a local .onnx file or a URL; by default the backend's standard model is used (downloaded "
             "on first use). num_threads is the ONNX Runtime intra-op thread count: 1 (default), or 0 "
             "for one per hardware thread. ignore_cache=True downloads the model (and its .sha256 and .readme) again "
             "even if it is cached. Use it as a context manager (`with`) or call close() to "
             "free the model's memory early.")
        .def_static("create",
                    &speech::models::denoiser::create,
                    "Factory form of the constructor: model_path may be a local file or a URL. "
                    "Prefer Denoiser(backend).",
                    nb::arg("backend"), nb::arg("model_path"), nb::arg("sample_rate") = 16000,
                    nb::arg("num_threads") = 1, nb::arg("ignore_cache") = false)
        .def("process", process_array, array_doc, nb::arg("input_audio"))
        .def("process", process_list, list_doc, nb::arg("input_audio"))
        .def("__call__", process_array, array_doc, nb::arg("input_audio"))
        .def("__call__", process_list, list_doc, nb::arg("input_audio"))
        .def_prop_ro("supports_streaming", &speech::models::denoiser::supports_streaming,
                     "True if stream() works for this backend (currently 'gtcrn').")
        .def("stream", &speech::models::denoiser::create_stream,
             "Starts a streaming session (a DenoiserStream): feed audio in chunks of any size and get the enhanced "
             "audio back with a small delay. Every call gives an independent session. Raises RuntimeError if the "
             "backend cannot stream or the denoiser is closed.")
        .def("close", &speech::models::denoiser::close,
             "Releases the model and its memory right away. Safe to call more than once; "
             "process() raises RuntimeError afterwards.")
        .def_prop_ro("closed", &speech::models::denoiser::closed, "True once close() was called.")
        .def_prop_ro("sample_rate", [](const speech::models::denoiser& d) { return d.sample_rate; },
                     "Sample rate (Hz) the model expects its input at.")
        .def_prop_ro("model_path", [](const speech::models::denoiser& d) { return d.model_path(); },
                     "Where the model file is (still valid after close()).")
        .def_prop_ro("sha256", [](const speech::models::denoiser& d) { return d.sha256(); },
                     "SHA-256 of the model file; '' if it was loaded without a checksum.")
        .def("readme", &speech::models::denoiser::readme,
             "The model's readme: what it is, who made it, its license and provenance.")
        .def("__enter__", [](nb::handle self) { return self; })
        .def("__exit__", [](speech::models::denoiser& self, nb::args) { self.close(); });

    // --- speech::models::SileroVadModel -----------------------------------------------------
    nb::class_<speech::models::speech_timestamp>(m, "SpeechTimestamp")
        .def(nb::init<int, int, int>(),
             nb::arg("start") = -1, nb::arg("end") = -1, nb::arg("sample_rate") = 16000)
        .def_ro("start", &speech::models::speech_timestamp::start, "Start of the speech segment, in samples.")
        .def_ro("end", &speech::models::speech_timestamp::end, "End of the speech segment, in samples.")
        .def_prop_ro("start_s", &speech::models::speech_timestamp::start_s, "Start of the speech segment, in seconds.")
        .def_prop_ro("end_s", &speech::models::speech_timestamp::end_s, "End of the speech segment, in seconds.")
        .def("__repr__", &speech::models::speech_timestamp::c_str)
        .def("__eq__", &speech::models::speech_timestamp::operator==);

    nb::class_<speech::models::silero_vad_model>(m, "SileroVad")
        .def(nb::init<const std::string&, int, int, float, int, int, int, float, bool>(),
             nb::arg("model_path") = "silero-vad.onnx",
             nb::arg("sample_rate") = 16000,
             nb::arg("window_frame_size") = 32,
             nb::arg("threshold") = 0.5f,
             nb::arg("min_silence_duration_ms") = 100,
             nb::arg("speech_pad_ms") = 30,
             nb::arg("min_speech_duration_ms") = 250,
             nb::arg("max_speech_duration_s") = std::numeric_limits<float>::infinity(),
             nb::arg("ignore_cache") = false,
             "Loads a Silero VAD model. model_path is a model file name from libspeech's models release "
             "or a URL (the .onnx weights are downloaded and checked on first use). ignore_cache=True "
             "downloads them again even if cached.")
        .def_prop_ro("model_path", [](const speech::models::silero_vad_model& v) { return v.model_file(); },
                     "Where the model file is.")
        .def_prop_ro("sha256", [](const speech::models::silero_vad_model& v) { return v.sha256(); },
                     "SHA-256 of the model file; '' if it was loaded without a checksum.")
        .def("readme", &speech::models::silero_vad_model::readme,
             "The model's readme: what it is, who made it, its license and provenance.")
        .def("process", &speech::models::silero_vad_model::process_on_vector,
             "Runs voice-activity detection over the full input audio "
             "(mono, [-1, 1]-normalized floats at the model's sample_rate).",
             nb::arg("input_audio"))
        .def("feed",
             [](speech::models::silero_vad_model& self, const std::vector<float>& chunk) {
                 self.feed(chunk.data(), chunk.size());
             },
             nb::arg("chunk"),
             "Streaming: appends a chunk of audio (any length; mono, [-1, 1] floats at the model's sample_rate). "
             "Complete windows are processed right away. Splitting the audio differently never changes the result: "
             "feed(all) + finish() equals process(all).")
        .def("finish", &speech::models::silero_vad_model::finish,
             "Streaming: ends the stream and closes a speech segment that is still open.")
        .def("pop_speech_timestamps", &speech::models::silero_vad_model::pop_speech_timestamps,
             "Streaming: the segments completed since the previous call (each is returned once).")
        .def_prop_ro("in_speech", &speech::models::silero_vad_model::in_speech,
                     "True while a speech segment has started and its end is not decided yet.")
        .def_prop_ro("samples_fed", &speech::models::silero_vad_model::samples_fed,
                     "Samples fed since the last reset().")
        .def("get_speech_timestamps", &speech::models::silero_vad_model::get_speech_timestamps,
             "Returns the detected speech segments (call after process()).")
        .def("reset", &speech::models::silero_vad_model::reset,
             "Resets internal state, so the same model instance can be "
             "reused on a new, unrelated piece of audio.");
}
