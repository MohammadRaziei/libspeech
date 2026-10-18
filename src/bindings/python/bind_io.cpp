#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>

#include "libspeech/audio.h"
#include "ndarray_util.h"

namespace nb = nanobind;

namespace {

// Planar float32 samples, (channels, samples), as the NumPy / DLPack world sees them.
using samples_in = nb::ndarray<const float, nb::c_contig, nb::device::cpu>;

// An Audio built from a 1-D (mono) or 2-D (channels, samples) float32 array. The samples are copied
// once (the C++ object owns its own planar buffer); float64 and other dtypes are converted by nanobind.
speech::io::audio audio_from_array(const samples_in& a, int sample_rate) {
    if (a.ndim() != 1 && a.ndim() != 2)
        throw std::invalid_argument("Audio needs a 1-D (samples) or 2-D (channels, samples) array, got " +
                                    std::to_string(a.ndim()) + "-D");
    const std::size_t channels = a.ndim() == 1 ? 1 : a.shape(0);
    const std::size_t frames = a.ndim() == 1 ? a.shape(0) : a.shape(1);
    if (channels == 0 || frames == 0) throw std::invalid_argument("Audio needs at least one channel and one sample");
    std::vector<std::vector<float>> data(channels);
    for (std::size_t c = 0; c < channels; ++c) data[c].assign(a.data() + c * frames, a.data() + (c + 1) * frames);
    speech::io::audio out;
    if (!out.load(data, sample_rate)) throw std::invalid_argument("sample_rate must be positive");
    return out;
}

// A view of the audio's samples, (channels, samples): no copy, and it keeps the buffer alive on its own
// (the capsule co-owns it), so it stays valid after the Audio is loaded again or garbage collected.
template <typename Array>
Array samples_view(const speech::io::audio& a) {
    auto* keep = new std::shared_ptr<float>(a.shared_samples());
    nb::capsule owner(keep, [](void* p) noexcept { delete static_cast<std::shared_ptr<float>*>(p); });
    const std::size_t shape[2] = {static_cast<std::size_t>(a.channels()), a.size()};
    return Array(keep->get(), 2, shape, owner);
}

}  // namespace

NB_MODULE(NB_MODULE_NAME, m) {
    nb::class_<speech::io::audio>(m, "Audio")
        .def(nb::init<>(), "Create an empty Audio object.")
        .def(nb::init<const speech::io::audio&>(), "Copy an existing Audio object.")
        .def(
            "__init__",
            [](speech::io::audio* self, const samples_in& array, int sample_rate) {
                new (self) speech::io::audio(audio_from_array(array, sample_rate));
            },
            nb::arg("array"), nb::arg("sample_rate"),
            "Audio(array, sample_rate): from a 1-D (mono) or 2-D (channels, samples) float array.\n"
            "The samples are copied once; float64 and other dtypes are converted to float32.")
        .def_static(
            "from_numpy",
            [](const samples_in& array, int sample_rate) { return audio_from_array(array, sample_rate); },
            nb::arg("array"), nb::arg("sample_rate"),
            "An Audio from a NumPy array: 1-D (mono) or 2-D (channels, samples). Same as Audio(array, sample_rate).")
        .def_static(
            "from_dlpack",
            [](nb::handle obj, int sample_rate) {
                if (!nb::hasattr(obj, "__dlpack__"))
                    throw nb::type_error("Audio.from_dlpack needs an object with __dlpack__ (a torch, jax, cupy or numpy array...)");
                return audio_from_array(nb::cast<samples_in>(obj), sample_rate);
            },
            nb::arg("obj"), nb::arg("sample_rate"),
            "An Audio from any object that implements DLPack (CPU memory): 1-D or 2-D (channels, samples).\n"
            "The samples are copied once. Raises TypeError without __dlpack__, ValueError for a bad shape.")

        // Loading
        .def("load",
             static_cast<bool (speech::io::audio::*)(const std::filesystem::path&)>(&speech::io::audio::load),
             "Load audio data from a file (WAV, MP3, or FLAC).",
             nb::arg("file_path"))
        .def("load",
             static_cast<bool (speech::io::audio::*)(const std::vector<std::vector<float>>&, int)>(&speech::io::audio::load),
             "Load audio data from raw PCM data: a list of channels, each a list of "
             "[-1, 1]-normalized float samples, plus the sample rate.",
             nb::arg("input_data"), nb::arg("sample_rate"))

        // Playback / saving
        .def("play", &speech::io::audio::play, "Play the loaded audio through the default output device.")
        .def("save", &speech::io::audio::save, "Save the audio to a WAV file.", nb::arg("output_path"))

        // Transformations (each returns a new Audio; the original is left untouched)
        .def("to_mono", &speech::io::audio::to_mono,
             "Returns a mono copy of this audio (channels averaged together).")
        .def("resample", &speech::io::audio::resample,
             "Returns a copy of this audio resampled to target_sample_rate.",
             nb::arg("target_sample_rate"))

        // Data access
        .def("data", static_cast<std::vector<std::vector<float>> (speech::io::audio::*)() const>(&speech::io::audio::data),
             "Returns all channels as a list of lists of floats.")
        .def("data", static_cast<std::vector<float> (speech::io::audio::*)(int) const>(&speech::io::audio::data),
             "Returns a single channel (by index) as a list of floats.",
             nb::arg("channel_index"))

        // NumPy access: one memcpy into a NumPy-owned buffer (no Python floats).
        .def("to_numpy",
             [](const speech::io::audio& self, int channel_index) {
                 size_t n = 0;
                 const float* src = self.channel_data(channel_index, &n);
                 speech::py::float_vec buf(n);
                 if (n) std::memcpy(buf.data(), src, n * sizeof(float));
                 return speech::py::wrap<speech::py::out_array_1d>(std::move(buf), {n});
             },
             nb::arg("channel_index") = 0,
             "Returns one channel as a float32 NumPy array (a copy; no list conversion).")

        // Array interop: planar float32, shape (channels, samples). One shared buffer, so these are views.
        .def_prop_ro("channels", &speech::io::audio::channels, "The number of channels.")
        .def_prop_ro("shape",
                     [](const speech::io::audio& self) { return std::make_tuple(self.channels(), self.size()); },
                     "(channels, samples), the shape of np.asarray(audio).")
        .def(
            "__array__",
            [](nb::handle self, nb::object dtype, nb::object copy) -> nb::object {
                const auto& a = nb::cast<const speech::io::audio&>(self);
                nb::module_ np = nb::module_::import_("numpy");
                nb::object arr =
                    a.shared_samples()
                        ? nb::cast(samples_view<nb::ndarray<nb::numpy, float, nb::ndim<2>>>(a))
                        : np.attr("empty")(nb::make_tuple(0, 0), "float32");
                const bool want_copy = !copy.is_none() && nb::cast<bool>(copy);
                const bool never_copy = !copy.is_none() && !want_copy;  // np.asarray(a, copy=False)
                if (!dtype.is_none() && !np.attr("dtype")(dtype).equal(np.attr("dtype")("float32"))) {
                    if (never_copy) throw nb::value_error("converting to another dtype needs a copy, but copy=False");
                    return arr.attr("astype")(dtype);  // always a copy
                }
                return want_copy ? arr.attr("copy")() : arr;
            },
            nb::arg("dtype") = nb::none(), nb::arg("copy") = nb::none())
        .def(
            "numpy",
            [](nb::handle self, nb::object dtype, bool copy) -> nb::object {
                // copy=False here means "do not force a copy" (a different dtype still converts).
                return self.attr("__array__")(dtype, copy ? nb::object(nb::bool_(true)) : nb::none());
            },
            nb::kw_only(), nb::arg("dtype") = nb::none(), nb::arg("copy") = false,
            "All channels as a float32 ndarray of shape (channels, samples): a view that shares the Audio's\n"
            "memory (writing to it changes the audio), unless copy=True or dtype is not float32. It stays valid\n"
            "if the Audio is loaded again (the Audio then has new memory and the view keeps the old). To get\n"
            "one channel as a 1-D array use to_numpy(channel).")
        .def(
            "__dlpack__",
            [](nb::handle self, nb::args args, nb::kwargs kwargs) {
                // copy=True is ours to honour (nanobind refuses it): export a deep copy.
                nb::object owner = nb::borrow(self);
                if (kwargs.contains("copy")) {
                    nb::object want_copy = kwargs["copy"];
                    nb::del(kwargs["copy"]);
                    if (!want_copy.is_none() && nb::cast<bool>(want_copy))
                        owner = nb::cast(speech::io::audio(nb::cast<const speech::io::audio&>(self)));
                }
                const auto& a = nb::cast<const speech::io::audio&>(owner);
                if (!a.shared_samples()) throw nb::value_error("this Audio has no samples to export");
                // nanobind 2.x hands back an object with __dlpack__, 3.x the capsule itself; the protocol
                // wants the capsule. (An unversioned capsule is always acceptable to a consumer.)
                nb::object exported = nb::cast(samples_view<nb::ndarray<float, nb::ndim<2>, nb::device::cpu>>(a));
                return nb::hasattr(exported, "__dlpack__") ? exported.attr("__dlpack__")(*args, **kwargs) : exported;
            },
            "DLPack export of the samples (float32, shape (channels, samples), no copy): np.from_dlpack(audio),\n"
            "torch.from_dlpack(audio), jax.numpy.from_dlpack(audio) ... NumPy gets a read-only view (the capsule\n"
            "is unversioned); np.asarray(audio) / audio.numpy() is the writable one.")
        .def("__dlpack_device__", [](const speech::io::audio&) { return std::make_tuple(1, 0); },  // kDLCPU, device 0
             "(device type, device id): always the CPU.")

        // Properties
        .def_prop_ro("sample_rate", &speech::io::audio::sample_rate, "The sample rate of the audio, in Hz.")
        .def_prop_ro("duration", &speech::io::audio::duration, "The duration of the audio, in seconds.")
        .def("__len__", &speech::io::audio::size,
             "The number of samples in this audio (per channel). NOT the "
             "channel count -- use len(audio.data()) for that.")

        .def("__repr__", [](const speech::io::audio& self) {
            return "Audio(sample_rate=" + std::to_string(self.sample_rate()) +
                   ", channels=" + std::to_string(self.channels()) +
                   ", duration=" + std::to_string(self.duration()) + "s)";
        });
}
