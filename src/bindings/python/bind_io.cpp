#include <nanobind/nanobind.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <filesystem>
#include <string>

#include "libspeech/audio.h"

namespace nb = nanobind;

NB_MODULE(NB_MODULE_NAME, m) {
    nb::class_<speech::io::Audio>(m, "Audio")
        .def(nb::init<>(), "Create an empty Audio object.")
        .def(nb::init<const speech::io::Audio&>(), "Copy an existing Audio object.")

        // Loading
        .def("load",
             static_cast<bool (speech::io::Audio::*)(const std::filesystem::path&)>(&speech::io::Audio::load),
             "Load audio data from a file (WAV, MP3, or FLAC).",
             nb::arg("file_path"))
        .def("load",
             static_cast<bool (speech::io::Audio::*)(const std::vector<std::vector<float>>&, int)>(&speech::io::Audio::load),
             "Load audio data from raw PCM data: a list of channels, each a list of "
             "[-1, 1]-normalized float samples, plus the sample rate.",
             nb::arg("input_data"), nb::arg("sample_rate"))

        // Playback / saving
        .def("play", &speech::io::Audio::play, "Play the loaded audio through the default output device.")
        .def("save", &speech::io::Audio::save, "Save the audio to a WAV file.", nb::arg("output_path"))

        // Transformations (each returns a new Audio; the original is left untouched)
        .def("to_mono", &speech::io::Audio::to_mono,
             "Returns a mono copy of this audio (channels averaged together).")
        .def("resample", &speech::io::Audio::resample,
             "Returns a copy of this audio resampled to target_sample_rate.",
             nb::arg("target_sample_rate"))

        // Data access
        .def("data", static_cast<std::vector<std::vector<float>> (speech::io::Audio::*)() const>(&speech::io::Audio::data),
             "Returns all channels as a list of lists of floats.")
        .def("data", static_cast<std::vector<float> (speech::io::Audio::*)(int) const>(&speech::io::Audio::data),
             "Returns a single channel (by index) as a list of floats.",
             nb::arg("channel_index"))

        // Properties
        .def_prop_ro("sample_rate", &speech::io::Audio::sample_rate, "The sample rate of the audio, in Hz.")
        .def_prop_ro("duration", &speech::io::Audio::duration, "The duration of the audio, in seconds.")
        .def("__len__", &speech::io::Audio::size,
             "The number of samples in this audio (per channel). NOT the "
             "channel count -- use len(audio.data()) for that.")

        .def("__repr__", [](const speech::io::Audio& self) {
            return "Audio(sample_rate=" + std::to_string(self.sample_rate()) +
                   ", channels=" + std::to_string(self.data().size()) +
                   ", duration=" + std::to_string(self.duration()) + "s)";
        });
}
