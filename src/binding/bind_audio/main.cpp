#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <filesystem>
#include <string>

#include "audio.h"

namespace nb = nanobind;

// Define the Python bindings for the Audio class
NB_MODULE(NB_MODULE_NAME, m) {
    nb::class_<speech::Audio>(m, "Audio")
        .def(nb::init<>(), "Initialize an Audio object.")

        // Methods
        .def("load",
             static_cast<bool (speech::Audio::*)(const std::filesystem::path&)>(&speech::Audio::load),
             "Load audio data from a file.",
             nb::arg("file_path"))
        .def("load",
             static_cast<bool (speech::Audio::*)(const std::vector<float>&, int, int)>(&speech::Audio::load),
             "Load audio data from raw PCM data.",
             nb::arg("input_data"), nb::arg("sample_rate"), nb::arg("channels"))
        .def("play", &speech::Audio::play, "Play the loaded audio.")
        .def("save", &speech::Audio::save, "Save the audio to a file.", nb::arg("output_path"))

        // Properties
        .def_property_readonly("data", &speech::Audio::data, "Get the raw PCM audio data as a list of floats.")
        .def_property_readonly("sample_rate", &speech::Audio::sampleRate, "Get the sample rate of the audio.")
        .def_property_readonly("duration", &speech::Audio::duration, "Get the duration of the audio in seconds.");
}
