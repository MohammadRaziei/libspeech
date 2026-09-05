# Test targets follow a two-axis naming scheme: speech_test_<language>_<module>.
# ctoon only has one axis (language: ctoon_test_c/cpp/python, aggregated
# under ctoon_test) because it's a single-module library; libspeech needs
# both explicit because the C++ side itself is split into independent
# libraries (speech::dsp, speech::models), and because "audio" (file I/O,
# the umbrella Audio class) is conceptually distinct from "dsp" (signal
# processing algorithms) even though only one of the two currently has a
# matching test on each language side (dsp: C++ only so far; audio: Python
# only so far, via the Audio binding -- see checklist.md for binding dsp
# to Python as a tracked follow-up, which would add speech_test_python_dsp).
#
#   speech_test                     -- everything
#   |-- speech_test_cpp             -- all C++ tests
#   |   |-- speech_test_cpp_dsp     -- speech::dsp (this file; cheap, no
#   |   |                             ONNXRuntime/network deps)
#   |   `-- speech_test_cpp_models  -- speech::models (tests/models/)
#   `-- speech_test_python          -- all Python tests
#       |-- speech_test_python_audio   -- libspeech.Audio (tests/python/)
#       `-- speech_test_python_models  -- libspeech.Denoiser/SileroVad (tests/python/)
#
# Every leaf gets both a ctest entry (so plain `ctest` still works) AND a
# directly buildable "run" target, so `cmake --build . --target <name>`
# builds *and runs* it immediately -- discoverable via `cmake --build .
# --target help`. The `speech_test`/`speech_test_cpp`/`speech_test_python`
# aggregates are declared here (even under -DBUILD_MODELS=OFF, where only
# speech_test_cpp_dsp will ever actually attach to them) so tests/models/
# and tests/python/ always have something to add_dependencies() onto,
# regardless of which subset of this file tree gets add_subdirectory'd.

add_custom_target(speech_test
    COMMAND ${CMAKE_COMMAND} -E echo "-- Running all speech tests..."
)
add_custom_target(speech_test_cpp
    COMMAND ${CMAKE_COMMAND} -E echo "-- Running all speech C++ tests..."
)
add_custom_target(speech_test_python
    COMMAND ${CMAKE_COMMAND} -E echo "-- Running all speech Python tests..."
)
add_dependencies(speech_test speech_test_cpp speech_test_python)

file(GLOB SPEECH_DSP_TEST_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/dsp/*.cpp
)

add_executable(test_cpp_dsp
    ${SPEECH_DSP_TEST_SOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/utest/utest_main.cxx
)
target_include_directories(test_cpp_dsp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/utest)
target_link_libraries(test_cpp_dsp PRIVATE speech_dsp)

add_test(NAME cpp_dsp COMMAND test_cpp_dsp)

add_custom_target(speech_test_cpp_dsp
    COMMAND ${CMAKE_COMMAND} -E echo "-- Running speech::dsp C++ tests..."
    COMMAND $<TARGET_FILE:test_cpp_dsp>
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    USES_TERMINAL
)
add_dependencies(speech_test_cpp_dsp test_cpp_dsp)
add_dependencies(speech_test_cpp speech_test_cpp_dsp)
