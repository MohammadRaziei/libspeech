# speech::dsp unit tests, built with utest.h (see tests/utest/) against the
# lightweight `speech_dsp` target -- deliberately independent of the full
# `speech` library's ONNXRuntime/httplib/miniaudio dependencies, so these
# build and run fast with nothing but a C/C++ compiler.
#
# Mirrors ctoon's per-language test convention (see tests/cpp/CMakeLists.txt
# and tests/c/CMakeLists.txt in github.com/mohammadraziei/ctoon): every
# test_*.cpp file for a given module is globbed into ONE executable (there,
# test_ctoon_cpp/test_ctoon_c; here, test_speech_dsp), not one binary per
# source file. utest.h's UTEST() macros self-register regardless of which
# .cpp they're compiled in, so a single shared main -- tests/utest/utest_main.cxx,
# living next to utest.h itself so any future test module (e.g.
# speech::models) can reuse it the same way -- is enough for the whole
# module; it's added explicitly, not part of the dsp/ glob. That one
# executable gets both a ctest entry (so plain `ctest` still works) AND a
# directly buildable "run" target, so `cmake --build . --target <name>`
# builds *and runs* it immediately -- discoverable via `cmake --build .
# --target help`. As speech::models grows its own tests, add a parallel
# tests/models/ directory + `speech_test_models` target the same way, and
# add it as a dependency of `speech_test`.

add_custom_target(speech_test
    COMMAND ${CMAKE_COMMAND} -E echo "-- Running all speech tests..."
)

file(GLOB SPEECH_DSP_TEST_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/dsp/*.cpp
)

add_executable(test_speech_dsp
    ${SPEECH_DSP_TEST_SOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/utest/utest_main.cxx
)
target_include_directories(test_speech_dsp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/utest)
target_link_libraries(test_speech_dsp PRIVATE speech_dsp)

add_test(NAME speech_dsp COMMAND test_speech_dsp)

add_custom_target(speech_test_dsp
    COMMAND ${CMAKE_COMMAND} -E echo "-- Running speech::dsp tests..."
    COMMAND $<TARGET_FILE:test_speech_dsp>
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    USES_TERMINAL
)
add_dependencies(speech_test_dsp test_speech_dsp)
add_dependencies(speech_test speech_test_dsp)
