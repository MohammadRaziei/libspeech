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

# ---------------------------------------------------------------------------
# Coverage (speech::dsp only, for now -- this is the one C++ target with a
# real test suite; extend to speech::models once tests/models/ exists).
#
# Gated on an explicit BUILD_COVERAGE option (default OFF), NOT purely on
# finding lcov/genhtml the way ctoon's tests/cpp/CMakeLists.txt does --
# ctoon.c is compiled directly into ctoon's C++ test binary for this exact
# reason, so instrumenting it only affects that one binary. speech_dsp
# here is a real, separately-built STATIC library that plenty of *other*
# targets link too (the `example` executable, `speech` umbrella library,
# the Python bindings) -- discovered the hard way: with only a
# find_program(lcov) guard, `--coverage` silently landed on speech_dsp on
# any machine that happens to have lcov installed, and every consumer that
# doesn't also link libgcov (i.e. everything except test_cpp_dsp) failed
# with "undefined reference to `__gcov_init'" et al. An explicit,
# default-off option keeps a normal build unaffected regardless of what's
# installed on the machine; the CI coverage job passes -DBUILD_COVERAGE=ON
# explicitly.
# ---------------------------------------------------------------------------
find_program(LCOV_EXECUTABLE lcov)
find_program(GENHTML_EXECUTABLE genhtml)

if(BUILD_COVERAGE AND LCOV_EXECUTABLE AND GENHTML_EXECUTABLE)
    target_compile_options(speech_dsp PRIVATE -O0 -g --coverage)
    target_link_options(speech_dsp PRIVATE --coverage)
    target_compile_options(test_cpp_dsp PRIVATE -O0 -g --coverage)
    target_link_options(test_cpp_dsp PRIVATE --coverage)

    set(SPEECH_COVERAGE_DIR ${CMAKE_BINARY_DIR}/coverage/cpp_dsp)

    add_custom_target(speech_coverage_cpp_dsp
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${SPEECH_COVERAGE_DIR}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${SPEECH_COVERAGE_DIR}

        COMMAND ${CMAKE_COMMAND} -E echo "-- Running instrumented speech::dsp tests..."
        COMMAND $<TARGET_FILE:test_cpp_dsp>

        COMMAND ${CMAKE_COMMAND} -E echo "-- Capturing coverage..."
        COMMAND ${LCOV_EXECUTABLE}
                --capture
                --base-directory ${PROJECT_SOURCE_DIR}
                --directory ${CMAKE_BINARY_DIR}
                --exclude "*/tests/*"
                --no-external
                --rc branch_coverage=1
                --output-file ${SPEECH_COVERAGE_DIR}/coverage_temp.lcov

        COMMAND ${CMAKE_COMMAND} -E echo "-- Filtering speech::dsp source files..."
        COMMAND ${LCOV_EXECUTABLE}
                --extract
                ${SPEECH_COVERAGE_DIR}/coverage_temp.lcov
                "*/src/dsp/*"
                "*/include/libspeech/dsp/*"
                --output-file ${SPEECH_COVERAGE_DIR}/coverage.lcov
        COMMAND ${CMAKE_COMMAND} -E remove ${SPEECH_COVERAGE_DIR}/coverage_temp.lcov

        COMMAND ${CMAKE_COMMAND} -E echo "-- Generating HTML report..."
        COMMAND ${GENHTML_EXECUTABLE} --quiet
                ${SPEECH_COVERAGE_DIR}/coverage.lcov
                --output-directory ${SPEECH_COVERAGE_DIR}/html
                --prefix ${PROJECT_SOURCE_DIR}
                --title "libspeech speech::dsp Coverage"

        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        USES_TERMINAL
        COMMENT "Generating speech::dsp coverage report"
    )
    add_dependencies(speech_coverage_cpp_dsp test_cpp_dsp)

    add_custom_target(speech_coverage
        COMMAND ${CMAKE_COMMAND} -E echo "-- speech coverage: cpp_dsp only so far"
    )
    add_dependencies(speech_coverage speech_coverage_cpp_dsp)
else()
    if(NOT BUILD_COVERAGE)
        message(STATUS "BUILD_COVERAGE=off - speech_coverage target not available (default; pass -DBUILD_COVERAGE=ON to enable)")
    else()
        message(STATUS "lcov/genhtml not found - speech_coverage target will not be available")
    endif()
endif()
