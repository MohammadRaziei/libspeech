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
# Coverage -- mirrors ctoon's tests/CMakeLists.txt layout: one
# speech_coverage_<component> target per component (cpp_dsp, cpp_models,
# python), a merged speech_coverage_total, a static HTML dashboard tying the
# reports together (coverage/index.html), and a speech_coverage umbrella.
# Component targets live next to the tests they run (this file for cpp_dsp,
# tests/models/ for cpp_models, tests/python/ for python); the merge +
# dashboard step needs to see every component that actually got configured,
# so it lives in speech_coverage_finalize() (defined below), called from the
# root CMakeLists.txt once all test subdirectories have been added.
#
# One deliberate difference from ctoon: everything here is gated on an
# explicit BUILD_COVERAGE option (default OFF), not merely on lcov/genhtml
# being installed. ctoon.c is compiled directly into ctoon's own C++ test
# binary, so instrumenting it only ever affects that one binary. speech_dsp/
# speech_models are real, separately-built STATIC libraries that plenty of
# *other* targets link too (the `example` executable, the `speech` umbrella
# library, the Python bindings) -- gating purely on find_program(lcov) once
# put `--coverage` on speech_dsp on any machine that merely had lcov
# installed, and every consumer that didn't also link libgcov failed with
# "undefined reference to `__gcov_init'". So: explicit opt-in, and the
# instrumented libraries export --coverage as a PUBLIC *link* option so
# consumers always link libgcov too. CI's coverage job passes
# -DBUILD_COVERAGE=ON explicitly.
# ---------------------------------------------------------------------------
set(SPEECH_COVERAGE_BINARY_DIR ${CMAKE_BINARY_DIR}/coverage CACHE PATH "Coverage output directory")
find_program(LCOV_EXECUTABLE lcov)
find_program(GENHTML_EXECUTABLE genhtml)

set(SPEECH_COVERAGE_ENABLED OFF CACHE INTERNAL "")
if(BUILD_COVERAGE AND LCOV_EXECUTABLE AND GENHTML_EXECUTABLE)
    set(SPEECH_COVERAGE_ENABLED ON CACHE INTERNAL "")
elseif(BUILD_COVERAGE)
    message(WARNING "BUILD_COVERAGE=ON but lcov/genhtml not found - coverage targets will not be available")
else()
    message(STATUS "BUILD_COVERAGE=off - speech_coverage* targets not available (default; pass -DBUILD_COVERAGE=ON to enable)")
endif()

if(SPEECH_COVERAGE_ENABLED)
    file(REMOVE_RECURSE ${SPEECH_COVERAGE_BINARY_DIR})
    file(MAKE_DIRECTORY ${SPEECH_COVERAGE_BINARY_DIR})

    add_custom_target(speech_coverage
        COMMAND ${CMAKE_COMMAND} -E echo "-- Running coverage for all components ..."
    )
endif()

# Instrument a library and make sure everything linking it also links libgcov.
function(speech_enable_coverage target)
    if(SPEECH_COVERAGE_ENABLED)
        target_compile_options(${target} PRIVATE -O0 -g --coverage)
        target_link_options(${target} PUBLIC --coverage)
    endif()
endfunction()

# speech_add_cpp_coverage_target(<id> <title> <run_target> <extract patterns...>)
#   <id>          -> target speech_coverage_<id>, output coverage/<id>/
#   <run_target>  -> the existing speech_test_* target that builds + runs the
#                    (instrumented) tests, producing the .gcda files
#   patterns      -> lcov --extract globs limiting the report to this
#                    component's own sources (the test executables aren't
#                    instrumented and third_party/ is filtered out here)
function(speech_add_cpp_coverage_target id title run_target)
    if(NOT SPEECH_COVERAGE_ENABLED)
        return()
    endif()
    set(_dir ${SPEECH_COVERAGE_BINARY_DIR}/${id})
    file(MAKE_DIRECTORY ${_dir})

    add_custom_target(speech_coverage_${id}
        # Clean previous coverage data (only html, keep index.html wrapper)
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${_dir}/html
        COMMAND ${CMAKE_COMMAND} -E make_directory ${_dir}/html

        # Capture coverage data (the whole build tree: speech_dsp/speech_models
        # are separately-built libraries whose .gcda files don't live next to
        # the test executable's)
        COMMAND ${CMAKE_COMMAND} -E echo "-- Capturing coverage..."
        COMMAND ${LCOV_EXECUTABLE}
                --capture
                --base-directory ${PROJECT_SOURCE_DIR}
                --directory ${CMAKE_BINARY_DIR}
                --exclude "*/src/third_party/*"
                --ignore-errors unused
                --no-external
                --rc branch_coverage=1
                --rc geninfo_unexecuted_blocks=1
                --output-file ${_dir}/coverage_temp.lcov
        # Extract only this component's source files
        COMMAND ${CMAKE_COMMAND} -E echo "-- Filtering ${id} source files..."
        COMMAND ${LCOV_EXECUTABLE}
                --extract ${_dir}/coverage_temp.lcov ${ARGN}
                --ignore-errors unused
                --rc branch_coverage=1
                --output-file ${_dir}/coverage.lcov
        COMMAND ${CMAKE_COMMAND} -E remove ${_dir}/coverage_temp.lcov

        # Generate HTML report
        COMMAND ${CMAKE_COMMAND} -E echo "-- Generating HTML report..."
        COMMAND ${GENHTML_EXECUTABLE} --quiet
                ${_dir}/coverage.lcov
                --output-directory ${_dir}/html
                --prefix ${PROJECT_SOURCE_DIR}
                --title "${title}"
                --branch-coverage
                --css-file ${PROJECT_SOURCE_DIR}/tests/genhtml.css
                --dark-mode
        COMMAND ${CMAKE_COMMAND} -E echo "-- HTML entry point created here: ${_dir}/html/index.html"

        # Summary. `lcov --summary`, not ctoon's `lcov --list`: lcov 2.0-1
        # (what Ubuntu 24.04 ships) prints the --list columns wrong -- e.g.
        # 4.8% for a file whose record says LF:21 LH:21, and function "rates"
        # like 7200% -- while --summary agrees with the raw LF/LH/FNF/FNH
        # counters in the .lcov file.
        COMMAND ${CMAKE_COMMAND} -E echo "-- Coverage summary:"
        COMMAND ${LCOV_EXECUTABLE} --summary ${_dir}/coverage.lcov --rc branch_coverage=1

        # Fix paths to be relative (for SonarQube, Codecov, merging, ...)
        COMMAND ${CMAKE_COMMAND} -E echo "-- Fixing paths to relative..."
        COMMAND ${CMAKE_COMMAND}
                -DCOV_FILE=${_dir}/coverage.lcov
                -DSOURCE_PATH=${PROJECT_SOURCE_DIR}/
                -P ${PROJECT_SOURCE_DIR}/cmake/FixLCovPaths.cmake

        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        USES_TERMINAL
    )
    add_dependencies(speech_coverage_${id} ${run_target})
    add_dependencies(speech_coverage speech_coverage_${id})

    # iframe wrapper around html/index.html (same look as the dashboard)
    add_custom_command(TARGET speech_coverage_${id} POST_BUILD
        COMMAND ${CMAKE_COMMAND}
            -DIFRAME_SRC="html/index.html"
            -DSECTION_TITLE="${title}"
            -DLOGO_SVG="${PROJECT_SOURCE_DIR}/docs/logo/libspeech-logo.svg"
            -DPROJECT_NAME="${PROJECT_NAME}"
            -DOUTPUT_DIR="${_dir}"
            -P "${PROJECT_SOURCE_DIR}/cmake/GenerateIframe.cmake"
        COMMENT "Generating iframe wrapper for ${id} coverage report"
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests
    )
endfunction()

speech_enable_coverage(speech_dsp)
speech_add_cpp_coverage_target(cpp_dsp "libspeech C++ DSP Coverage" speech_test_cpp_dsp
    "*/src/dsp/*" "*/include/libspeech/dsp/*")

# Merge every component's lcov file into coverage/total/ and render the
# dashboard. Called from the root CMakeLists.txt after tests/models and
# tests/python have had their chance to define their targets.
function(speech_coverage_finalize)
    if(NOT SPEECH_COVERAGE_ENABLED)
        return()
    endif()
    set(_total ${SPEECH_COVERAGE_BINARY_DIR}/total)

    set(_merge_args)
    set(_merge_deps)
    foreach(_id cpp_dsp cpp_models python)
        if(TARGET speech_coverage_${_id})
            list(APPEND _merge_args --add-tracefile ${SPEECH_COVERAGE_BINARY_DIR}/${_id}/coverage.lcov)
            list(APPEND _merge_deps speech_coverage_${_id})
        endif()
    endforeach()

    add_custom_target(speech_coverage_total
        # Clean previous coverage data
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${_total}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${_total}/html
        # Merge all coverage info files
        COMMAND ${CMAKE_COMMAND} -E echo "-- Merging coverage of all components..."
        COMMAND ${LCOV_EXECUTABLE} ${_merge_args} --rc branch_coverage=1 --output-file ${_total}/coverage.lcov
        COMMAND ${CMAKE_COMMAND} -E echo "-- Total coverage info created at: ${_total}/coverage.lcov"

        # genhtml needs resolvable (absolute) paths; put the relative ones back after
        COMMAND ${CMAKE_COMMAND}
                -DCOV_FILE=${_total}/coverage.lcov
                -DPREFIX_PATH=${PROJECT_SOURCE_DIR}/
                -P ${PROJECT_SOURCE_DIR}/cmake/FixLCovPaths.cmake
        COMMAND ${CMAKE_COMMAND} -E echo "-- Generating HTML report..."
        COMMAND ${GENHTML_EXECUTABLE} --quiet
                ${_total}/coverage.lcov
                --output-directory ${_total}/html
                --prefix ${PROJECT_SOURCE_DIR}
                --title "libspeech Total Coverage"
                --branch-coverage
                --css-file ${PROJECT_SOURCE_DIR}/tests/genhtml.css
                --dark-mode
        COMMAND ${CMAKE_COMMAND}
                -DCOV_FILE=${_total}/coverage.lcov
                -DSOURCE_PATH=${PROJECT_SOURCE_DIR}/
                -P ${PROJECT_SOURCE_DIR}/cmake/FixLCovPaths.cmake
        COMMAND ${CMAKE_COMMAND} -E echo "-- HTML entry point created here: ${_total}/html/index.html"
        COMMAND ${CMAKE_COMMAND} -E echo "-- Dashboard: ${SPEECH_COVERAGE_BINARY_DIR}/index.html"

        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        USES_TERMINAL
    )
    foreach(_dep IN LISTS _merge_deps)
        add_dependencies(speech_coverage_total ${_dep})
    endforeach()
    add_dependencies(speech_coverage speech_coverage_total)

    add_custom_command(TARGET speech_coverage_total POST_BUILD
        COMMAND ${CMAKE_COMMAND}
            -DIFRAME_SRC="html/index.html"
            -DSECTION_TITLE="Total%20Coverage"
            -DLOGO_SVG="${PROJECT_SOURCE_DIR}/docs/logo/libspeech-logo.svg"
            -DPROJECT_NAME="${PROJECT_NAME}"
            -DOUTPUT_DIR="${_total}"
            -P "${PROJECT_SOURCE_DIR}/cmake/GenerateIframe.cmake"
        COMMENT "Generating iframe wrapper for total coverage report"
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests
    )

    # --- Dashboard (coverage/index.html), same JSON-driven page as ctoon's ---
    set(COVERAGE_DATA_JSON "[
        {
            \"id\": \"total\",
            \"title\": \"Total Coverage\",
            \"badge\": \"Merged\",
            \"badgeClass\": \"total\",
            \"color\": \"#a371f7\",
            \"desc\": \"Combined coverage report for every component below (C++ and Python). Useful for SonarQube/Codecov analysis.\",
            \"htmlLink\": \"total/index.html\",
            \"lcovLink\": \"total/coverage.lcov\"
        },")
    if(TARGET speech_coverage_cpp_dsp)
        string(APPEND COVERAGE_DATA_JSON "
        {
            \"id\": \"cpp_dsp\",
            \"title\": \"C++ DSP (speech::dsp)\",
            \"badge\": \"Native\",
            \"color\": \"#f0883e\",
            \"desc\": \"Coverage of the signal-processing layer (FFT/STFT/MFCC/DCT/window/resample), generated by LCOV from the utest.h suite.\",
            \"htmlLink\": \"cpp_dsp/index.html\",
            \"lcovLink\": \"cpp_dsp/coverage.lcov\"
        },")
    endif()
    if(TARGET speech_coverage_cpp_models)
        string(APPEND COVERAGE_DATA_JSON "
        {
            \"id\": \"cpp_models\",
            \"title\": \"C++ Models (speech::models)\",
            \"badge\": \"Native\",
            \"color\": \"#2ecc71\",
            \"desc\": \"Coverage of the model layer (BaseModel/ONNXModel/denoisers/SileroVad + download utilities), generated by LCOV.\",
            \"htmlLink\": \"cpp_models/index.html\",
            \"lcovLink\": \"cpp_models/coverage.lcov\"
        },")
    endif()
    if(TARGET speech_coverage_python)
        string(APPEND COVERAGE_DATA_JSON "
        {
            \"id\": \"py\",
            \"title\": \"Python Bindings\",
            \"badge\": \"Interpreter\",
            \"color\": \"#3572A5\",
            \"desc\": \"Coverage of the Python package (libspeech/*.py), generated by coverage.py.\",
            \"htmlLink\": \"python/index.html\",
            \"lcovLink\": \"python/coverage.lcov\"
        },")
    endif()
    string(APPEND COVERAGE_DATA_JSON "]")

    set(_logo "${PROJECT_SOURCE_DIR}/docs/logo/libspeech-logo.svg")
    if(EXISTS "${_logo}")
        file(READ "${_logo}" LOGO_CONTENT)
    else()
        set(LOGO_CONTENT "")
    endif()
    configure_file(
        ${PROJECT_SOURCE_DIR}/tests/coverage_dashboard.html.in
        ${SPEECH_COVERAGE_BINARY_DIR}/index.html
        @ONLY
    )
endfunction()
