cmake_minimum_required(VERSION 3.19)

# ==============================================================================
# libspeech Benchmarks — fully standalone CMake project.
#
# This has NO relationship to the repository root. The cpp/ benchmark
# fetches libspeech the exact same way it would fetch any competitor
# library: from https://github.com/MohammadRaziei/libspeech.git via
# FetchContent. libspeech is never a special case here.
#
# Build it on its own:
#
#   cd benchmarks
#   cmake -S . -B build-bench -DCMAKE_BUILD_TYPE=Release
#   cmake --build build-bench --target libspeech_benchmarks
#
# Layout mirrors tests/: one folder per language, each with its own
# CMakeLists.txt that fetches its own dependencies and defines a
# libspeech_benchmarks_<lang> target. A language is skipped with a warning
# only when its toolchain isn't found.
# ==============================================================================

project(LibSpeechBenchmarks LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(WARNING "Benchmarks are being built in Debug mode -- numbers will "
                     "not reflect real performance. Pass -DCMAKE_BUILD_TYPE=Release.")
endif()

set(LIBSPEECH_BENCH_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(LIBSPEECH_BENCH_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(LIBSPEECH_GITHUB_URL "https://github.com/MohammadRaziei/libspeech.git")

add_custom_target(libspeech_benchmarks
    COMMENT "Running all available libspeech language benchmarks"
)

# C++ always runs: this project itself requires a C/CXX compiler to
# configure at all, so there's nothing to conditionally detect.
add_subdirectory(cpp)

# Python: compares libspeech against librosa / audioflux / scipy / soundfile
# (time, peak memory, install size) and renders one standalone HTML report.
# Everything happens in a venv under the build dir; libspeech itself is
# pip-installed from GitHub (never from this checkout) unless overridden:
#
#   cmake -S . -B build-bench -DLIBSPEECH_BENCH_GITHUB_URL=/path/to/local/checkout
#   cmake --build build-bench --target libspeech_benchmarks_python
#
# The venv is created at CONFIGURE time (slow the first time: it compiles
# libspeech and downloads ONNX Runtime); delete build-bench/python/.venv_ready
# to force a rebuild of it. Individual stages have their own targets:
# libspeech_bench_{corpus,throughput,memory,sizes,system_info,report}.
set(LIBSPEECH_BENCH_GITHUB_URL
    "git+https://github.com/MohammadRaziei/libspeech.git"
    CACHE STRING "Where to pip-install libspeech from for the Python benchmarks")
set(LIBSPEECH_BENCH_RESULTS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/results"
    CACHE PATH "Where the final standalone report.html is written (the one artifact meant to be committed)")

find_package(Python3 COMPONENTS Interpreter)
if(Python3_FOUND)
    add_subdirectory(python)
    add_subdirectory(report)
    add_custom_target(libspeech_benchmarks_python
        COMMENT "libspeech_benchmarks_python: full Python benchmark suite + report"
    )
    add_dependencies(libspeech_benchmarks_python libspeech_bench_report)
    add_dependencies(libspeech_benchmarks libspeech_benchmarks_python)
else()
    message(WARNING "Python 3 interpreter not found -- skipping the Python benchmarks.")
endif()
