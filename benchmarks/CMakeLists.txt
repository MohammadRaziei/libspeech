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

# As Python (and other language) benchmarks are added, follow the same
# toolchain-detection pattern used in tests/ and ctoon's own
# benchmarks/CMakeLists.txt: `find_package`/`find_program`, skip with a
# `message(WARNING ...)` if not found, never a hard configure error.
