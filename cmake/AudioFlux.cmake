# AudioFlux CMake Module
#
# libspeech no longer depends on the full AudioFlux submodule. Only the
# specific C files needed by speech::dsp::* wrappers are vendored under
# src/third_party/audioflux (copied one DSP operator at a time as each is
# ported and test-covered -- see src/third_party/audioflux/README.md and
# UPSTREAM_PATCHES.md for what's there and why).

set(AUDIOFLUX_VENDOR_DIR "${PROJECT_SOURCE_DIR}/src/third_party/audioflux")

file(GLOB_RECURSE AUDIOFLUX_SOURCES
        "${AUDIOFLUX_VENDOR_DIR}/src/*.c"
)

add_library(audioflux STATIC ${AUDIOFLUX_SOURCES})

target_include_directories(audioflux
        PUBLIC  ${AUDIOFLUX_VENDOR_DIR}/include
        PRIVATE ${AUDIOFLUX_VENDOR_DIR}/src
)

# -w/-fPIC are GCC/Clang-only flags; MSVC doesn't understand them (it just
# warns D9025/D9002 and continues), so only pass them on non-MSVC compilers.
# (NOMINMAX/_USE_MATH_DEFINES for MSVC are set project-wide in the top-level
# CMakeLists.txt, since they're needed by more than just this target.)
if(NOT MSVC)
    target_compile_options(audioflux PRIVATE "-w" "-fPIC")
endif()

# AudioFlux's STFT already has a parallel-frame-computation path (each
# frame's FFT is independent, so this is safe), guarded behind HAVE_OMP --
# it was just never enabled. STFT frames genuinely are embarrassingly
# parallel, so this is a real, low-risk speedup rather than a guess -- but
# it could not be *verified* in the sandbox this was developed in (only 1
# CPU core available there, so OpenMP has no parallelism to exploit by
# construction). LIBSPEECH_ENABLE_OPENMP exists specifically so this can be
# A/B tested on real multi-core hardware -- see benchmark/bench_stft.cpp.
option(LIBSPEECH_ENABLE_OPENMP "Enable OpenMP for speech::dsp::STFT's parallel-frame path" ON)
if(LIBSPEECH_ENABLE_OPENMP)
    find_package(OpenMP QUIET)
endif()
if(LIBSPEECH_ENABLE_OPENMP AND OpenMP_C_FOUND)
    target_compile_definitions(audioflux PRIVATE HAVE_OMP)
    target_link_libraries(audioflux PUBLIC OpenMP::OpenMP_C)
    message(STATUS "OpenMP enabled for speech::dsp::STFT.")
else()
    message(STATUS "OpenMP disabled -- speech::dsp::STFT will run single-threaded.")
endif()

set(AUDIOFLUX_FOUND TRUE)
set(AUDIOFLUX_LIBRARIES audioflux)