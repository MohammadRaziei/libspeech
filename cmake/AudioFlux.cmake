# AudioFlux CMake Module
#
# libspeech no longer depends on the full AudioFlux submodule. Only the
# specific C files needed by speech::dsp::* wrappers are vendored under
# src/vendor/audioflux (copied one DSP operator at a time as each is
# ported and test-covered -- see src/vendor/audioflux/README.md and
# UPSTREAM_PATCHES.md for what's there and why).

set(AUDIOFLUX_VENDOR_DIR "${CMAKE_SOURCE_DIR}/src/vendor/audioflux")

file(GLOB_RECURSE AUDIOFLUX_SOURCES
        "${AUDIOFLUX_VENDOR_DIR}/src/*.c"
)

add_library(audioflux STATIC ${AUDIOFLUX_SOURCES})

target_include_directories(audioflux
        PUBLIC  ${AUDIOFLUX_VENDOR_DIR}/include
        PRIVATE ${AUDIOFLUX_VENDOR_DIR}/src
)

target_compile_options(audioflux PRIVATE "-w" "-fPIC")

# AudioFlux's STFT already has a parallel-frame-computation path (each
# frame's FFT is independent, so this is safe), guarded behind HAVE_OMP --
# it was just never enabled. STFT frames genuinely are embarrassingly
# parallel, so this is a real, low-risk speedup rather than a guess.
find_package(OpenMP QUIET)
if(OpenMP_C_FOUND)
    target_compile_definitions(audioflux PRIVATE HAVE_OMP)
    target_link_libraries(audioflux PUBLIC OpenMP::OpenMP_C)
else()
    message(STATUS "OpenMP not found -- speech::dsp::STFT will run single-threaded.")
endif()

set(AUDIOFLUX_FOUND TRUE)
set(AUDIOFLUX_LIBRARIES audioflux)