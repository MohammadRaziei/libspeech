cmake_minimum_required(VERSION 3.19)

option(BUILD_PYTHON "export python module" off)
option(BUILD_TESTS "build unit tests (speech::dsp only, no ONNXRuntime/httpp needed)" on)
option(BUILD_MODELS "build speech::models + the speech library (needs ONNXRuntime download + the httpp package); off configures speech::dsp + tests only" on)

set(CMAKE_C_STANDARD 99 CACHE STRING "C version selection")
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ version selection")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# speech_dsp, speech_models, and audioflux are STATIC libraries that end up
# linked into the `speech` SHARED library -- every object file involved
# needs to be position-independent for that to link.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
include(DynamicVersion)

extract_version_string(
    HEADER_FILE ${CMAKE_CURRENT_LIST_DIR}/include/libspeech/version.h
    VERSION_PREFIX LIBSPEECH_
    OUTPUT_VAR PROJECT_VERSION
)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

project(libspeech LANGUAGES C CXX VERSION ${PROJECT_VERSION})

# Several dependencies -- miniaudio, httpp, and the vendored utest.h test
# framework -- transitively #include <Windows.h> on Windows. Without
# NOMINMAX, that header defines `min`/`max` function-like macros that
# silently rewrite every `std::min(...)`/`std::max(...)` call in the same
# translation unit (e.g. src/audio.cpp, tests/dsp/test_stft.cpp) into
# invalid syntax -- MSVC reports this as the rather cryptic
# "C2589: '(': illegal token on right side of '::'". WIN32_LEAN_AND_MEAN
# additionally trims <Windows.h> itself down to the common subset, avoiding
# a class of unrelated symbol clashes (e.g. with winsock).
# MSVC's own <math.h>/<cmath> also don't define M_PI unless
# _USE_MATH_DEFINES is set beforehand, which vendored AudioFlux code and
# several DSP tests rely on unconditionally.
# All three are applied project-wide (every target, main lib and tests
# alike) since any of them can pull in the offending headers.
if(WIN32)
    add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN _USE_MATH_DEFINES)
endif()

# Every target that dlopens/links libonnxruntime.so.1 or libhttpp_core.so
# (speech, and every Python extension module below, since each one
# independently links against them too -- see the CMakeLists.txt comment
# near `add_library(speech SHARED ...)` for why) ends up installed into the
# exact same site-packages/libspeech/ directory as those bundled runtime
# libraries. CMake's install step strips build-tree RPATH down to empty by
# default, and that directory is on no standard search path, so without an
# explicit RPATH pointing back at "wherever this .so itself ended up",
# nothing can find them at runtime -- and since auditwheel's repair step
# does the same ELF dependency walk a real dynamic link would, it fails
# identically ("required library ... could not be located") even though
# the file is right there in the wheel. $ORIGIN (Linux) / @loader_path
# (macOS) mean exactly "the directory containing this .so", correct
# regardless of where the wheel ultimately gets installed -- unlike an
# absolute path, which would only be valid on this build machine. Windows
# needs no equivalent call at all: its default DLL search order already
# includes the loading module's own directory first.
function(libspeech_set_origin_rpath target)
    if(APPLE)
        set_target_properties(${target} PROPERTIES INSTALL_RPATH "@loader_path")
    elseif(NOT WIN32)
        set_target_properties(${target} PROPERTIES INSTALL_RPATH "$ORIGIN")
    endif()
endfunction()

message(STATUS "Project: ${PROJECT_NAME}@v${PROJECT_VERSION}")

include(AudioFlux)

# ---------------------------------------------------------------------------
# speech::dsp -- resample, window, FFT, STFT, MFCC, ...
#
# Deliberately independent of ONNXRuntime/httpp/miniaudio: this lets DSP
# unit tests (tests/CMakeLists.txt) build and run without pulling in the
# model/audio-I/O stack, matching the project's "minimal dependencies" goal
# and keeping DSP iteration fast. See BUILD_MODELS below.
# ---------------------------------------------------------------------------
file(GLOB SPEECH_DSP_SOURCES CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/src/dsp/*.cpp")
add_library(speech_dsp STATIC ${SPEECH_DSP_SOURCES})
add_library(speech::dsp ALIAS speech_dsp)

target_include_directories(speech_dsp
        PUBLIC ${PROJECT_SOURCE_DIR}/include
        PUBLIC src/vendor/aixlog/include
        PRIVATE src/vendor/audioflux/src   # private AudioFlux headers (fft_algorithm.h, stft_algorithm.h, ...)
)
target_link_libraries(speech_dsp PUBLIC audioflux)

# See include/libspeech/export.h -- this makes SPEECH_API expand to
# __declspec(dllexport) for FFT/STFT/MFCC/Resample when compiled here (as
# part of a static lib that may end up bundled into the speech.dll umbrella
# target below). No-op on non-Windows.
target_compile_definitions(speech_dsp PRIVATE SPEECH_BUILDING_SHARED)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(NOT BUILD_MODELS)
    message(STATUS "BUILD_MODELS=off: configuring speech::dsp + tests only, skipping ONNXRuntime/models/audio I/O.")
    return()
endif()

include(ONNXRuntime)

add_subdirectory(src/third_party/miniaudio)

# ---------------------------------------------------------------------------
# Model downloading (speech::utils::downloadFile) used to link the SYSTEM
# libcurl (find_package(CURL REQUIRED) + apt-installed libcurl4-openssl-dev)
# -- a real system dependency the "pip install and you're done" story
# doesn't want, and one this project's own build has hit in practice (a
# sandboxed CI environment without apt access to security.ubuntu.com failed
# here). Replaced with `httpp` (https://github.com/mohammadraziei/httpp,
# `pip install httpp`): it ships its own HTTP(S) client, URL parser, and
# terminal progress bar prebuilt as a shared library + CMake config, so
# neither Mbed TLS nor `indicators` needs to be a submodule/build target of
# this project anymore -- `find_package(httpp)` is enough.
#
# httpp is listed in pyproject.toml's [build-system] requires precisely so
# that this succeeds even inside scikit-build-core's isolated build venv
# (`python -m build`/cibuildwheel) -- that venv only has whatever's listed
# there, and without it `import httpp` failed with "ModuleNotFoundError: No
# module named 'httpp'" even though a normal `pip install .` never hit this
# (a local/dev Python environment has httpp for other reasons already).
# ---------------------------------------------------------------------------
find_package(Python3 REQUIRED COMPONENTS Interpreter)
execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import httpp; print(httpp.get_cmake_dir())"
        OUTPUT_VARIABLE HTTPP_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE HTTPP_CMAKE_DIR_RESULT
)
if(NOT HTTPP_CMAKE_DIR_RESULT EQUAL 0 OR NOT HTTPP_CMAKE_DIR)
    message(FATAL_ERROR
            "Could not locate the `httpp` CMake config via Python (`import httpp; "
            "httpp.get_cmake_dir()`). Install it with `pip install httpp` into the "
            "Python environment CMake is using (${Python3_EXECUTABLE}).")
endif()
message(STATUS "Using httpp CMake package config from: ${HTTPP_CMAKE_DIR}")
find_package(httpp REQUIRED CONFIG PATHS "${HTTPP_CMAKE_DIR}" NO_DEFAULT_PATH)

# ---------------------------------------------------------------------------
# speech::models -- BaseModel/ONNXModel and the VAD/denoiser backends.
#
# Separate from speech::dsp (no shared code) and from the main `speech`
# library (no audio-file-I/O dependency): models take raw float buffers in
# and out, and only need ONNXRuntime plus the download utility code
# (base_model.cpp downloads .onnx weights on first use, via httpp).
# ---------------------------------------------------------------------------
file(GLOB SPEECH_MODELS_SOURCES CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/src/models/*.cpp")
add_library(speech_models STATIC
        ${SPEECH_MODELS_SOURCES}
        src/utils/utils.cpp
)
add_library(speech::models ALIAS speech_models)

target_include_directories(speech_models
        PUBLIC ${PROJECT_SOURCE_DIR}/include
        PUBLIC src/vendor/aixlog/include
)
target_link_libraries(speech_models
        PUBLIC onnxruntime_interface
        PUBLIC httpp::httpp_core
)

# See include/libspeech/export.h. No-op on non-Windows.
target_compile_definitions(speech_models PRIVATE SPEECH_BUILDING_SHARED)

if(BUILD_TESTS)
    add_subdirectory(tests/models)
endif()

# ---------------------------------------------------------------------------
# speech::io -- audio file I/O (the Audio class: load/save/play/resample/
# to_mono), backed by miniaudio + dr_libs. Mirrors speech::dsp/speech::models:
# its own static library, its own namespace (speech::io in the C++ code,
# matching this CMake alias).
# ---------------------------------------------------------------------------
add_library(speech_io STATIC src/audio.cpp)
add_library(speech::io ALIAS speech_io)

target_include_directories(speech_io
        PUBLIC ${PROJECT_SOURCE_DIR}/include
        PRIVATE src/third_party/dr_libs
        PRIVATE src/vendor/aixlog/include
)
target_link_libraries(speech_io
        PRIVATE miniaudio
        PRIVATE httpp::httpp_core
        PUBLIC speech::dsp
)

# See include/libspeech/export.h. No-op on non-Windows.
target_compile_definitions(speech_io PRIVATE SPEECH_BUILDING_SHARED)

# ---------------------------------------------------------------------------
# speech -- the umbrella library: speech::io, speech::dsp, and
# speech::models, all in one linkable/installable target for consumers (the
# Python bindings, example executables, downstream projects).
#
# speech::speech (the PackageName::PackageName convention many CMake
# packages export) is the "I want everything" entry point: link it to pull
# in speech::io + speech::dsp + speech::models in one go, as opposed to
# linking speech::dsp (etc.) directly when only one piece is needed. Used
# below for the Python bindings and the example executable, both of which
# genuinely need the whole library.
# ---------------------------------------------------------------------------
add_library(speech SHARED src/speech_umbrella_placeholder.cpp)
add_library(speech::speech ALIAS speech)

set_target_properties(speech PROPERTIES LINKER_LANGUAGE CXX)

# See libspeech_set_origin_rpath's definition/comment near the top of this
# file for why this is needed.
libspeech_set_origin_rpath(speech)

# On Linux/macOS, a shared library (.so/.dylib) exports every global symbol
# by default, so bundling speech::io + speech::dsp + speech::models into
# `speech` just works with no extra effort. Windows DLLs are the opposite:
# MSVC exports *nothing* unless a symbol is explicitly marked dllexport (or
# listed in a .def file) -- and with zero exported symbols, MSVC still
# happily builds speech.dll, but never writes a companion speech.lib import
# library at all, so anything linking against speech::speech fails with
# LNK1104 ("cannot open file 'speech.lib'").
#
# Every class/function that's actually part of the public API (FFT, STFT,
# Audio, BaseModel, Denoiser, ...) is annotated SPEECH_API in its header
# (see include/libspeech/export.h) and compiled with SPEECH_BUILDING_SHARED
# defined (set on speech_dsp/speech_io/speech_models above), which expands
# that to __declspec(dllexport) on Windows. The other half of the same
# macro needs SPEECH_SHARED defined when a *consumer* includes those
# headers, so it sees __declspec(dllimport) instead -- INTERFACE here means
# anyone who links speech::speech (the example executable, Python bindings)
# gets that define automatically, without speech.dll's own build seeing it.
target_compile_definitions(speech INTERFACE SPEECH_SHARED)

# SPEECH_API alone (above) turned out not to be sufficient on MSVC, and the
# reason is a separate, more fundamental linker behavior than "symbols
# aren't marked dllexport": a static library (.lib) is an *archive* --
# fundamentally different from a plain set of object files. When linking
# a static archive into another target, the linker only pulls in whichever
# .obj members are actually needed to resolve some already-pending
# undefined-symbol reference; every unreferenced member is left out of the
# link entirely. speech_umbrella_placeholder.cpp -- this DLL's only own
# source -- calls into nothing, so nothing in speech_dsp.lib/speech_io.lib/
# speech_models.lib is "needed" from the DLL's own perspective, and MSVC's
# linker never even looks inside them, dllexport annotations and all: an
# annotation on a symbol the linker never pulled into the link in the first
# place has nothing to export. (Non-Windows shared libraries don't have
# this problem here only because every *downstream* consumer -- `example`,
# the Python bindings -- happens to also transitively re-link these same
# static archives directly via speech_dsp/io/models being PUBLIC link
# dependencies of `speech`, and those consumers' own code does reference
# FFT/Audio/etc. symbols, which pulls the right .obj members in on their
# own link line, independent of whatever speech.dll/.so itself exports.)
#
# /WHOLEARCHIVE (MSVC's linker flag) fixes this the targeted way, matching
# in spirit -- not blanket-exporting -- what SPEECH_API is trying to do: it
# forces every object file inside the named static archive into the link
# unconditionally, so the exports our SPEECH_API annotations already
# specify actually end up in the DLL for the linker to see and put in the
# generated speech.lib. It's a no-op define on non-MSVC compilers/linkers.
if(MSVC)
    target_link_options(speech PRIVATE
            "/WHOLEARCHIVE:$<TARGET_FILE_NAME:speech_dsp>"
            "/WHOLEARCHIVE:$<TARGET_FILE_NAME:speech_io>"
            "/WHOLEARCHIVE:$<TARGET_FILE_NAME:speech_models>"
    )
endif()

target_include_directories(speech
        PUBLIC ${PROJECT_SOURCE_DIR}/include          # public headers live under include/libspeech/*
)

target_link_libraries(speech
        PUBLIC speech::io
        PUBLIC speech::dsp
        PUBLIC speech::models
)


find_package(Python3 REQUIRED COMPONENTS Interpreter)

if(DEFINED SKBUILD)
    set(PYTHON_PROJECT_NAME "${SKBUILD_PROJECT_NAME}")
elseif(BUILD_PYTHON)
    set(PYTHON_PROJECT_NAME "${CMAKE_BINARY_DIR}")

    if(NOT PYTHON_REQUIREMENT_INSTALLED)
        # --break-system-packages: modern Debian/Ubuntu (PEP 668) refuse a
        # plain `pip install` into the system Python otherwise. This flag
        # itself requires pip >= 23.0.1 (Jan 2023) to even be recognized --
        # a safe assumption in 2026, but worth knowing if this ever needs
        # to support a much older pip.
        execute_process(
                COMMAND "${Python3_EXECUTABLE}" -m pip install --break-system-packages
                nanobind ninja pytest-xdist pip-tools # build requirements
                OUTPUT_QUIET COMMAND_ERROR_IS_FATAL ANY
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        )
        execute_process(
                COMMAND "${Python3_EXECUTABLE}" -m piptools compile --output-file ${CMAKE_CURRENT_BINARY_DIR}/requirements.txt pyproject.toml
                    --no-emit-options --quiet --no-strip-extras --extra test
                OUTPUT_QUIET COMMAND_ERROR_IS_FATAL ANY
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        )
        execute_process(
                COMMAND "${Python3_EXECUTABLE}" -m pip install --break-system-packages -r ${CMAKE_CURRENT_BINARY_DIR}/requirements.txt
                OUTPUT_QUIET COMMAND_ERROR_IS_FATAL ANY
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        )
        set(PYTHON_REQUIREMENT_INSTALLED TRUE CACHE INTERNAL "Python requirements installed")
    endif()

    execute_process(
            COMMAND "${Python3_EXECUTABLE}" -m nanobind --cmake_dir
            OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    )

    message(STATUS "Found NanoBind at ${NB_DIR}")
    list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
endif()



if(DEFINED PYTHON_PROJECT_NAME)
    find_package(Python 3.8
            REQUIRED COMPONENTS Interpreter Development.Module
            OPTIONAL_COMPONENTS Development.SABIModule)
    find_package(nanobind CONFIG REQUIRED)

    set(NB_MODULE _about)
    nanobind_add_module(${NB_MODULE} STABLE_ABI NB_STATIC ${PROJECT_SOURCE_DIR}/src/binding/about/about.cpp)
    libspeech_set_origin_rpath(${NB_MODULE})

    # __init__.py's cdll.LoadLibrary() calls need the *exact* on-disk
    # filename of each preloaded runtime library (speech itself, httpp,
    # ONNXRuntime) -- these vary by platform (libspeech.so / speech.dll /
    # libspeech.dylib, etc.) and, for ONNXRuntime, by version too. RPATH
    # (see libspeech_set_origin_rpath above) can't help with any of this:
    # it only lets the OS loader resolve a DT_NEEDED entry when one
    # already-loaded .so pulls in another at normal link time. ctypes'
    # cdll.LoadLibrary() is a direct, explicit dlopen()/LoadLibrary() call
    # from Python itself, entirely bypassing that mechanism -- it needs the
    # literal filename handed to it, RPATH or not.
    #
    # Rather than have __init__.py guess that filename per-platform (with
    # if/else branches or fragile globs), compute it once, here, where
    # CMake already knows it precisely, and bake it into _about (already a
    # tiny, dependency-free nanobind module that __init__.py imports first,
    # for __version__) as extra module attributes -- alongside __version__,
    # the same way this file already bakes VERSION_INFO in below.
    # $<TARGET_FILE_NAME:...> is a generator expression, resolved once CMake
    # knows the real output name of each target.
    get_filename_component(_ONNXRUNTIME_LIB_FILENAME "${ONNXRUNTIME_LIB_FILE}" NAME)
    target_compile_definitions(${NB_MODULE} PRIVATE
            VERSION_INFO=${PROJECT_VERSION}
            NB_MODULE_NAME=${NB_MODULE}
            LIB_PATH=$<TARGET_FILE_NAME:speech>
            HTTPP_LIB_PATH=$<TARGET_FILE_NAME:httpp::httpp_core>
            ONNXRUNTIME_LIB_PATH=${_ONNXRUNTIME_LIB_FILENAME})

    # CONFIGURE_DEPENDS: best-effort auto-detection of new bind_* directories
    # without a manual reconfigure. This is NOT guaranteed by CMake to work
    # on every generator/environment (e.g. observed unreliable with the Unix
    # Makefiles generator in some setups) -- if a newly added binding module
    # doesn't show up after `cmake --build`, run `cmake ..` (or delete and
    # recreate the build directory) once to force a fresh configure.
    file(GLOB PYTHON_BIND_MODULES_PATH CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/src/binding/bind_*)
    set(PYTHON_BIND_MODULES "${NB_MODULE}")
    foreach(NB_MODULE_PATH IN LISTS PYTHON_BIND_MODULES_PATH)
        get_filename_component(NB_MODULE ${NB_MODULE_PATH} NAME)
        string(REGEX REPLACE "bind_(.*)" "speech_\\1_py" NB_MODULE ${NB_MODULE})
        list(APPEND PYTHON_BIND_MODULES ${NB_MODULE})
        file(GLOB NB_MODULE_SOURCES CONFIGURE_DEPENDS ${NB_MODULE_PATH}/*.cpp)

        message(STATUS "Found module: ${NB_MODULE}")

        # We are now ready to compile the actual extension module
        nanobind_add_module(
                # Name of the extension
                ${NB_MODULE}

                # Target the stable ABI for Python 3.12+, which reduces
                # the number of binary wheels that must be built. This
                # does nothing on older Python versions
                STABLE_ABI

                # Build libnanobind statically and merge it into the
                # extension (which itself remains a shared library)
                #
                # If your project builds multiple extensions, you can
                # replace this flag by NB_SHARED to conserve space by
                # reusing a shared libnanobind across libraries
                NB_STATIC

                # Source code goes here
                ${NB_MODULE_SOURCES}
        )

        target_include_directories(${NB_MODULE} PRIVATE include src)
        target_compile_definitions(${NB_MODULE} PRIVATE NB_MODULE_NAME=${NB_MODULE})
        target_link_libraries(${NB_MODULE} PRIVATE speech::speech)
        libspeech_set_origin_rpath(${NB_MODULE})
    endforeach()

    if(DEFINED SKBUILD)
        install (TARGETS ${PYTHON_BIND_MODULES} speech
                CONFIGURATIONS Release
                LIBRARY DESTINATION ${PYTHON_PROJECT_NAME})

        # ONNXRuntime ships as a shared library that the compiled extension
        # modules dlopen symbols from at import time (see __init__.py's
        # cdll.LoadLibrary() calls), so it has to be copied into the wheel
        # next to those extensions rather than relied upon via RPATH:
        # RPATH would only point back at wherever it got downloaded to on
        # the *build* machine, which has no reason to exist on whatever
        # machine later installs this wheel.
        #
        # httpp is NOT bundled here -- __init__.py loads it from the
        # separately pip-installed httpp package instead (see __init__.py).
        #
        # ONNXRUNTIME_SONAME_FILE (see cmake/ONNXRuntime.cmake) additionally
        # ships a copy under onnxruntime's actual SONAME (libonnxruntime.so.1)
        # on Linux/macOS -- every compiled .so's DT_NEEDED entry references
        # that name, not the fully-versioned ONNXRUNTIME_LIB_FILE one, and
        # without it `auditwheel repair` fails to find/bundle it at all.
        # ONNXRUNTIME_PROVIDERS_SHARED_FILE (Windows only) is the companion
        # onnxruntime_providers_shared.dll upstream ships alongside
        # onnxruntime.dll -- see cmake/ONNXRuntime.cmake for why.
        set(_LIBSPEECH_RUNTIME_LIBS "${ONNXRUNTIME_LIB_FILE}")
        if(ONNXRUNTIME_SONAME_FILE)
            list(APPEND _LIBSPEECH_RUNTIME_LIBS "${ONNXRUNTIME_SONAME_FILE}")
        endif()
        if(ONNXRUNTIME_PROVIDERS_SHARED_FILE)
            list(APPEND _LIBSPEECH_RUNTIME_LIBS "${ONNXRUNTIME_PROVIDERS_SHARED_FILE}")
        endif()
        install (FILES ${_LIBSPEECH_RUNTIME_LIBS}
                DESTINATION ${SKBUILD_PROJECT_NAME})
    else()
        file(COPY ${PROJECT_SOURCE_DIR}/src/${PROJECT_NAME} DESTINATION ${PYTHON_PROJECT_NAME})
        file(COPY ${ONNXRUNTIME_LIB_FILE} DESTINATION ${PYTHON_PROJECT_NAME}/${PROJECT_NAME})
        if(ONNXRUNTIME_SONAME_FILE)
            file(COPY ${ONNXRUNTIME_SONAME_FILE} DESTINATION ${PYTHON_PROJECT_NAME}/${PROJECT_NAME})
        endif()
        if(ONNXRUNTIME_PROVIDERS_SHARED_FILE)
            file(COPY ${ONNXRUNTIME_PROVIDERS_SHARED_FILE} DESTINATION ${PYTHON_PROJECT_NAME}/${PROJECT_NAME})
        endif()

        foreach(_core IN LISTS PYTHON_BIND_MODULES)
            set_target_properties(${_core} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PYTHON_PROJECT_NAME}/${PROJECT_NAME})
        endforeach()

        # speech itself must land inside the package directory too --
        # __init__.py's `cdll.LoadLibrary(_here / LIB_PATH)` looks for
        # it right next to _audio.abi3.so/_about.abi3.so, not in the
        # top-level build directory (where it would otherwise default to).
        set_target_properties(speech PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PYTHON_PROJECT_NAME}/${PROJECT_NAME})
    endif()
endif()

# Python tests need the bindings fully configured above (the _audio target,
# ${PYTHON_PROJECT_NAME}) -- and only make sense for a local dev build, not
# while scikit-build-core is driving an actual `pip install .` package build.
if(BUILD_TESTS AND DEFINED PYTHON_PROJECT_NAME AND NOT DEFINED SKBUILD)
    add_subdirectory(tests/python)
endif()

if(DEFINED SKBUILD)
    RETURN()
endif()

add_executable(example examples/main.cpp)
target_link_libraries(example PRIVATE speech::speech)

# See the matching comment in tests/models/CMakeLists.txt (the same DLL-
# placement gap, applied to the C++ speech::speech example instead of the
# models test executable).
if(WIN32)
    add_custom_command(TARGET example POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${ONNXRUNTIME_LIB_FILE}" "$<TARGET_FILE:httpp::httpp_core>"
                    "$<TARGET_FILE_DIR:example>")
endif()

install(TARGETS speech
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    INCLUDES DESTINATION include
)
