cmake_minimum_required(VERSION 3.19)

option(BUILD_PYTHON "export python module" off)
option(BUILD_TESTS "build unit tests (speech::dsp only, no ONNXRuntime/httplib needed)" on)
option(BUILD_MODELS "build speech::models + the speech library (needs ONNXRuntime download + mbedtls build); off configures speech::dsp + tests only" on)

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

message(STATUS "Project: ${PROJECT_NAME}@v${PROJECT_VERSION}")

include(AudioFlux)

# ---------------------------------------------------------------------------
# speech::dsp -- resample, window, FFT, STFT, MFCC, ...
#
# Deliberately independent of ONNXRuntime/httplib/miniaudio: this lets DSP
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
add_subdirectory(src/third_party/indicators)

# ---------------------------------------------------------------------------
# Model downloading (speech::utils::downloadFile) used to link the SYSTEM
# libcurl (find_package(CURL REQUIRED) + apt-installed libcurl4-openssl-dev)
# -- a real system dependency the "pip install and you're done" story
# doesn't want, and one this project's own build has hit in practice (a
# sandboxed CI environment without apt access to security.ubuntu.com failed
# here). Replaced with httplib.h (single header, vendored under
# src/vendor/httplib/) + Mbed TLS (a git submodule, NOT vendored/patched
# like AudioFlux -- crypto code should stay pristine and upstream-
# updatable) built from source, so no system TLS/HTTP library needs to be
# pre-installed.
# ---------------------------------------------------------------------------
set(ENABLE_PROGRAMS OFF CACHE BOOL "" FORCE)
set(ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(DISABLE_PACKAGE_CONFIG_AND_INSTALL ON CACHE BOOL "" FORCE)
add_subdirectory(src/third_party/mbedtls)

# ---------------------------------------------------------------------------
# speech::models -- BaseModel/ONNXModel and the VAD/denoiser backends.
#
# Separate from speech::dsp (no shared code) and from the main `speech`
# library (no audio-file-I/O dependency): models take raw float buffers in
# and out, and only need ONNXRuntime plus the download/progress-bar utility
# code (base_model.cpp downloads .onnx weights on first use).
# ---------------------------------------------------------------------------
file(GLOB SPEECH_MODELS_SOURCES CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/src/models/*.cpp")
add_library(speech_models STATIC
        ${SPEECH_MODELS_SOURCES}
        src/utils/utils.cpp
        src/utils/progressbar.cpp
)
add_library(speech::models ALIAS speech_models)

target_include_directories(speech_models
        PUBLIC ${PROJECT_SOURCE_DIR}/include
        PUBLIC src/vendor/aixlog/include
        PRIVATE ${PROJECT_SOURCE_DIR}/src   # for utils/progressbar.h
        PRIVATE src/vendor/httplib
)
target_compile_definitions(speech_models PRIVATE CPPHTTPLIB_MBEDTLS_SUPPORT)
target_link_libraries(speech_models
        PUBLIC onnxruntime_interface
        PUBLIC indicators
        PRIVATE mbedtls mbedx509 mbedcrypto
)

# ---------------------------------------------------------------------------
# speech -- the umbrella library: audio file I/O (miniaudio/dr_libs) plus
# speech::dsp and speech::models, all in one linkable/installable target for
# consumers (the Python bindings, example executables, downstream projects).
# ---------------------------------------------------------------------------
add_library(speech SHARED src/audio.cpp)
add_library(speech::speech ALIAS speech)

target_include_directories(speech
        PUBLIC ${PROJECT_SOURCE_DIR}/include          # public headers live under include/libspeech/*
        PRIVATE ${PROJECT_SOURCE_DIR}/src
        PRIVATE src/third_party/dr_libs
)

target_link_libraries(speech
        PRIVATE miniaudio
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
    target_compile_definitions(${NB_MODULE} PRIVATE
            VERSION_INFO=${PROJECT_VERSION}
            NB_MODULE_NAME=${NB_MODULE})

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
        string(REGEX REPLACE "bind(.*)" "\\1" NB_MODULE ${NB_MODULE})
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
        target_link_libraries(${NB_MODULE} PRIVATE speech)
    endforeach()

    if(DEFINED SKBUILD)
        install (TARGETS ${PYTHON_BIND_MODULES} speech
                CONFIGURATIONS Release
                LIBRARY DESTINATION ${PYTHON_PROJECT_NAME})

        install (FILES ${ONNXRUNTIME_LIB_FILE}
                DESTINATION ${SKBUILD_PROJECT_NAME})
    else()
        file(COPY ${PROJECT_SOURCE_DIR}/src/${PROJECT_NAME} DESTINATION ${PYTHON_PROJECT_NAME})
        file(COPY ${ONNXRUNTIME_LIB_FILE} DESTINATION ${PYTHON_PROJECT_NAME}/${PROJECT_NAME})

        foreach(_core IN LISTS PYTHON_BIND_MODULES)
            set_target_properties(${_core} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PYTHON_PROJECT_NAME}/${PROJECT_NAME})
        endforeach()

        # libspeech.so itself must land inside the package directory too --
        # __init__.py's `cdll.LoadLibrary(_here / "libspeech.so")` looks for
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
target_link_libraries(example PRIVATE speech)

install(TARGETS speech
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    INCLUDES DESTINATION include
)
