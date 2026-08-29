cmake_minimum_required(VERSION 3.19)

option(BUILD_PYTHON "export python module" off)
option(BUILD_TESTS "build unit tests (speech::dsp only, no ONNXRuntime/httplib needed)" on)
option(SPEECH_DSP_ONLY "only configure speech::dsp + its tests; skip ONNXRuntime/models/audio I/O" off)

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
include(PyProject)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

project(${PyProject_NAME} LANGUAGES C CXX VERSION ${PyProject_VERSION})

message(STATUS "Project: ${PROJECT_NAME}@v${PROJECT_VERSION}")

include(AudioFlux)

# ---------------------------------------------------------------------------
# speech::dsp -- resample, window, FFT, STFT, MFCC, ...
#
# Deliberately independent of ONNXRuntime/httplib/miniaudio: this lets DSP
# unit tests (tests/CMakeLists.txt) build and run without pulling in the
# model/audio-I/O stack, matching the project's "minimal dependencies" goal
# and keeping DSP iteration fast. See SPEECH_DSP_ONLY below.
# ---------------------------------------------------------------------------
file(GLOB SPEECH_DSP_SOURCES "${PROJECT_SOURCE_DIR}/src/dsp/*.cpp")
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

if(SPEECH_DSP_ONLY)
    message(STATUS "SPEECH_DSP_ONLY=on: skipping ONNXRuntime/models/audio I/O.")
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
file(GLOB SPEECH_MODELS_SOURCES "${PROJECT_SOURCE_DIR}/src/models/*.cpp")
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
        execute_process(
                COMMAND "${Python3_EXECUTABLE}" -m pip install
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
                COMMAND "${Python3_EXECUTABLE}" -m pip install -r ${CMAKE_CURRENT_BINARY_DIR}/requirements.txt
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

    file(GLOB PYTHON_BIND_MODULES_PATH ${PROJECT_SOURCE_DIR}/src/binding/bind_*)
    set(PYTHON_BIND_MODULES "${NB_MODULE}")
    foreach(NB_MODULE_PATH IN LISTS PYTHON_BIND_MODULES_PATH)
        get_filename_component(NB_MODULE ${NB_MODULE_PATH} NAME)
        string(REGEX REPLACE "bind(.*)" "\\1" NB_MODULE ${NB_MODULE})
        list(APPEND PYTHON_BIND_MODULES ${NB_MODULE})
        file(GLOB NB_MODULE_SOURCES ${NB_MODULE_PATH}/*.cpp)

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
                ${NB_MODULE_SOURCES} ${SOURCES}
        )

        target_include_directories(${NB_MODULE} PRIVATE include src)
        target_compile_definitions(${NB_MODULE} PRIVATE NB_MODULE_NAME=${NB_MODULE})
        target_link_directories(${NB_MODULE} PRIVATE speech)
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
    endif()
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
