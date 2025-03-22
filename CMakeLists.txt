cmake_minimum_required(VERSION 3.19)

option(BUILD_PYTHON "export python module" off)

set(CMAKE_C_STANDARD 99 CACHE STRING "C version selection")
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ version selection")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
include(PyProject)


if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()



#set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}" CACHE STRING "" FORCE)
#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}" CACHE STRING "" FORCE)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "" FORCE)


project(${PyProject_NAME} LANGUAGES C CXX VERSION ${PyProject_VERSION})

message(STATUS "Project: ${PROJECT_NAME}@v${PROJECT_VERSION}")

file(GLOB SPEECH_ROOT_SOURCES "${PROJECT_SOURCE_DIR}/src/*.cpp")
file(GLOB SPEECH_SUBDIR_SOURCES "${PROJECT_SOURCE_DIR}/src/*/*.cpp")

set(SPEECH_SOURCES ${SPEECH_ROOT_SOURCES} ${SPEECH_SUBDIR_SOURCES})



include(ONNXRuntime)
find_package(CURL REQUIRED)
include(AudioFlux)


#add_subdirectory(src/third_party/dr_libs)
add_subdirectory(src/third_party/miniaudio)
add_subdirectory(src/third_party/indicators)
#add_subdirectory(src/third_party/aixlog)

add_library(speech SHARED ${SPEECH_SOURCES})


target_include_directories(speech
        PUBLIC ${PROJECT_SOURCE_DIR}/include
        PRIVATE ${PROJECT_SOURCE_DIR}/src
        PRIVATE src/third_party/dr_libs
        PUBLIC src/third_party/aixlog/include
)

target_link_libraries(speech
        PRIVATE miniaudio
        PRIVATE CURL::libcurl
        PUBLIC indicators
        PUBLIC onnxruntime_interface
        PUBLIC audioflux
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
    # Try to import all Python components potentially needed by nanobind
    #set(Python3_FIND_STRATEGY LOCATION)
    find_package(Python 3.8
            REQUIRED COMPONENTS Interpreter Development.Module
            OPTIONAL_COMPONENTS Development.SABIModule)
    # Import nanobind through CMake's find_package mechanism
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

        # Add more commands here to process each directory as needed

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
#        target_compile_definitions(${NB_MODULE} PRIVATE
#                DONT_INIT_PSL
#                PUBLIC_SUFFIX_LIST_URL="${PUBLIC_SUFFIX_LIST_URL}"
#                PUBLIC_SUFFIX_LIST_DAT="${PUBLIC_SUFFIX_LIST_DAT}"
#        )
    endforeach()


#    get_target_property(SPEECH_LIBS speech LINK_LIBRARIES)
    set(SPEECH_LIBS miniaudio indicators onnxruntime_interface audioflux)

    if(DEFINED SKBUILD)


        install (TARGETS ${PYTHON_BIND_MODULES} speech audioflux
                CONFIGURATIONS Release
                LIBRARY DESTINATION ${PYTHON_PROJECT_NAME})

        install (FILES ${ONNXRUNTIME_LIB_FILE}
                DESTINATION ${SKBUILD_PROJECT_NAME}/${PROJECT_NAME})

        file(COPY ${ONNXRUNTIME_LIB_FILE} DESTINATION ${PYTHON_PROJECT_NAME}/${PROJECT_NAME} )

        #        file(COPY ${ONNXRUNTIME_LIB_FILE} DESTINATION ${PYTHON_PROJECT_NAME}/${PROJECT_NAME})

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




#add_executable(silero examples/silero/silero-vad-onnx.cpp)
#target_link_libraries(silero PRIVATE speech)



add_executable(example examples/main.cpp)
target_link_libraries(example PRIVATE speech)



# add_subdirectory(src/third_party/crow)

set(CMAKE_TOOLCHAIN_FILE "${CMAKE_BINARY_DIR}/conan_toolchain.cmake" CACHE PATH "Path to Conan-generated toolchain file")

# Include the toolchain file if it exists
if(EXISTS "${CMAKE_TOOLCHAIN_FILE}")
    message(STATUS "Using Conan-generated toolchain file: ${CMAKE_TOOLCHAIN_FILE}")
else()
    message(WARNING "Conan-generated toolchain file not found: ${CMAKE_TOOLCHAIN_FILE}")
endif()

#include(Conan)
#
#find_package(oatpp REQUIRED)
#find_package(oatpp-swagger REQUIRED)


#add_executable(denoiser_app examples/denoiser_app/main.cpp)
#target_link_libraries(denoiser_app PRIVATE oatpp::oatpp oatpp::oatpp-swagger)

#
#add_executable(silero_app examples/silero_app/main.cpp)
#target_link_libraries(silero_app PRIVATE speech oatpp::oatpp oatpp::oatpp-swagger)
#


# find_package(Vorbis REQUIRED)
# target_link_libraries(example PRIVATE Vorbis::Vorbis)

install(TARGETS speech
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    INCLUDES DESTINATION include
)



