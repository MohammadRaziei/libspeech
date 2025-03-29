# AudioFlux CMake Module
#
# This module finds and configures the AudioFlux library.
# It assumes that AudioFlux is located in the `src/third_party/audioflux` directory.

# Define the path to the AudioFlux source directory
set(AUDIOFLUX_SOURCE_DIR "${CMAKE_SOURCE_DIR}/src/third_party/audioflux")

# Source files for AudioFlux
file(GLOB_RECURSE AUDIOFLUX_SOURCES
        "${AUDIOFLUX_SOURCE_DIR}/src/*.c"
)

# Add a static or shared library for AudioFlux
add_library(audioflux STATIC ${AUDIOFLUX_SOURCES})

# Set include directories for the AudioFlux library
#target_include_directories(audioflux PUBLIC ${AUDIOFLUX_INCLUDE_DIRS})
target_include_directories(audioflux
        PUBLIC  ${AUDIOFLUX_SOURCE_DIR}/include
        PRIVATE ${AUDIOFLUX_SOURCE_DIR}/src
)

target_compile_options(audioflux PRIVATE "-w" "-fPIC")

# Export the AudioFlux library for use in other parts of the project
set(AUDIOFLUX_FOUND TRUE)
set(AUDIOFLUX_LIBRARIES audioflux)