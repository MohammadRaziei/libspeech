# Options for GPU and ONNX Runtime version
option(USE_GPU "Enable GPU support for ONNX Runtime" OFF)
set(onnx_version "1.21.0" CACHE STRING "ONNX Runtime version to download")

# Determine the operating system
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(OS_PREFIX "linux")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(OS_PREFIX "osx")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(OS_PREFIX "win")
else()
    message(FATAL_ERROR "Unsupported operating system: ${CMAKE_SYSTEM_NAME}")
endif()

# Determine the architecture
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    set(ARCH_SUFFIX "aarch64")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "AMD64")
    set(ARCH_SUFFIX "x64")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "i386" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "i686")
    set(ARCH_SUFFIX "x86")
else()
    message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

# Determine the file extension based on OS
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(FILE_EXT ".zip")
else()
    set(FILE_EXT ".tgz")
endif()

# Construct the URL based on options
if(USE_GPU)
    set(BUILD_TYPE "-gpu")
else()
    set(BUILD_TYPE "")
endif()

set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${onnx_version}/onnxruntime-${OS_PREFIX}-${ARCH_SUFFIX}${BUILD_TYPE}-${onnx_version}${FILE_EXT}")

# Define the target directory for ONNX Runtime
set(ONNXRUNTIME_DIR "${CMAKE_SOURCE_DIR}/src/third_party/onnxruntime")

# Check if ONNX Runtime is already downloaded
if(NOT EXISTS "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so" AND NOT EXISTS "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll")
    message(STATUS "ONNX Runtime not found in '${ONNXRUNTIME_DIR}'. Downloading from '${ONNXRUNTIME_URL}'...")

    # Download the file directly into the parent directory
    file(DOWNLOAD ${ONNXRUNTIME_URL} "${CMAKE_SOURCE_DIR}/src/third_party/onnxruntime${FILE_EXT}"
            SHOW_PROGRESS
            STATUS DOWNLOAD_STATUS)

    # Check if the download was successful
    list(GET DOWNLOAD_STATUS 0 STATUS_CODE)
    if(NOT STATUS_CODE EQUAL 0)
        message(FATAL_ERROR "Failed to download ONNX Runtime from '${ONNXRUNTIME_URL}'")
    endif()

    # Extract the downloaded archive into the parent directory
    if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        execute_process(
                COMMAND powershell -Command "Expand-Archive -Path \"${CMAKE_SOURCE_DIR}/src/third_party/onnxruntime${FILE_EXT}\" -DestinationPath \"${CMAKE_SOURCE_DIR}/src/third_party\""
                WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/src/third_party"
        )
    else()
        execute_process(
                COMMAND ${CMAKE_COMMAND} -E tar xzf "${CMAKE_SOURCE_DIR}/src/third_party/onnxruntime${FILE_EXT}"
                WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/src/third_party"
        )
    endif()

    # Rename the extracted folder to 'onnxruntime'
    file(GLOB EXTRACTED_DIR "${CMAKE_SOURCE_DIR}/src/third_party/onnxruntime-*")
    if(EXISTS "${EXTRACTED_DIR}" AND IS_DIRECTORY "${EXTRACTED_DIR}")
        file(RENAME "${EXTRACTED_DIR}" "${ONNXRUNTIME_DIR}")
    else()
        message(FATAL_ERROR "Failed to find the extracted ONNX Runtime directory.")
    endif()

    # Remove the downloaded archive after extraction
    file(REMOVE "${CMAKE_SOURCE_DIR}/src/third_party/onnxruntime${FILE_EXT}")
else()
    message(STATUS "ONNX Runtime already exists in '${ONNXRUNTIME_DIR}'. Skipping download.")
endif()

# Create a .gitignore file in the ONNX Runtime directory
file(WRITE "${ONNXRUNTIME_DIR}/.gitignore" "*\n")


# Define an interface library for ONNX Runtime
add_library(onnxruntime_interface INTERFACE)

# Add include directories to the interface
target_include_directories(onnxruntime_interface INTERFACE "${ONNXRUNTIME_DIR}/include")

# Add link directories to the interface
target_link_directories(onnxruntime_interface INTERFACE "${ONNXRUNTIME_DIR}/lib")

# Link the ONNX Runtime library to the interface
target_link_libraries(onnxruntime_interface INTERFACE onnxruntime)
