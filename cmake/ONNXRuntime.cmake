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
    if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        # Microsoft's ONNX Runtime release assets use "arm64" for macOS
        # (e.g. onnxruntime-osx-arm64-<ver>.tgz), unlike Linux which uses
        # "aarch64" (e.g. onnxruntime-linux-aarch64-<ver>.tgz).
        set(ARCH_SUFFIX "arm64")
    else()
        set(ARCH_SUFFIX "aarch64")
    endif()
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
set(ONNXRUNTIME_DIR "${PROJECT_SOURCE_DIR}/src/third_party/onnxruntime")

# Check if ONNX Runtime is already downloaded and valid
if(EXISTS "${ONNXRUNTIME_DIR}")
    # Check if the required library file exists based on the current OS
    if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        set(ONNXRUNTIME_LIB_FILE "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll")
    else()
        set(ONNXRUNTIME_LIB_FILE "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so.${onnx_version}")
    endif()

    # If the required library file does not exist, remove the directory
    if(NOT EXISTS "${ONNXRUNTIME_LIB_FILE}")
        message(STATUS "ONNX Runtime directory exists but the required library file is missing. Removing '${ONNXRUNTIME_DIR}'...")
        file(REMOVE_RECURSE "${ONNXRUNTIME_DIR}")
    endif()
endif()

# If ONNX Runtime directory does not exist or was removed, download it
if(NOT EXISTS "${ONNXRUNTIME_DIR}")
    message(STATUS "ONNX Runtime not found in '${ONNXRUNTIME_DIR}'. Downloading from '${ONNXRUNTIME_URL}'...")

    # Download the file directly into the parent directory
    file(DOWNLOAD ${ONNXRUNTIME_URL} "${PROJECT_SOURCE_DIR}/src/third_party/onnxruntime${FILE_EXT}"
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
                COMMAND powershell -Command "Expand-Archive -Path \"${PROJECT_SOURCE_DIR}/src/third_party/onnxruntime${FILE_EXT}\" -DestinationPath \"${PROJECT_SOURCE_DIR}/src/third_party\""
                WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/src/third_party"
        )
    else()
        execute_process(
                COMMAND ${CMAKE_COMMAND} -E tar xzf "${PROJECT_SOURCE_DIR}/src/third_party/onnxruntime${FILE_EXT}"
                WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/src/third_party"
        )
    endif()

    # Rename the extracted folder to 'onnxruntime'
    file(GLOB EXTRACTED_DIR "${PROJECT_SOURCE_DIR}/src/third_party/onnxruntime-*")
    if(EXISTS "${EXTRACTED_DIR}" AND IS_DIRECTORY "${EXTRACTED_DIR}")
        file(RENAME "${EXTRACTED_DIR}" "${ONNXRUNTIME_DIR}")
    else()
        message(FATAL_ERROR "Failed to find the extracted ONNX Runtime directory.")
    endif()

    # Remove the downloaded archive after extraction
    file(REMOVE "${PROJECT_SOURCE_DIR}/src/third_party/onnxruntime${FILE_EXT}")
else()
    message(STATUS "ONNX Runtime already exists in '${ONNXRUNTIME_DIR}'. Skipping download.")
endif()

# Determine the runtime library file path -- unconditionally, regardless of
# whether we just downloaded it above or it already existed. This used to
# live only inside the "already exists" branch above, which meant a
# completely fresh (first-ever) download never set this variable at all,
# silently breaking the Python package's runtime library copy step later
# (file(COPY "${ONNXRUNTIME_LIB_FILE}" ...) with an empty path is a no-op,
# not an error) -- exactly the scenario every new contributor or a CI cache
# miss hits.
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(ONNXRUNTIME_LIB_FILE "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll")
else()
    set(ONNXRUNTIME_LIB_FILE "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so.${onnx_version}")

    # The official Linux/macOS release tarballs extract a symlink chain:
    # libonnxruntime.so -> libonnxruntime.so.1 -> libonnxruntime.so.1.21.0
    # (the last being ONNXRUNTIME_LIB_FILE above). Everything we build
    # links against "-lonnxruntime", which the linker resolves via that
    # chain at build time -- but only the fully-versioned real file
    # (ONNXRUNTIME_LIB_FILE) was ever installed/copied into the wheel /
    # local package dir, never the middle symlink. That middle name is the
    # one that actually matters at runtime: it's the library's SONAME
    # (embedded in the .so itself, confirmed via `readelf -d
    # libonnxruntime.so.1.21.0 | grep SONAME` -> "libonnxruntime.so.1"),
    # i.e. the exact name every consumer .so's DT_NEEDED entry references
    # and the name the dynamic linker (or auditwheel, when it verifies a
    # wheel bundles everything it needs) looks for -- not the fully-
    # versioned filename. Without shipping a file by this exact name,
    # `auditwheel repair` fails with 'required library
    # "libonnxruntime.so.1" could not be located', even though the actual
    # library bytes are right there under a different name.
    set(ONNXRUNTIME_SONAME_FILE "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so.1")
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
# Add _CRT_SECURE_NO_WARNINGS as a compile definition
target_compile_definitions(onnxruntime_interface INTERFACE _CRT_SECURE_NO_WARNINGS)

set(CMAKE_SUPPRESS_REGENERATION ON)