# Check if conanfile.txt exists
if(EXISTS "${CMAKE_SOURCE_DIR}/conanfile.txt")
    message(STATUS "Found conanfile.txt in the project directory.")

    # Set the path to the Conan-generated toolchain file
    set(CONAN_TOOLCHAIN_FILE "${CMAKE_BINARY_DIR}/conan_toolchain.cmake")

    # Check if the Conan toolchain file already exists
    if(NOT EXISTS "${CONAN_TOOLCHAIN_FILE}")
        message(STATUS "Conan toolchain file not found. Running 'conan install'...")

        # Run Conan to install dependencies and generate the toolchain file
        execute_process(
                COMMAND conan install ${CMAKE_SOURCE_DIR} --output-folder=${CMAKE_BINARY_DIR} --build=missing
                RESULT_VARIABLE conan_result
        )

        # Check if Conan installation was successful
        if(NOT conan_result EQUAL 0)
            message(FATAL_ERROR "Conan installation failed with error code: ${conan_result}")
        endif()

        message(STATUS "Conan installation completed successfully.")
    else()
        message(STATUS "Conan toolchain file already exists: ${CONAN_TOOLCHAIN_FILE}")
    endif()

    # Set the CMAKE_TOOLCHAIN_FILE to the Conan-generated toolchain file
    set(CMAKE_TOOLCHAIN_FILE "${CONAN_TOOLCHAIN_FILE}" CACHE PATH "Path to Conan-generated toolchain file" FORCE)
else()
    message(WARNING "conanfile.txt not found in the project directory. Skipping Conan setup.")
endif()