# speech::models C++ unit tests -- see tests/CMakeLists.txt for the full
# speech_test_<language>_<module> naming scheme this fits into. Kept
# separate from speech_test_cpp_dsp since these do need the full
# ONNXRuntime/httpp stack.

file(GLOB SPEECH_MODELS_TEST_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

add_executable(test_cpp_models
    ${SPEECH_MODELS_TEST_SOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/../utest/utest_main.cxx
)
target_include_directories(test_cpp_models PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../utest)
target_link_libraries(test_cpp_models PRIVATE speech_models)

add_test(NAME cpp_models COMMAND test_cpp_models)

# speech_models links onnxruntime + httpp as *shared* libraries (DLLs on
# Windows). On Linux/macOS this just works for a plain build-tree
# executable like this one: CMake bakes the real (absolute, build-machine)
# path into the binary's build-tree RPATH by default, so the dynamic
# linker finds them without any extra help. Windows has no RPATH
# equivalent at all -- its loader only ever looks in the executable's own
# directory, system directories, or PATH -- so without a DLL actually
# sitting next to test_cpp_models.exe, the process can't resolve its own
# imports and never gets past process startup. Critically, this doesn't
# fail fast: on a non-interactive CI runner, an unhandled Windows startup
# failure like this triggers a Windows Error Reporting dialog that never
# gets dismissed, so the process (and the `ctest` run waiting on it) just
# hangs until something else kills it -- which is exactly the "test #1
# (cpp_dsp, no onnxruntime/httpp dependency) passes in 0.08s, test #2
# (cpp_models) hangs for hours" symptom this fixes.
if(WIN32)
    add_custom_command(TARGET test_cpp_models POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${ONNXRUNTIME_LIB_FILE}" "$<TARGET_FILE:httpp::httpp_core>"
                    "$<TARGET_FILE_DIR:test_cpp_models>")
endif()

add_custom_target(speech_test_cpp_models
    COMMAND ${CMAKE_COMMAND} -E echo "-- Running speech::models C++ tests..."
    COMMAND $<TARGET_FILE:test_cpp_models>
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    USES_TERMINAL
)
add_dependencies(speech_test_cpp_models test_cpp_models)
add_dependencies(speech_test_cpp speech_test_cpp_models)
