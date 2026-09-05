# speech::models C++ unit tests -- see tests/CMakeLists.txt for the full
# speech_test_<language>_<module> naming scheme this fits into. Kept
# separate from speech_test_cpp_dsp since these do need the full
# ONNXRuntime/httplib/mbedtls stack.

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

add_custom_target(speech_test_cpp_models
    COMMAND ${CMAKE_COMMAND} -E echo "-- Running speech::models C++ tests..."
    COMMAND $<TARGET_FILE:test_cpp_models>
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    USES_TERMINAL
)
add_dependencies(speech_test_cpp_models test_cpp_models)
add_dependencies(speech_test_cpp speech_test_cpp_models)
