include(FetchContent)

# Keep the fetched build minimal and fast: only speech::dsp is needed for
# this benchmark, so skip ONNXRuntime/mbedtls/models and the test suite
# entirely.
set(BUILD_MODELS OFF CACHE BOOL "" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    libspeech
    GIT_REPOSITORY ${LIBSPEECH_GITHUB_URL}
    GIT_TAG        master
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(libspeech)

add_executable(bench_stft bench_stft.cpp)
target_link_libraries(bench_stft PRIVATE speech::dsp)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(bench_stft PRIVATE -O3)
endif()

add_custom_target(libspeech_benchmarks_cpp
    COMMAND ${CMAKE_COMMAND} -E echo "-- Running C++ STFT benchmark..."
    COMMAND $<TARGET_FILE:bench_stft>
    DEPENDS bench_stft
    USES_TERMINAL
)
add_dependencies(libspeech_benchmarks libspeech_benchmarks_cpp)
