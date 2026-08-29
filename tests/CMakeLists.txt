# DSP unit tests, built with utest.h (see tests/utest/) against the
# lightweight `speech_dsp` target -- deliberately independent of the full
# `speech` library's ONNXRuntime/CURL/miniaudio dependencies, so these build
# and run fast with nothing but a C/C++ compiler.

function(add_dsp_test name)
    add_executable(test_${name} dsp/test_${name}.cpp)
    target_include_directories(test_${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/utest)
    target_link_libraries(test_${name} PRIVATE speech_dsp)
    add_test(NAME ${name} COMMAND test_${name})
endfunction()

add_dsp_test(resample)
add_dsp_test(window)
add_dsp_test(fft)
add_dsp_test(stft)
add_dsp_test(dct)
add_dsp_test(mfcc)
