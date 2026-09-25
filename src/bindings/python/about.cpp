//
// Created by mohammad on 10/17/24.
//

#include <nanobind/nanobind.h>

#include <string>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)


NB_MODULE(NB_MODULE_NAME, m) {
#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

    // Exact on-disk filenames (not full paths -- just the filename,
    // resolved next to this very module once installed) of the runtime
    // libraries __init__.py preloads via cdll.LoadLibrary(): speech
    // itself, httpp, and ONNXRuntime. These vary by platform (e.g.
    // libspeech.so / speech.dll / libspeech.dylib) and, for ONNXRuntime,
    // by version too -- CMake computes the real name for each one (see
    // the target_compile_definitions() call on this module's target in
    // CMakeLists.txt) instead of __init__.py guessing it per-platform.
    m.attr("LIB_PATH") = MACRO_STRINGIFY(LIB_PATH);
    m.attr("HTTPP_LIB_PATH") = MACRO_STRINGIFY(HTTPP_LIB_PATH);
    m.attr("ONNXRUNTIME_LIB_PATH") = MACRO_STRINGIFY(ONNXRUNTIME_LIB_PATH);
}
