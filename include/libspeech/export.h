//
// SPEECH_API -- marks a class or free function as part of libspeech's
// public API surface, i.e. reachable by code outside the `speech` shared
// library (speech.dll / libspeech.so / libspeech.dylib).
//
// Why this exists: on Linux/macOS, a shared library exports every global
// symbol by default, so nothing special is needed there. Windows DLLs are
// the opposite -- MSVC exports *nothing* from a DLL unless a symbol is
// explicitly decorated, so every class that crosses the `speech` DLL
// boundary (speech::dsp::FFT, speech::models::BaseModel, ...) needs:
//   - __declspec(dllexport) when compiling the .cpp that defines it
//     (i.e. as part of the speech_dsp/speech_io/speech_models static
//     libraries that get bundled into the speech DLL), and
//   - __declspec(dllimport) when a consumer merely includes the header
//     and links against the already-built speech.dll.
// Which one applies is selected by which of SPEECH_BUILDING_SHARED /
// SPEECH_SHARED the build sets (see CMakeLists.txt), never by hand.
//
// Apply this at class level (`class SPEECH_API Foo { ... };`), not per
// member function -- that's what correctly exports the vtable/RTTI for
// polymorphic classes (BaseModel, Denoiser, ONNXModel, ...) too, which
// per-method annotation would miss.
//

#ifndef LIBSPEECH_EXPORT_H
#define LIBSPEECH_EXPORT_H

#if defined(_WIN32)
    #if defined(SPEECH_BUILDING_SHARED)
        #define SPEECH_API __declspec(dllexport)
    #elif defined(SPEECH_SHARED)
        #define SPEECH_API __declspec(dllimport)
    #else
        // Built/consumed as a plain static library on Windows -- no
        // decoration needed or wanted.
        #define SPEECH_API
    #endif
#else
    // ELF/Mach-O already export everything by default; nothing to do.
    #define SPEECH_API
#endif

#endif  // LIBSPEECH_EXPORT_H
