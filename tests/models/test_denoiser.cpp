#include "utest.h"

#include <stdexcept>

#include "libspeech/models/denoiser.h"

// These tests exercise speech::models::Denoiser::Create()'s dispatch logic only -- the
// unknown-backend check happens before any network/ONNX Runtime code runs,
// so this stays fast and offline-friendly. Actually constructing a real
// FacebookDenoiser/SpeechBrainDenoiser downloads real ONNX model weights
// from the network (see BaseModel), which isn't appropriate for a fast
// default test suite -- see checklist.md for the follow-up on integration
// tests against real model weights.

UTEST(Denoiser, UnknownBackendThrows) {
    bool threw = false;
    try {
        speech::models::Denoiser::Create("not_a_real_backend", "foo.onnx");
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    ASSERT_TRUE(threw);
}

UTEST(Denoiser, UnknownBackendErrorMessageNamesTheBackend) {
    try {
        speech::models::Denoiser::Create("bogus_backend_xyz", "foo.onnx");
        ASSERT_TRUE(false);  // Create() should have thrown before reaching here.
    } catch (const std::invalid_argument& e) {
        std::string message = e.what();
        ASSERT_TRUE(message.find("bogus_backend_xyz") != std::string::npos);
    }
}

UTEST(Denoiser, EmptyBackendNameThrows) {
    bool threw = false;
    try {
        speech::models::Denoiser::Create("", "foo.onnx");
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    ASSERT_TRUE(threw);
}
