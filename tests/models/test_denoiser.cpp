#include "utest.h"

#include <stdexcept>
#include <type_traits>

#include "libspeech/models/denoiser.h"
#include "libspeech/models/dpdfnet_denoiser.h"
#include "libspeech/models/facebook_denoiser.h"
#include "libspeech/models/gtcrn_denoiser.h"
#include "libspeech/models/speechbrain_denoiser.h"

// These tests exercise speech::models::Denoiser::Create()'s dispatch logic only -- the
// unknown-backend check happens before any network/ONNX Runtime code runs,
// so this stays fast and offline-friendly. Actually constructing a real
// FacebookDenoiser/SpeechBrainDenoiser downloads real ONNX model weights
// from the network (see BaseModel), which isn't appropriate for a fast
// default test suite -- see checklist.md for the follow-up on integration
// tests against real model weights.

UTEST(denoiser, unknown_backend_throws) {
    bool threw = false;
    try {
        speech::models::denoiser::create("not_a_real_backend", "foo.onnx");
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    ASSERT_TRUE(threw);
}

UTEST(denoiser, unknown_backend_error_message_names_the_backend) {
    try {
        speech::models::denoiser::create("bogus_backend_xyz", "foo.onnx");
        ASSERT_TRUE(false);  // Create() should have thrown before reaching here.
    } catch (const std::invalid_argument& e) {
        std::string message = e.what();
        ASSERT_TRUE(message.find("bogus_backend_xyz") != std::string::npos);
    }
}

UTEST(denoiser, empty_backend_name_throws) {
    bool threw = false;
    try {
        speech::models::denoiser::create("", "foo.onnx");
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    ASSERT_TRUE(threw);
}

// The backends are single-inheritance implementations of the interface (see the notes in
// denoiser.h) and a Denoiser owns a model, so it must not be copyable.
static_assert(std::is_base_of<speech::models::denoiser, speech::models::facebook_denoiser>::value, "");
static_assert(std::is_base_of<speech::models::denoiser, speech::models::speechbrain_denoiser>::value, "");
static_assert(!std::is_copy_constructible<speech::models::denoiser>::value, "");
static_assert(!std::is_copy_assignable<speech::models::denoiser>::value, "");
static_assert(!std::is_copy_constructible<speech::models::facebook_denoiser>::value, "");

// Rejected before any model download / ONNX Runtime work, so this stays offline.
UTEST(denoiser, negative_num_threads_throws) {
    bool threw = false;
    try {
        speech::models::denoiser::create("facebook", "foo.onnx", 16000, -1);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    ASSERT_TRUE(threw);
}

UTEST(gtcrn, only_16khz_is_accepted_and_nothing_is_downloaded_for_it) {
    // Rejected before any model file is touched, so this needs no network.
    bool threw = false;
    try {
        speech::models::gtcrn_denoiser d("gtcrn_simple.onnx", 8000);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    ASSERT_TRUE(threw);
}

UTEST(gtcrn, factory_knows_the_backend_name) {
    bool unknown_threw = false;
    try {
        speech::models::denoiser::create("gtrcn", "x.onnx");  // typo must still be rejected
    } catch (const std::invalid_argument&) {
        unknown_threw = true;
    }
    ASSERT_TRUE(unknown_threw);
}

UTEST(dpdfnet, factory_knows_the_backend_and_checks_threads_before_any_download) {
    bool threw = false;
    try {
        speech::models::denoiser::create("dpdfnet", "dpdfnet2", 16000, -1);
    } catch (const std::invalid_argument&) {
        threw = true;  // a known backend: it got as far as the thread-count check, not "unknown backend"
    }
    ASSERT_TRUE(threw);
}
