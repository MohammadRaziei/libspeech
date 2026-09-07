#include "utest.h"

#include <cstdlib>
#include <string>

#include "libspeech/utils/utils.h"

// Regression tests for a real bug: BaseModel/ONNXModel used to default
// their model-cache directory to
//   std::filesystem::path(getenv("HOME")) / ".libspeech"
// directly. On Windows, HOME is essentially never set, so getenv("HOME")
// returns nullptr there, and constructing a std::filesystem::path from a
// null pointer is undefined behavior (crashes in practice). These tests
// exercise every fallback speech::utils::getDefaultModelCacheDir() needs
// to handle correctly.

namespace {

// Small RAII helper: sets an env var for the scope of the test, restoring
// (or unsetting) it afterward -- so these tests don't leak state into
// whatever runs after them in the same process.
class ScopedEnv {
   public:
    ScopedEnv(const char* name, const char* value) : name_(name) {
        const char* old = std::getenv(name);
        hadOld_ = old != nullptr;
        if (hadOld_) oldValue_ = old;
        if (value) {
            setenv(name, value, 1);
        } else {
            unsetenv(name);
        }
    }
    ~ScopedEnv() {
        if (hadOld_) {
            setenv(name_.c_str(), oldValue_.c_str(), 1);
        } else {
            unsetenv(name_.c_str());
        }
    }
    ScopedEnv(const ScopedEnv&) = delete;

   private:
    std::string name_;
    std::string oldValue_;
    bool hadOld_;
};

}  // namespace

UTEST(DefaultModelCacheDir, UsesHomeWhenSet) {
    ScopedEnv home("HOME", "/home/testuser");
    auto path = speech::utils::getDefaultModelCacheDir();
    ASSERT_TRUE(path.string().find("/home/testuser") == 0);
    ASSERT_TRUE(path.string().find(".libspeech") != std::string::npos);
}

UTEST(DefaultModelCacheDir, FallsBackToUserProfileWhenHomeUnset) {
    // The exact scenario that used to crash: HOME unset (as on essentially
    // all real Windows systems), USERPROFILE set instead.
    ScopedEnv home("HOME", nullptr);
    ScopedEnv userProfile("USERPROFILE", "C:\\Users\\testuser");
    auto path = speech::utils::getDefaultModelCacheDir();
    ASSERT_TRUE(path.string().find("testuser") != std::string::npos);
    ASSERT_TRUE(path.string().find(".libspeech") != std::string::npos);
}

UTEST(DefaultModelCacheDir, FallsBackToHomeDriveAndHomePathWhenOthersUnset) {
    ScopedEnv home("HOME", nullptr);
    ScopedEnv userProfile("USERPROFILE", nullptr);
    ScopedEnv homeDrive("HOMEDRIVE", "C:");
    ScopedEnv homePath("HOMEPATH", "\\Users\\testuser");
    auto path = speech::utils::getDefaultModelCacheDir();
    ASSERT_TRUE(path.string().find("testuser") != std::string::npos);
    ASSERT_TRUE(path.string().find(".libspeech") != std::string::npos);
}

UTEST(DefaultModelCacheDir, FallsBackToTempDirWithoutCrashingWhenNothingSet) {
    // The critical regression test: none of HOME/USERPROFILE/HOMEDRIVE/
    // HOMEPATH set at all. Must not crash (the original bug: constructing
    // std::filesystem::path(nullptr) is UB) -- must return *some* usable
    // path instead.
    ScopedEnv home("HOME", nullptr);
    ScopedEnv userProfile("USERPROFILE", nullptr);
    ScopedEnv homeDrive("HOMEDRIVE", nullptr);
    ScopedEnv homePath("HOMEPATH", nullptr);

    auto path = speech::utils::getDefaultModelCacheDir();
    ASSERT_TRUE(!path.empty());
    ASSERT_TRUE(path.string().find(".libspeech") != std::string::npos);
}
