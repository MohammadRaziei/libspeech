#include "utest.h"

#include <cstdlib>
#include <string>

#ifdef _WIN32
// MSVC's CRT has no POSIX setenv()/unsetenv() ("identifier not found" at
// link/compile time) -- _putenv_s() is the closest equivalent, and per its
// documentation, calling it with an empty value string removes the
// variable entirely, which is exactly unsetenv()'s behavior.
namespace {
void setenv(const char* name, const char* value, int /*overwrite*/) { _putenv_s(name, value); }
void unsetenv(const char* name) { _putenv_s(name, ""); }
}  // namespace
#endif

#include "libspeech/utils/utils.h"

// Regression tests for a real bug: BaseModel/ONNXModel used to default
// their model-cache directory to
//   std::filesystem::path(getenv("HOME")) / ".libspeech"
// directly. On Windows, HOME is essentially never set, so getenv("HOME")
// returns nullptr there, and constructing a std::filesystem::path from a
// null pointer is undefined behavior (crashes in practice). These tests
// exercise every fallback speech::utils::get_default_model_cache_dir() needs
// to handle correctly.

namespace {

// Small RAII helper: sets an env var for the scope of the test, restoring
// (or unsetting) it afterward -- so these tests don't leak state into
// whatever runs after them in the same process.
class scoped_env {
   public:
    scoped_env(const char* name, const char* value) : name_(name) {
        const char* old = std::getenv(name);
        had_old_ = old != nullptr;
        if (had_old_) old_value_ = old;
        if (value) {
            setenv(name, value, 1);
        } else {
            unsetenv(name);
        }
    }
    ~scoped_env() {
        if (had_old_) {
            setenv(name_.c_str(), old_value_.c_str(), 1);
        } else {
            unsetenv(name_.c_str());
        }
    }
    scoped_env(const scoped_env&) = delete;

   private:
    std::string name_;
    std::string old_value_;
    bool had_old_;
};

}  // namespace

UTEST(get_default_model_cache_dir, uses_home_when_set) {
    scoped_env home("HOME", "/home/testuser");
    auto path = speech::utils::get_default_model_cache_dir();
    ASSERT_TRUE(path.string().find("/home/testuser") == 0);
    ASSERT_TRUE(path.string().find(".libspeech") != std::string::npos);
}

UTEST(get_default_model_cache_dir, falls_back_to_user_profile_when_home_unset) {
    // The exact scenario that used to crash: HOME unset (as on essentially
    // all real Windows systems), USERPROFILE set instead.
    scoped_env home("HOME", nullptr);
    scoped_env user_profile("USERPROFILE", "C:\\Users\\testuser");
    auto path = speech::utils::get_default_model_cache_dir();
    ASSERT_TRUE(path.string().find("testuser") != std::string::npos);
    ASSERT_TRUE(path.string().find(".libspeech") != std::string::npos);
}

UTEST(get_default_model_cache_dir, falls_back_to_home_drive_and_home_path_when_others_unset) {
    scoped_env home("HOME", nullptr);
    scoped_env user_profile("USERPROFILE", nullptr);
    scoped_env home_drive("HOMEDRIVE", "C:");
    scoped_env home_path("HOMEPATH", "\\Users\\testuser");
    auto path = speech::utils::get_default_model_cache_dir();
    ASSERT_TRUE(path.string().find("testuser") != std::string::npos);
    ASSERT_TRUE(path.string().find(".libspeech") != std::string::npos);
}

UTEST(get_default_model_cache_dir, falls_back_to_temp_dir_without_crashing_when_nothing_set) {
    // The critical regression test: none of HOME/USERPROFILE/HOMEDRIVE/
    // HOMEPATH set at all. Must not crash (the original bug: constructing
    // std::filesystem::path(nullptr) is UB) -- must return *some* usable
    // path instead.
    scoped_env home("HOME", nullptr);
    scoped_env user_profile("USERPROFILE", nullptr);
    scoped_env home_drive("HOMEDRIVE", nullptr);
    scoped_env home_path("HOMEPATH", nullptr);

    auto path = speech::utils::get_default_model_cache_dir();
    ASSERT_TRUE(!path.empty());
    ASSERT_TRUE(path.string().find(".libspeech") != std::string::npos);
}
