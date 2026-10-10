#include "utest.h"

#include <stdexcept>
#include <string>

#include "libspeech/utils/log.h"

using speech::utils::LogLevel;

namespace {
// Restores the global level when a test ends, so tests don't leak state into each other.
struct LevelGuard {
    LogLevel saved = speech::utils::getLogLevel();
    ~LevelGuard() { speech::utils::setLogLevel(saved); }
};
}  // namespace

UTEST(Log, ParseAcceptsAllNamesCaseInsensitively) {
    ASSERT_TRUE(speech::utils::parseLogLevel("trace") == LogLevel::Trace);
    ASSERT_TRUE(speech::utils::parseLogLevel("DEBUG") == LogLevel::Debug);
    ASSERT_TRUE(speech::utils::parseLogLevel("Info") == LogLevel::Info);
    ASSERT_TRUE(speech::utils::parseLogLevel("warning") == LogLevel::Warning);
    ASSERT_TRUE(speech::utils::parseLogLevel("warn") == LogLevel::Warning);
    ASSERT_TRUE(speech::utils::parseLogLevel("error") == LogLevel::Error);
    ASSERT_TRUE(speech::utils::parseLogLevel("off") == LogLevel::Off);
    ASSERT_TRUE(speech::utils::parseLogLevel("none") == LogLevel::Off);
    ASSERT_TRUE(speech::utils::parseLogLevel("  Off \n") == LogLevel::Off);  // surrounding space ignored
}

UTEST(Log, ParseRejectsUnknownNames) {
    for (const char* bad : {"", "verbose", "wa rning", "5"}) {
        bool threw = false;
        try {
            speech::utils::parseLogLevel(bad);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        ASSERT_TRUE(threw);
    }
}

UTEST(Log, SetAndGetRoundTrip) {
    LevelGuard guard;
    for (LogLevel level : {LogLevel::Trace, LogLevel::Debug, LogLevel::Info, LogLevel::Warning,
                           LogLevel::Error, LogLevel::Off}) {
        speech::utils::setLogLevel(level);
        ASSERT_TRUE(speech::utils::getLogLevel() == level);
        // the name round-trips through the parser
        ASSERT_TRUE(speech::utils::parseLogLevel(speech::utils::logLevelName(level)) == level);
    }
}
