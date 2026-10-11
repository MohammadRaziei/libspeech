#include "utest.h"

#include <stdexcept>
#include <string>

#include "libspeech/utils/log.h"

using speech::utils::log_level;

namespace {
// Restores the global level when a test ends, so tests don't leak state into each other.
struct level_guard {
    log_level saved = speech::utils::get_log_level();
    ~level_guard() { speech::utils::set_log_level(saved); }
};
}  // namespace

UTEST(log, parse_accepts_all_names_case_insensitively) {
    ASSERT_TRUE(speech::utils::parse_log_level("trace") == log_level::trace);
    ASSERT_TRUE(speech::utils::parse_log_level("DEBUG") == log_level::debug);
    ASSERT_TRUE(speech::utils::parse_log_level("Info") == log_level::info);
    ASSERT_TRUE(speech::utils::parse_log_level("warning") == log_level::warning);
    ASSERT_TRUE(speech::utils::parse_log_level("warn") == log_level::warning);
    ASSERT_TRUE(speech::utils::parse_log_level("error") == log_level::error);
    ASSERT_TRUE(speech::utils::parse_log_level("off") == log_level::off);
    ASSERT_TRUE(speech::utils::parse_log_level("none") == log_level::off);
    ASSERT_TRUE(speech::utils::parse_log_level("  Off \n") == log_level::off);  // surrounding space ignored
}

UTEST(log, parse_rejects_unknown_names) {
    for (const char* bad : {"", "verbose", "wa rning", "5"}) {
        bool threw = false;
        try {
            speech::utils::parse_log_level(bad);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        ASSERT_TRUE(threw);
    }
}

UTEST(log, set_and_get_round_trip) {
    level_guard guard;
    for (log_level level : {log_level::trace, log_level::debug, log_level::info, log_level::warning,
                           log_level::error, log_level::off}) {
        speech::utils::set_log_level(level);
        ASSERT_TRUE(speech::utils::get_log_level() == level);
        // the name round-trips through the parser
        ASSERT_TRUE(speech::utils::parse_log_level(speech::utils::log_level_name(level)) == level);
    }
}
