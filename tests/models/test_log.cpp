#include "utest.h"

#include <cstdio>
#include <fstream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "libspeech/utils/log.h"

using speech::utils::log_level;

namespace {
// Restores the global level when a test ends, so tests don't leak state into each other.
struct level_guard {
    log_level saved = speech::utils::get_log_level();
    ~level_guard() { speech::utils::set_log_level(saved); }
};

// Puts the output back on stderr when a test ends (a callback or a file would leak into the next test).
struct output_guard {
    ~output_guard() {
        speech::utils::set_log_callback({});
        speech::utils::set_log_file("");
    }
};

struct captured_line {
    log_level level;
    std::string where;
    std::string message;
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

// <windows.h> defines `ERROR` as 0, and aixlog.hpp restores that macro after its own header. The old
// SPEECH_LOG(ERROR) expanded it to severity 0 (= trace), so on Windows no error message was ever
// written. Reproduce the macro here so a regression is caught on every platform, not only Windows.
#include "libspeech/detail/log.h"
#define ERROR 0

UTEST(log, error_severity_is_not_hijacked_by_a_windows_style_error_macro) {
    level_guard guard;
    speech::utils::set_log_level(log_level::error);
    bool error_built = false;
    bool warning_built = false;
    SPEECH_LOG(ERROR) << (error_built = true, "expected test output: error line is written");
    SPEECH_LOG(WARNING) << (warning_built = true, "must not be built at the error level");
    ASSERT_TRUE(error_built);
    ASSERT_FALSE(warning_built);
}

// --- outputs: callback and file ------------------------------------------------------------------

UTEST(log, callback_receives_level_where_and_message) {
    level_guard guard;
    output_guard outputs;
    std::vector<captured_line> got;
    speech::utils::set_log_level(log_level::debug);
    speech::utils::set_log_callback([&](log_level level, const std::string& where, const std::string& message) {
        got.push_back({level, where, message});
    });
    SPEECH_LOG(WARNING) << TAG("speech::test::tagged") << "value " << 42 << std::endl;
    SPEECH_LOG(DEBUG) << "no tag here";
    ASSERT_EQ(got.size(), static_cast<size_t>(2));
    ASSERT_TRUE(got[0].level == log_level::warning);
    ASSERT_STREQ(got[0].where.c_str(), "speech::test::tagged");  // the tag wins
    ASSERT_STREQ(got[0].message.c_str(), "value 42");            // trailing newline trimmed
    ASSERT_TRUE(got[1].level == log_level::debug);
    ASSERT_FALSE(got[1].where.empty());                          // no tag: the function name
    ASSERT_STREQ(got[1].message.c_str(), "no tag here");
}

UTEST(log, callback_is_not_called_below_the_level_and_the_message_is_not_built) {
    level_guard guard;
    output_guard outputs;
    int calls = 0;
    bool built = false;
    speech::utils::set_log_level(log_level::error);
    speech::utils::set_log_callback([&](log_level, const std::string&, const std::string&) { ++calls; });
    SPEECH_LOG(INFO) << (built = true, "dropped");
    ASSERT_EQ(calls, 0);
    ASSERT_FALSE(built);
    speech::utils::set_log_level(log_level::off);
    SPEECH_LOG(ERROR) << "dropped too";
    ASSERT_EQ(calls, 0);
}

UTEST(log, a_throwing_callback_does_not_escape) {
    level_guard guard;
    output_guard outputs;
    speech::utils::set_log_level(log_level::info);
    speech::utils::set_log_callback([](log_level, const std::string&, const std::string&) {
        throw std::runtime_error("expected test output: callback failure");
    });
    SPEECH_LOG(INFO) << "this must not throw";
    ASSERT_TRUE(true);
}

UTEST(log, callback_may_be_called_from_many_threads) {
    level_guard guard;
    output_guard outputs;
    std::mutex mu;
    int calls = 0;
    speech::utils::set_log_level(log_level::info);
    speech::utils::set_log_callback([&](log_level, const std::string&, const std::string&) {
        std::lock_guard<std::mutex> lock(mu);
        ++calls;
    });
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t)
        threads.emplace_back([] {
            for (int i = 0; i < 100; ++i) SPEECH_LOG(INFO) << "thread line " << i;
        });
    for (auto& t : threads) t.join();
    ASSERT_EQ(calls, 400);
}

UTEST(log, file_gets_the_lines_and_an_empty_path_goes_back_to_stderr) {
    level_guard guard;
    output_guard outputs;
    const std::string path = "libspeech_test_log_file.txt";
    speech::utils::set_log_level(log_level::info);
    ASSERT_TRUE(speech::utils::set_log_file(path));
    SPEECH_LOG(INFO) << TAG("speech::test::file") << "written to the file";
    SPEECH_LOG(DEBUG) << "below the level";
    ASSERT_TRUE(speech::utils::set_log_file(""));  // closes (flushes) the file
    std::ifstream in(path);
    std::stringstream text;
    text << in.rdbuf();
    in.close();
    std::remove(path.c_str());
    const std::string content = text.str();
    ASSERT_TRUE(content.find("written to the file") != std::string::npos);
    ASSERT_TRUE(content.find("speech::test::file") != std::string::npos);
    ASSERT_TRUE(content.find("below the level") == std::string::npos);
}

UTEST(log, a_file_that_cannot_be_opened_returns_false_and_keeps_the_previous_output) {
    level_guard guard;
    output_guard outputs;
    int calls = 0;
    speech::utils::set_log_level(log_level::info);
    speech::utils::set_log_callback([&](log_level, const std::string&, const std::string&) { ++calls; });
    ASSERT_FALSE(speech::utils::set_log_file("no_such_directory_for_libspeech/log.txt"));
    SPEECH_LOG(INFO) << "still goes to the callback";
    ASSERT_EQ(calls, 1);
}

UTEST(log, a_file_replaces_a_callback_and_a_callback_replaces_a_file) {
    level_guard guard;
    output_guard outputs;
    const std::string path = "libspeech_test_log_switch.txt";
    int calls = 0;
    speech::utils::set_log_level(log_level::info);
    speech::utils::set_log_callback([&](log_level, const std::string&, const std::string&) { ++calls; });
    ASSERT_TRUE(speech::utils::set_log_file(path));
    SPEECH_LOG(INFO) << "to the file";
    ASSERT_EQ(calls, 0);
    speech::utils::set_log_callback([&](log_level, const std::string&, const std::string&) { ++calls; });
    SPEECH_LOG(INFO) << "to the callback";
    ASSERT_EQ(calls, 1);
    speech::utils::set_log_callback({});
    speech::utils::set_log_file("");
    std::remove(path.c_str());
}

#undef ERROR
