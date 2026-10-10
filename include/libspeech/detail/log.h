//
// Internal (not part of the public API): libspeech's logger.
//
// Everything is `inline` on purpose. speech::dsp, speech::io and speech::models are separate static
// libraries that all end up in one libspeech.so, so inline state is merged into a single instance
// there, while a DSP-only build (no models) still links on its own. Code outside the shared library
// never touches this state directly: it goes through speech::utils::setLogLevel() (utils.cpp).
//
// Usage inside the library (same shape as the AixLog macro this replaces):
//     SPEECH_LOG(DEBUG) << TAG(kTag) << "message " << value << std::endl;
// TAG(...) and COND(...) are still the AixLog helpers; only the output side is ours, so nothing
// here takes over std::clog and the host application's own logging is left alone.
//
#ifndef LIBSPEECH_DETAIL_LOG_H
#define LIBSPEECH_DETAIL_LOG_H

#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <ostream>
#include <sstream>
#include <string>

#include "aixlog.hpp"
#include "libspeech/utils/log.h"

namespace speech::detail {

using speech::utils::LogLevel;

inline const char* levelName(LogLevel level) noexcept {
    switch (level) {
        case LogLevel::Trace: return "trace";
        case LogLevel::Debug: return "debug";
        case LogLevel::Info: return "info";
        case LogLevel::Warning: return "warning";
        case LogLevel::Error: return "error";
        case LogLevel::Off: return "off";
    }
    return "off";
}

/** Returns false (and leaves `out` untouched) for an unknown name. */
inline bool tryParseLevel(const std::string& name, LogLevel& out) {
    // Trim surrounding whitespace only (an inner space is a typo, not a separator), then lower-case.
    const char* const blank = " \t\n\r";
    const std::size_t first = name.find_first_not_of(blank);
    if (first == std::string::npos) return false;
    std::string s = name.substr(first, name.find_last_not_of(blank) - first + 1);
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (s == "trace") out = LogLevel::Trace;
    else if (s == "debug") out = LogLevel::Debug;
    else if (s == "info") out = LogLevel::Info;
    else if (s == "warning" || s == "warn") out = LogLevel::Warning;
    else if (s == "error") out = LogLevel::Error;
    else if (s == "off" || s == "none") out = LogLevel::Off;
    else return false;
    return true;
}

/** Level used until setLogLevel() is called: LIBSPEECH_LOG if valid, otherwise Warning. */
inline LogLevel initialLevel() {
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)  // getenv: no portable secure alternative needed for a read-only lookup
#endif
    const char* env = std::getenv("LIBSPEECH_LOG");
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
    LogLevel level = LogLevel::Warning;
    if (env != nullptr && *env != '\0' && !tryParseLevel(env, level)) {
        std::fprintf(stderr,
                     "libspeech: ignoring invalid LIBSPEECH_LOG='%s' "
                     "(use trace, debug, info, warning, error or off)\n",
                     env);
        level = LogLevel::Warning;
    }
    return level;
}

inline std::atomic<int>& levelStorage() {
    static std::atomic<int> level{static_cast<int>(initialLevel())};
    return level;
}

inline LogLevel currentLevel() noexcept {
    return static_cast<LogLevel>(levelStorage().load(std::memory_order_relaxed));
}

inline bool logEnabled(LogLevel level) noexcept {
    return static_cast<int>(level) >= static_cast<int>(currentLevel());
}

/** Maps the AixLog severity token used at a call site (TRACE, DEBUG, ...) to our level. */
inline LogLevel levelFromSeverity(int severity) noexcept {
    switch (severity) {
        case 0: return LogLevel::Trace;
        case 1: return LogLevel::Debug;
        case 2:
        case 3: return LogLevel::Info;  // info, notice
        case 4: return LogLevel::Warning;
        default: return LogLevel::Error;  // error, fatal
    }
}

/**
 * One log line: collects the message, writes it to stderr in a single call on destruction
 * (so lines from different threads never interleave), as
 *     2026-10-09 12:00:00.123 [debug] (tag) message
 */
class LogLine {
   public:
    LogLine(LogLevel level, const char* function) : level_(level), function_(function) {}
    LogLine(const LogLine&) = delete;
    LogLine& operator=(const LogLine&) = delete;

    ~LogLine() noexcept {
        if (!enabled_) return;
        try {
            std::string message = body_.str();
            while (!message.empty() && (message.back() == '\n' || message.back() == '\r')) message.pop_back();

            const auto now = std::chrono::system_clock::now();
            const std::time_t seconds = std::chrono::system_clock::to_time_t(now);
            const auto millis =
                std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000;
            std::tm local{};
#if defined(_WIN32)
            localtime_s(&local, &seconds);
#else
            localtime_r(&seconds, &local);
#endif
            char stamp[32];
            std::strftime(stamp, sizeof(stamp), "%Y-%m-%d %H:%M:%S", &local);

            std::string line = stamp;
            line += '.';
            line += static_cast<char>('0' + millis / 100);
            line += static_cast<char>('0' + millis / 10 % 10);
            line += static_cast<char>('0' + millis % 10);
            line += " [";
            line += levelName(level_);
            line += "] (";
            line += tag_.empty() ? (function_ != nullptr ? function_ : "") : tag_.c_str();
            line += ") ";
            line += message;
            line += '\n';
            std::fwrite(line.data(), 1, line.size(), stderr);  // one call: atomic per line
        } catch (...) {
            // Logging must never throw out of a destructor.
        }
    }

    template <typename T>
    LogLine& operator<<(const T& value) {
        if (enabled_) body_ << value;
        return *this;
    }
    LogLine& operator<<(std::ostream& (*manipulator)(std::ostream&)) {  // std::endl, std::flush, ...
        if (enabled_) manipulator(body_);
        return *this;
    }
    LogLine& operator<<(const AixLog::Tag& tag) {
        if (tag) tag_ = tag.text;
        return *this;
    }
    LogLine& operator<<(const AixLog::Conditional& condition) {
        enabled_ = enabled_ && condition.is_true();
        return *this;
    }

   private:
    LogLevel level_;
    const char* function_;
    std::string tag_;
    std::ostringstream body_;
    bool enabled_ = true;
};

}  // namespace speech::detail

// `SPEECH_LOG(SEV) << ...;` -- the message is not even built when SEV is below the log level.
// (`if (...) {} else` keeps it a single statement that is safe inside an unbraced if/else.)
#define SPEECH_LOG(SEV)                                                              \
    if (!::speech::detail::logEnabled(::speech::detail::levelFromSeverity(SEV))) {   \
    } else                                                                           \
        ::speech::detail::LogLine(::speech::detail::levelFromSeverity(SEV), __func__)

#endif  // LIBSPEECH_DETAIL_LOG_H
