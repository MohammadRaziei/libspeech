//
// Internal (not part of the public API): libspeech's logger.
//
// Everything is `inline` on purpose. speech::dsp, speech::io and speech::models are separate static
// libraries that all end up in one libspeech.so, so inline state is merged into a single instance
// there, while a DSP-only build (no models) still links on its own. Code outside the shared library
// never touches this state directly: it goes through speech::utils::set_log_level() (utils.cpp).
//
// Usage inside the library (same shape as the AixLog macro this replaces):
//     SPEECH_LOG(DEBUG) << TAG(log_tag) << "message " << value << std::endl;
//
// Output goes through AixLog (vendored, src/third_party/aixlog): a line is written into AixLog's
// streambuf through a private ostream and AixLog's SinkCerr / SinkFile format it. The vendored copy
// carries one patch so that AixLog does not point std::clog at itself for the whole process (see
// src/third_party/aixlog/README.md): the host application's own logging is left alone.
//
// Three outputs, one active at a time (the same model as httpp):
//   - stderr (default)
//   - a file          -> speech::utils::set_log_file()
//   - a callback      -> speech::utils::set_log_callback(); this is how the Python package routes
//                        the native log into Python's `logging`
// The level check comes first and is one atomic load: a disabled level builds no message at all.
//
#ifndef LIBSPEECH_DETAIL_LOG_H
#define LIBSPEECH_DETAIL_LOG_H

#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <ostream>
#include <sstream>
#include <string>

#include "aixlog.hpp"
#include "libspeech/utils/log.h"

namespace speech::detail {

using speech::utils::log_callback;
using speech::utils::log_level;

inline const char* level_name(log_level level) noexcept {
    switch (level) {
        case log_level::trace: return "trace";
        case log_level::debug: return "debug";
        case log_level::info: return "info";
        case log_level::warning: return "warning";
        case log_level::error: return "error";
        case log_level::off: return "off";
    }
    return "off";
}

/** Returns false (and leaves `out` untouched) for an unknown name. */
inline bool try_parse_level(const std::string& name, log_level& out) {
    // Trim surrounding whitespace only (an inner space is a typo, not a separator), then lower-case.
    const char* const blank = " \t\n\r";
    const std::size_t first = name.find_first_not_of(blank);
    if (first == std::string::npos) return false;
    std::string s = name.substr(first, name.find_last_not_of(blank) - first + 1);
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (s == "trace") out = log_level::trace;
    else if (s == "debug") out = log_level::debug;
    else if (s == "info") out = log_level::info;
    else if (s == "warning" || s == "warn") out = log_level::warning;
    else if (s == "error") out = log_level::error;
    else if (s == "off" || s == "none") out = log_level::off;
    else return false;
    return true;
}

/** Level used until set_log_level() is called: LIBSPEECH_LOG if valid, otherwise Warning. */
inline log_level initial_level() {
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)  // getenv: no portable secure alternative needed for a read-only lookup
#endif
    const char* env = std::getenv("LIBSPEECH_LOG");
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
    log_level level = log_level::warning;
    if (env != nullptr && *env != '\0' && !try_parse_level(env, level)) {
        std::fprintf(stderr,
                     "libspeech: ignoring invalid LIBSPEECH_LOG='%s' "
                     "(use trace, debug, info, warning, error or off)\n",
                     env);
        level = log_level::warning;
    }
    return level;
}

inline std::atomic<int>& level_storage() {
    static std::atomic<int> level{static_cast<int>(initial_level())};
    return level;
}

inline log_level current_level() noexcept {
    return static_cast<log_level>(level_storage().load(std::memory_order_relaxed));
}

inline bool log_enabled(log_level level) noexcept {
    return static_cast<int>(level) >= static_cast<int>(current_level());
}

inline AixLog::Severity to_aix(log_level level) noexcept {
    switch (level) {
        case log_level::trace: return AixLog::Severity::trace;
        case log_level::debug: return AixLog::Severity::debug;
        case log_level::info: return AixLog::Severity::info;
        case log_level::warning: return AixLog::Severity::warning;
        case log_level::error: return AixLog::Severity::error;
        case log_level::off: break;
    }
    return AixLog::Severity::fatal;
}

// stderr / file line layout: 2026-10-09 12:00:00.123 [Debug] (tag) message
inline const char* log_format() noexcept { return "%Y-%m-%d %H:%M:%S.#ms [#severity] (#tag_func)"; }

/** AixLog's file sink, plus a way to ask whether the file could be opened (its constructor just tries). */
struct file_sink : AixLog::SinkFile {
    file_sink(const std::string& path) : AixLog::SinkFile(AixLog::Filter(AixLog::Severity::trace), path, log_format()) {}
    bool is_open() const { return ofs.is_open(); }
};

struct log_state {
    // `mu` guards `callback` and is never held while a callback runs: the Python callback needs the GIL,
    // and a thread that holds the GIL may be waiting for `write_mu` to log -- running it under a lock
    // would be a deadlock. A callback can therefore be called from several threads at once.
    std::mutex mu;
    std::shared_ptr<const log_callback> callback;

    // `write_mu` serialises the lines written through AixLog (its metadata is set by successive <<) and
    // the swapping of its sinks.
    std::mutex write_mu;
    std::unique_ptr<std::ostream> out;
    bool ready = false;

    void use_stderr_locked() {
        AixLog::Log::init({std::make_shared<AixLog::SinkCerr>(AixLog::Filter(AixLog::Severity::trace), log_format())});
    }
    void ensure_ready_locked() {
        if (ready) return;
        out = std::make_unique<std::ostream>(&AixLog::Log::instance());  // private stream: std::clog untouched
        use_stderr_locked();
        ready = true;
    }
};

inline log_state& state() {
    // Leaked on purpose: a library thread may still log while the process is shutting down.
    static log_state* s = new log_state;
    return *s;
}

/** Sends everything to `fn`; an empty function goes back to stderr. The previous callback dies outside every lock. */
inline void install_callback(log_callback fn) {
    auto& s = state();
    std::shared_ptr<const log_callback> fresh = fn ? std::make_shared<const log_callback>(std::move(fn)) : nullptr;
    std::shared_ptr<const log_callback> previous;
    {
        std::lock_guard<std::mutex> lock(s.mu);
        previous = std::move(s.callback);
        s.callback = fresh;
    }
    if (!fresh) {
        std::lock_guard<std::mutex> lock(s.write_mu);
        s.ensure_ready_locked();
        s.use_stderr_locked();  // also closes a log file that was open
    }
}

/** Writes to `path` (truncated); "" goes back to stderr. Returns false, changing nothing, if it cannot be opened. */
inline bool install_file(const std::string& path) {
    if (path.empty()) {
        install_callback(nullptr);
        return true;
    }
    auto sink = std::make_shared<file_sink>(path);
    if (!sink->is_open()) return false;
    install_callback(nullptr);  // a file replaces a callback
    auto& s = state();
    std::lock_guard<std::mutex> lock(s.write_mu);
    s.ensure_ready_locked();
    AixLog::Log::init({sink});
    return true;
}

/** One finished line. `tag` may be empty; `where` (what a callback receives) is the tag, else the function. */
inline void log_write(log_level level, const char* function, const std::string& tag, const std::string& message) noexcept {
    try {
        auto& s = state();
        std::shared_ptr<const log_callback> callback;
        {
            std::lock_guard<std::mutex> lock(s.mu);
            callback = s.callback;
        }
        const std::string where = tag.empty() ? std::string(function != nullptr ? function : "") : tag;
        if (callback) {
            (*callback)(level, where, message);
            return;
        }
        std::lock_guard<std::mutex> lock(s.write_mu);
        s.ensure_ready_locked();
        *s.out << to_aix(level) << (tag.empty() ? AixLog::Tag() : AixLog::Tag(tag))
               << AixLog::Timestamp(std::chrono::system_clock::now())
               << AixLog::Function(function != nullptr ? function : "", "", 0) << message << std::endl;
    } catch (...) {
        // Logging must never throw out of a destructor.
    }
}

/**
 * One log line: collects the message and sends it on destruction (so a line is never split between
 * threads).
 */
class log_line {
   public:
    log_line(log_level level, const char* function) : level_(level), function_(function) {}
    log_line(const log_line&) = delete;
    log_line& operator=(const log_line&) = delete;

    ~log_line() noexcept {
        if (!enabled_) return;
        try {
            std::string message = body_.str();
            while (!message.empty() && (message.back() == '\n' || message.back() == '\r')) message.pop_back();
            log_write(level_, function_, tag_, message);
        } catch (...) {
        }
    }

    template <typename T>
    log_line& operator<<(const T& value) {
        if (enabled_) body_ << value;
        return *this;
    }
    log_line& operator<<(std::ostream& (*manipulator)(std::ostream&)) {  // std::endl, std::flush, ...
        if (enabled_) manipulator(body_);
        return *this;
    }
    log_line& operator<<(const AixLog::Tag& tag) {
        if (tag) tag_ = tag.text;
        return *this;
    }
    log_line& operator<<(const AixLog::Conditional& condition) {
        enabled_ = enabled_ && condition.is_true();
        return *this;
    }

   private:
    log_level level_;
    const char* function_;
    std::string tag_;
    std::ostringstream body_;
    bool enabled_ = true;
};

}  // namespace speech::detail

// The severity token at a call site (TRACE, DEBUG, INFO, WARNING, ERROR, ...) is turned into a level by
// pasting it onto a prefix, never by letting the preprocessor expand it. That matters on Windows:
// <windows.h> (pulled in by aixlog.hpp, which restores it after its own header) defines `ERROR` as 0,
// so the old `level_from_severity(SEV)` turned every SPEECH_LOG(ERROR) into the trace level and the
// message was dropped. An operand of `##` is not macro-expanded, so `ERROR` stays a plain token --
// but only if the paste happens directly in SPEECH_LOG's own body: an argument that is merely
// forwarded to another macro (SPEECH_LOG_LEVEL(SEV)) is expanded first, which is why SPEECH_LOG
// pastes itself instead of calling a helper.
#define SPEECH_LOG_LEVEL_TRACE ::speech::utils::log_level::trace
#define SPEECH_LOG_LEVEL_DEBUG ::speech::utils::log_level::debug
#define SPEECH_LOG_LEVEL_INFO ::speech::utils::log_level::info
#define SPEECH_LOG_LEVEL_NOTICE ::speech::utils::log_level::info
#define SPEECH_LOG_LEVEL_WARNING ::speech::utils::log_level::warning
#define SPEECH_LOG_LEVEL_ERROR ::speech::utils::log_level::error
#define SPEECH_LOG_LEVEL_FATAL ::speech::utils::log_level::error

// `SPEECH_LOG(SEV) << ...;` -- the message is not even built when SEV is below the log level.
// (`if (...) {} else` keeps it a single statement that is safe inside an unbraced if/else.)
#define SPEECH_LOG(SEV)                                                                \
    if (!::speech::detail::log_enabled(SPEECH_LOG_LEVEL_##SEV)) {                      \
    } else                                                                             \
        ::speech::detail::log_line(SPEECH_LOG_LEVEL_##SEV, __func__)

#endif  // LIBSPEECH_DETAIL_LOG_H
