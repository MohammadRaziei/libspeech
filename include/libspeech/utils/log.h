//
// Controls what libspeech logs about itself and where that goes: stderr (default), a file, or your own
// callback. The Python package uses the callback to hand every line to Python's `logging`.
//

#ifndef LIBSPEECH_UTILS_LOG_H
#define LIBSPEECH_UTILS_LOG_H

#include <functional>
#include <string>

#include "libspeech/export.h"

namespace speech::utils {

/**
 * Severity threshold for libspeech's own log output: messages below the level are dropped.
 * Off silences everything.
 */
enum class log_level { trace = 0, debug, info, warning, error, off };

/**
 * Sets the process-wide log level. Thread-safe.
 *
 * Until this is called, the level comes from the LIBSPEECH_LOG environment variable
 * (trace | debug | info | warning | error | off), or is Warning if that is unset or invalid.
 */
SPEECH_API void set_log_level(log_level level) noexcept;

/** The current log level. */
SPEECH_API log_level get_log_level() noexcept;

/**
 * Parses a level name, case-insensitively and ignoring surrounding spaces:
 * trace, debug, info, warning (or warn), error, off (or none).
 * @throws std::invalid_argument for anything else.
 */
SPEECH_API log_level parse_log_level(const std::string& name);

/** Lower-case name of a level ("warning", "off", ...). */
SPEECH_API const char* log_level_name(log_level level) noexcept;

/**
 * Receives one log line: its level, where it came from (the tag, e.g. "speech::dsp::STFT", or else the
 * function name) and the message.
 */
using log_callback = std::function<void(log_level level, const std::string& where, const std::string& message)>;

/**
 * Sends every message that passes the log level to `fn`, instead of stderr or a file. Thread-safe.
 *
 * `fn` is called on whatever thread logs, possibly from several threads at once; it must be quick, must not
 * throw, and must not call back into libspeech. An empty function restores stderr. A callback replaces a log
 * file, and the other way round: only one output is active at a time.
 */
SPEECH_API void set_log_callback(log_callback fn);

/**
 * Writes the log to `path` (truncated) instead of stderr; "" goes back to stderr.
 * @return false, with the previous output kept, if the file cannot be opened.
 */
SPEECH_API bool set_log_file(const std::string& path);

}  // namespace speech::utils

#endif  // LIBSPEECH_UTILS_LOG_H
