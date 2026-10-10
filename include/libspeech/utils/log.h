//
// Controls how much libspeech writes to stderr.
//

#ifndef LIBSPEECH_UTILS_LOG_H
#define LIBSPEECH_UTILS_LOG_H

#include <string>

#include "libspeech/export.h"

namespace speech::utils {

/**
 * Severity threshold for libspeech's own log output: messages below the level are dropped.
 * Off silences everything.
 */
enum class LogLevel { Trace = 0, Debug, Info, Warning, Error, Off };

/**
 * Sets the process-wide log level. Thread-safe.
 *
 * Until this is called, the level comes from the LIBSPEECH_LOG environment variable
 * (trace | debug | info | warning | error | off), or is Warning if that is unset or invalid.
 */
SPEECH_API void setLogLevel(LogLevel level) noexcept;

/** The current log level. */
SPEECH_API LogLevel getLogLevel() noexcept;

/**
 * Parses a level name, case-insensitively and ignoring surrounding spaces:
 * trace, debug, info, warning (or warn), error, off (or none).
 * @throws std::invalid_argument for anything else.
 */
SPEECH_API LogLevel parseLogLevel(const std::string& name);

/** Lower-case name of a level ("warning", "off", ...). */
SPEECH_API const char* logLevelName(LogLevel level) noexcept;

}  // namespace speech::utils

#endif  // LIBSPEECH_UTILS_LOG_H
