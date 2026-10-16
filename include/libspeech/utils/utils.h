//
// Created by Mohammad on 3/15/2025.
//

#ifndef LIBSPEECH_UTILS_H
#define LIBSPEECH_UTILS_H


#include <cstddef>
#include <filesystem>
#include <string>

#include "libspeech/export.h"


namespace speech::utils {

template<typename... Args>
    std::string format(const char* fmt, Args... args)
{
    size_t size = snprintf(nullptr, 0, fmt, args...);
    std::string buf;
    buf.reserve(size + 1);
    buf.resize(size);
    snprintf(&buf[0], size + 1, fmt, args...);
    return buf;
}

// Function to download a file from a given URL and save it in a specified folder
SPEECH_API std::filesystem::path download_file(const std::string& url,
                                   const std::filesystem::path& output_path,
                                   bool force, bool quiet);


// SHA-256 digest as 64 lowercase hex characters. Streams the file in chunks, so the file size is
// not limited by memory. sha256_file throws std::runtime_error if the file cannot be opened/read.
SPEECH_API std::string sha256_hex(const void* data, std::size_t size);
SPEECH_API std::string sha256_file(const std::filesystem::path& path);

// Which implementation sha256_* uses: "sha-ni" (x86 SHA extensions, picked automatically when the CPU
// has them) or "portable". set_sha256_hardware_enabled(false) forces the portable code (for tests and
// benchmarks); the digest is identical either way.
SPEECH_API const char* sha256_backend() noexcept;
SPEECH_API void set_sha256_hardware_enabled(bool enabled) noexcept;

// Function to get the system's temporary directory as a filesystem::path
SPEECH_API std::filesystem::path get_temp_directory();

// Cross-platform default cache directory for downloaded model weights
// (<home>/.libspeech). Checks HOME (Unix/macOS), then USERPROFILE and
// HOMEDRIVE+HOMEPATH (Windows), falling back to the system temp directory
// (with a logged warning) if none of those are set -- raw `getenv("HOME")`
// returns nullptr on essentially all Windows systems (HOME isn't a
// standard Windows env var), and constructing a std::filesystem::path from
// a null pointer is undefined behavior.
SPEECH_API std::filesystem::path get_default_model_cache_dir();
}
#endif  // LIBSPEECH_UTILS_H
