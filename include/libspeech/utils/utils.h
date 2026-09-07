//
// Created by Mohammad on 3/15/2025.
//

#ifndef LIBSPEECH_UTILS_H
#define LIBSPEECH_UTILS_H


#include <filesystem>


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
std::filesystem::path downloadFile(const std::string& url,
                                   const std::filesystem::path& outputPath,
                                   bool force, bool quiet);


// Function to get the system's temporary directory as a filesystem::path
std::filesystem::path getTempDirectory();

// Cross-platform default cache directory for downloaded model weights
// (<home>/.libspeech). Checks HOME (Unix/macOS), then USERPROFILE and
// HOMEDRIVE+HOMEPATH (Windows), falling back to the system temp directory
// (with a logged warning) if none of those are set -- raw `getenv("HOME")`
// returns nullptr on essentially all Windows systems (HOME isn't a
// standard Windows env var), and constructing a std::filesystem::path from
// a null pointer is undefined behavior.
std::filesystem::path getDefaultModelCacheDir();
}
#endif  // LIBSPEECH_UTILS_H
