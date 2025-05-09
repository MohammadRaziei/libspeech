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
}
#endif  // LIBSPEECH_UTILS_H
