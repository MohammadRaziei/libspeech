//
// Created by Mohammad on 3/15/2025.
//

#ifndef LIBSPEECH_UTILS_H
#define LIBSPEECH_UTILS_H


#include <filesystem>


namespace speech {
namespace utils {
// Function to download a file from a given URL and save it in a specified folder
bool downloadFile(const std::string& url,
                  const std::filesystem::path& outputPath,
                  bool force = false,
                  bool quiet = false);

// Function to get the system's temporary directory as a filesystem::path
std::filesystem::path getTempDirectory();
}}
#endif  // LIBSPEECH_UTILS_H
