//
// Created by Mohammad on 3/15/2025.
//

#ifndef LIBSPEECH_UTILS_H
#define LIBSPEECH_UTILS_H


#include <filesystem>


namespace fs = std::filesystem;

namespace speech{ namespace utils{
// Callback function to write data received from the server into a file
size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp);

// Function to download a file from a given URL and save it in a specified folder
bool downloadFile(const std::string& url, const fs::path& outputPath, bool force = false);

// Function to get the system's temporary directory as a filesystem::path
fs::path getTempDirectory();
}}

#endif  // LIBSPEECH_UTILS_H
