//
// Created by Mohammad on 3/15/2025.
//

#include "utils.h"
#include <fstream>
#include <curl/curl.h>
#include <iostream>

fs::path speech::utils::getTempDirectory() {
    // Use filesystem to get the temp directory
    return fs::temp_directory_path();
}

bool speech::utils::downloadFile(const std::string& url, const fs::path& outputPath, bool force) {
    // Check if the file already exists and force is false
    if (!force && fs::exists(outputPath)) {
        std::cout << "File already exists: " << outputPath << ". Skipping download." << std::endl;
        return true;
    }

    // Create the parent directory of the output file if it doesn't exist
    fs::path parentDir = outputPath.parent_path();
    if (!parentDir.empty() && !fs::exists(parentDir)) {
        fs::create_directories(parentDir);
    }

    // Open the output file
    std::ofstream outFile(outputPath, std::ios::binary);
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << outputPath << std::endl;
        return false;
    }

    // Initialize libcurl
    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "Error: Could not initialize libcurl." << std::endl;
        return false;
    }

    // Set libcurl options
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &outFile);

    // Perform the request
    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        std::cerr << "Error: Failed to download file: " << curl_easy_strerror(res) << std::endl;
    }

    // Cleanup
    curl_easy_cleanup(curl);
    outFile.close();

    return res == CURLE_OK;
}

size_t speech::utils::WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    std::ofstream* file = static_cast<std::ofstream*>(userp);
    file->write(static_cast<char*>(contents), size * nmemb);
    return size * nmemb;
}

