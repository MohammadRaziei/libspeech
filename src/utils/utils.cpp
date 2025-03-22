//
// Created by Mohammad on 3/15/2025.
//
#include <fstream>
#include <iostream>

#include <curl/curl.h>

#include "utils/utils.h"

#include "utils/progressbar.h"

#include "aixlog.hpp"



std::filesystem::path speech::utils::getTempDirectory() {
    // Use filesystem to get the temp directory
    return std::filesystem::temp_directory_path();
}

size_t WriteCallback(void* ptr, size_t size, size_t nmemb, void* stream) {
    std::ofstream* outFile = static_cast<std::ofstream*>(stream);
    outFile->write(static_cast<char*>(ptr), size * nmemb);
    return size * nmemb;
}


// Progress callback function for CURL
int ProgressCallback(void* progressPtr, curl_off_t total, curl_off_t now, curl_off_t, curl_off_t) {
    if (total > 0 && progressPtr) {
        auto* progressBar = static_cast<indicators::ProgressBar*>(progressPtr);
        progressBar->set_progress(static_cast<float>(now) / total * 100.0);
    }
    return 0;
}
// Function to download a file

// Function to download a file
std::filesystem::path speech::utils::downloadFile(const std::string& url, const std::filesystem::path& outputPath, bool force, bool quiet) {
    // Check if the output path is a directory
    std::filesystem::path finalOutputPath = outputPath;
    if (std::filesystem::is_directory(outputPath)) {
        // Extract filename from the URL
        std::string fileName = std::filesystem::path(url).filename().string();
        finalOutputPath /= fileName;  // Append filename to the directory path
    }

    // Check if the file already exists and force is false
    if (!force && std::filesystem::exists(finalOutputPath)) {
        LOG(INFO) << TAG("speech::utils::downloadFile")  << COND(!quiet)
                  << "File already exists: " << finalOutputPath << ". Skipping download." << std::endl;
        return finalOutputPath;
    }

    // Create the parent directory if it doesn't exist
    std::filesystem::path parentDir = finalOutputPath.parent_path();
    if (!parentDir.empty() && !std::filesystem::exists(parentDir)) {
        if (!std::filesystem::create_directories(parentDir)) {
            LOG(INFO) << TAG("speech::utils::downloadFile")  << COND(!quiet)
                      << "Error: Could not create directory: " << parentDir << std::endl;
            return {};  // Return an empty path on failure
        }
    }

    // Open the output file
    std::ofstream outFile(finalOutputPath, std::ios::binary);
    if (!outFile.is_open()) {
        LOG(ERROR) << TAG("speech::utils::downloadFile")  << COND(!quiet)
                  << "Error: Could not open file for writing: " << finalOutputPath << std::endl;
        return {};  // Return an empty path on failure
    }

    // Initialize libcurl
    CURL* curl = curl_easy_init();
    if (!curl) {
        LOG(ERROR) << TAG("speech::utils::downloadFile")  << COND(!quiet) << "Error: Could not initialize libcurl." << std::endl;
        return {};  // Return an empty path on failure
    }

    // Set libcurl options
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &outFile);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 10L);

    // Extract Filename from `finalOutputPath`
    std::string filename = finalOutputPath.filename().string();

    // Custom Styled Progress Bar
    auto progressBar = speech::utils::createProgressBar("Downloading " + filename + " ");

    progressBar->set_progress(0);

    if (!quiet) {
        curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
        curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, ProgressCallback);
        curl_easy_setopt(curl, CURLOPT_XFERINFODATA, progressBar.get());
    } else {
        curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 1L);
    }

    // Perform the request
    CURLcode res = curl_easy_perform(curl);

    // Cleanup
    curl_easy_cleanup(curl);
    outFile.close();

    if (!quiet) {
        progressBar->set_progress(100);
        LOG(INFO) << TAG("speech::utils::downloadFile")
                 << "\nDownload completed: " << finalOutputPath << std::endl;
    }

    // Check if the download was successful
    if (res != CURLE_OK) {
        LOG(ERROR) << TAG("speech::utils::downloadFile")  << COND(!quiet)
                  << "Error: Failed to download file. CURL error: " << curl_easy_strerror(res) << std::endl;
        return {};  // Return an empty path on failure
    }

    return finalOutputPath;
}