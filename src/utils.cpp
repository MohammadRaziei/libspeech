//
// Created by Mohammad on 3/15/2025.
//
#include <fstream>
#include <iostream>

#include <curl/curl.h>
#include <indicators/progress_bar.hpp>
#include <memory>

#include "utils.h"



std::shared_ptr<indicators::ProgressBar> createProgressBar(
    const std::string& text = "Processing ",
    indicators::Color color = indicators::Color::green) {

    // Create a shared_ptr to manage the ProgressBar object
    auto progressBar = std::make_shared<indicators::ProgressBar>(
        indicators::option::BarWidth{50},
        indicators::option::Start{"["},
        indicators::option::Fill{"="},
        indicators::option::Lead{">"},
        indicators::option::Remainder{" "},
        indicators::option::End{"]"},
        indicators::option::ForegroundColor{color},
        indicators::option::ShowPercentage{true},
        indicators::option::ShowElapsedTime{true},
        indicators::option::ShowRemainingTime{true},
        indicators::option::PrefixText{text},
        indicators::option::FontStyles{std::vector<indicators::FontStyle>{indicators::FontStyle::bold}}
    );

    return progressBar; // Return the shared_ptr
}

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
bool speech::utils::downloadFile(const std::string& url, const std::filesystem::path& outputPath, bool force, bool quiet) {
    if (!force && std::filesystem::exists(outputPath)) {
        if (!quiet) std::cout << "File already exists: " << outputPath << ". Skipping download." << std::endl;
        return true;
    }

    std::filesystem::path parentDir = outputPath.parent_path();
    if (!parentDir.empty() && !std::filesystem::exists(parentDir)) {
        std::filesystem::create_directories(parentDir);
    }

    std::ofstream outFile(outputPath, std::ios::binary);
    if (!outFile.is_open()) {
        if (!quiet) std::cerr << "Error: Could not open file for writing: " << outputPath << std::endl;
        return false;
    }

    CURL* curl = curl_easy_init();
    if (!curl) {
        if (!quiet) std::cerr << "Error: Could not initialize libcurl." << std::endl;
        return false;
    }

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &outFile);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 10L);

    // **Extract Filename from `outputPath`**
    std::string filename = outputPath.filename().string();

    // **Custom Styled Progress Bar**
    auto progressBar = createProgressBar("Downloading " + filename + " ");

    progressBar->set_progress(0);

    if (!quiet) {
        curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
        curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, ProgressCallback);
        curl_easy_setopt(curl, CURLOPT_XFERINFODATA, progressBar.get());
    } else {
        curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 1L);
    }

    CURLcode res = curl_easy_perform(curl);

    curl_easy_cleanup(curl);
    outFile.close();

    if (!quiet) {
        progressBar->set_progress(100);
        std::cout << "\nDownload completed: " << outputPath << std::endl;
    }

    return res == CURLE_OK;
}

