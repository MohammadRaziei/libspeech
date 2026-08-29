//
// Created by Mohammad on 3/15/2025.
//
#include <fstream>
#include <iostream>
#include <regex>

#define CPPHTTPLIB_MBEDTLS_SUPPORT
#include "httplib.h"

#include "libspeech/utils/utils.h"

#include "utils/progressbar.h"

#include "aixlog.hpp"


namespace {

constexpr const char* kTag = "speech::utils::downloadFile";

// Splits a URL into "scheme://host[:port]" and the remaining "/path?query"
// (httplib::Client is constructed with the former and Get() takes the latter).
struct ParsedUrl {
    std::string schemeHost;
    std::string path;
};

ParsedUrl parseUrl(const std::string& url) {
    static const std::regex re(R"(^(https?://[^/]+)(/.*)?$)");
    std::smatch match;
    if (std::regex_match(url, match, re)) {
        return {match[1].str(), match[2].matched ? match[2].str() : "/"};
    }
    return {};  // empty schemeHost signals "invalid URL" to the caller
}

}  // namespace

std::filesystem::path speech::utils::getTempDirectory() {
    // Use filesystem to get the temp directory
    return std::filesystem::temp_directory_path();
}

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
        LOG(INFO) << TAG(kTag) << COND(!quiet)
                  << "File already exists: " << finalOutputPath << ". Skipping download." << std::endl;
        return finalOutputPath;
    }

    // Create the parent directory if it doesn't exist
    std::filesystem::path parentDir = finalOutputPath.parent_path();
    if (!parentDir.empty() && !std::filesystem::exists(parentDir)) {
        if (!std::filesystem::create_directories(parentDir)) {
            LOG(INFO) << TAG(kTag) << COND(!quiet)
                      << "Error: Could not create directory: " << parentDir << std::endl;
            return {};  // Return an empty path on failure
        }
    }

    ParsedUrl parsed = parseUrl(url);
    if (parsed.schemeHost.empty()) {
        LOG(ERROR) << TAG(kTag) << COND(!quiet) << "Error: Not a valid http(s) URL: " << url << std::endl;
        return {};
    }

    // Open the output file
    std::ofstream outFile(finalOutputPath, std::ios::binary);
    if (!outFile.is_open()) {
        LOG(ERROR) << TAG(kTag) << COND(!quiet)
                  << "Error: Could not open file for writing: " << finalOutputPath << std::endl;
        return {};  // Return an empty path on failure
    }

    httplib::Client client(parsed.schemeHost);
    // Model files (ONNXRuntime releases, model weights, ...) are often
    // served via a redirect to a different host (e.g. github.com ->
    // objects.githubusercontent.com); httplib follows cross-host redirects
    // transparently when this is enabled.
    client.set_follow_location(true);
    client.set_connection_timeout(30);
    client.set_read_timeout(300);  // model files can be large

    // Extract Filename from `finalOutputPath`
    std::string filename = finalOutputPath.filename().string();

    // Custom Styled Progress Bar
    auto progressBar = speech::utils::createProgressBar("Downloading " + filename + " ");
    progressBar->set_progress(0);

    auto contentReceiver = [&outFile](const char* data, size_t length) {
        outFile.write(data, static_cast<std::streamsize>(length));
        return true;
    };
    auto progressCallback = [&progressBar, quiet](uint64_t current, uint64_t total) {
        if (!quiet && total > 0) {
            progressBar->set_progress(static_cast<float>(current) / static_cast<float>(total) * 100.0f);
        }
        return true;  // true = keep going
    };

    httplib::Result res = client.Get(parsed.path, contentReceiver, progressCallback);
    outFile.close();

    if (!quiet) {
        progressBar->set_progress(100);
        LOG(INFO) << TAG(kTag) << "\nDownload completed: " << finalOutputPath << std::endl;
    }

    if (!res) {
        LOG(ERROR) << TAG(kTag) << COND(!quiet)
                  << "Error: Failed to download file. httplib error: " << httplib::to_string(res.error())
                  << std::endl;
        return {};
    }
    if (res->status < 200 || res->status >= 300) {
        LOG(ERROR) << TAG(kTag) << COND(!quiet)
                  << "Error: Failed to download file. HTTP status: " << res->status << std::endl;
        return {};
    }

    return finalOutputPath;
}
