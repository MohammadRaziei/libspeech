//
// Created by Mohammad on 3/15/2025.
//
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <stdexcept>

#ifdef LIBSPEECH_HAVE_HTTPP
#include <httpp/download.hpp>
#endif

#include "libspeech/utils/utils.h"

#include "aixlog.hpp"


namespace {

constexpr const char* kTag = "speech::utils::downloadFile";

}  // namespace

std::filesystem::path speech::utils::getTempDirectory() {
    // Use filesystem to get the temp directory
    return std::filesystem::temp_directory_path();
}

std::filesystem::path speech::utils::getDefaultModelCacheDir() {
    // HOME is the standard Unix/macOS env var. On Windows it's generally
    // NOT set -- getenv("HOME") returns nullptr there -- so it must not be
    // the only thing checked.
    if (const char* home = std::getenv("HOME")) {
        return std::filesystem::path(home) / ".libspeech";
    }

    // USERPROFILE is the standard Windows equivalent (e.g. C:\Users\Name).
    if (const char* userProfile = std::getenv("USERPROFILE")) {
        return std::filesystem::path(userProfile) / ".libspeech";
    }

    // Older Windows / some restricted environments only set these two
    // separately (e.g. HOMEDRIVE=C: HOMEPATH=\Users\Name) rather than
    // USERPROFILE.
    const char* homeDrive = std::getenv("HOMEDRIVE");
    const char* homePath = std::getenv("HOMEPATH");
    if (homeDrive && homePath) {
        return std::filesystem::path(std::string(homeDrive) + homePath) / ".libspeech";
    }

    // Last resort: fall back to the system temp directory rather than
    // crashing (constructing a std::filesystem::path from a null pointer,
    // e.g. `std::filesystem::path(getenv("HOME"))` when HOME is unset, is
    // undefined behavior -- this is exactly the bug being fixed here).
    LOG(WARNING) << TAG("speech::utils::getDefaultModelCacheDir")
                 << "Could not determine the user's home directory (HOME/USERPROFILE/"
                    "HOMEDRIVE+HOMEPATH all unset); falling back to the system temp "
                    "directory for downloaded model weights."
                 << std::endl;
    return std::filesystem::temp_directory_path() / ".libspeech";
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

#ifdef LIBSPEECH_HAVE_HTTPP
    // httpp::download replaces the old httplib.h (vendored) + Mbed TLS
    // (submodule) combination: URL parsing, the HTTP(S) client, TLS, and the
    // terminal progress bar are all implemented inside libhttpp_core and
    // never leak a third-party type into this translation unit.
    //
    // As of httpp 0.6.0, httpp::download() follows HTTP redirects by
    // default (follow_redirects(true) is the default) and streams straight
    // to disk regardless of file size, so GitHub Releases' 302 redirects
    // and this project's ~130 MB .onnx model downloads both just work with
    // no extra handling needed here (earlier httpp versions -- 0.1.0's
    // broken HTTPS/TLS, 0.5.0's non-redirecting download_file() and
    // large-body client::request() -- required workarounds that no longer
    // apply; see git history for those if ever needed against an older
    // pinned httpp).
    httpp::download_result res = httpp::download(url, finalOutputPath.string())
                                      .enable_progress(!quiet)
                                      .run();

    if (!quiet) {
        LOG(INFO) << TAG(kTag) << "\nDownload completed: " << finalOutputPath << std::endl;
    }

    if (!res.ok) {
        LOG(ERROR) << TAG(kTag) << COND(!quiet)
                  << "Error: Failed to download file. httpp error: " << res.error
                  << " (status " << res.status << ")" << std::endl;
        return {};
    }

    return finalOutputPath;
#else
    // Built with LIBSPEECH_ENABLE_HTTPP=OFF (the default for Python wheel
    // builds -- see the option's comment in CMakeLists.txt): this binary
    // has no HTTP client compiled in at all, so it can only ever serve
    // files that are already present at `finalOutputPath` -- which the
    // exists-check above already handles. Reaching this point means the
    // caller asked for a URL that isn't cached locally yet; the Python
    // layer (src/libspeech/_download.py) is expected to have downloaded it
    // before ever calling into this library, so surface that clearly
    // instead of silently failing.
    (void)quiet;
    throw std::runtime_error(
        "speech::utils::downloadFile: this build has no download support compiled in "
        "(LIBSPEECH_ENABLE_HTTPP=OFF) and '" + finalOutputPath.string() + "' does not exist "
        "locally yet. Download it first (e.g. via libspeech's Python layer, which fetches "
        "model weights with `requests` before calling into the compiled extension), then "
        "pass the local file path instead of a URL.");
#endif
}
