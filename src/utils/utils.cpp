//
// Created by Mohammad on 3/15/2025.
//
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <stdexcept>

#include <httpp/download.hpp>
#include <httpp/sha256.hpp>

#include "libspeech/utils/utils.h"

#include "aixlog.hpp"
#include "libspeech/detail/log.h"


namespace {

constexpr const char* log_tag = "speech::utils::download_file";

}  // namespace

std::filesystem::path speech::utils::get_temp_directory() {
    // Use filesystem to get the temp directory
    return std::filesystem::temp_directory_path();
}

std::filesystem::path speech::utils::get_default_model_cache_dir() {
    // HOME is the standard Unix/macOS env var. On Windows it's generally
    // NOT set -- getenv("HOME") returns nullptr there -- so it must not be
    // the only thing checked.
    if (const char* home = std::getenv("HOME")) {
        return std::filesystem::path(home) / ".libspeech";
    }

    // USERPROFILE is the standard Windows equivalent (e.g. C:\Users\Name).
    if (const char* user_profile = std::getenv("USERPROFILE")) {
        return std::filesystem::path(user_profile) / ".libspeech";
    }

    // Older Windows / some restricted environments only set these two
    // separately (e.g. HOMEDRIVE=C: HOMEPATH=\Users\Name) rather than
    // USERPROFILE.
    const char* home_drive = std::getenv("HOMEDRIVE");
    const char* home_path = std::getenv("HOMEPATH");
    if (home_drive && home_path) {
        return std::filesystem::path(std::string(home_drive) + home_path) / ".libspeech";
    }

    // Last resort: fall back to the system temp directory rather than
    // crashing (constructing a std::filesystem::path from a null pointer,
    // e.g. `std::filesystem::path(getenv("HOME"))` when HOME is unset, is
    // undefined behavior -- this is exactly the bug being fixed here).
    SPEECH_LOG(WARNING) << TAG("speech::utils::get_default_model_cache_dir")
                 << "Could not determine the user's home directory (HOME/USERPROFILE/"
                    "HOMEDRIVE+HOMEPATH all unset); falling back to the system temp "
                    "directory for downloaded model weights."
                 << std::endl;
    return std::filesystem::temp_directory_path() / ".libspeech";
}

// Function to download a file
std::filesystem::path speech::utils::download_file(const std::string& url, const std::filesystem::path& output_path, bool force, bool quiet) {
    // Check if the output path is a directory
    std::filesystem::path final_output_path = output_path;
    if (std::filesystem::is_directory(output_path)) {
        // Extract filename from the URL
        std::string file_name = std::filesystem::path(url).filename().string();
        final_output_path /= file_name;  // Append filename to the directory path
    }

    // Check if the file already exists and force is false
    if (!force && std::filesystem::exists(final_output_path)) {
        SPEECH_LOG(INFO) << TAG(log_tag) << COND(!quiet)
                  << "File already exists: " << final_output_path << ". Skipping download." << std::endl;
        return final_output_path;
    }

    // Create the parent directory if it doesn't exist
    std::filesystem::path parent_dir = final_output_path.parent_path();
    if (!parent_dir.empty() && !std::filesystem::exists(parent_dir)) {
        if (!std::filesystem::create_directories(parent_dir)) {
            SPEECH_LOG(INFO) << TAG(log_tag) << COND(!quiet)
                      << "Error: Could not create directory: " << parent_dir << std::endl;
            return {};  // Return an empty path on failure
        }
    }

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
    httpp::download_result res = httpp::download(url, final_output_path.string())
                                      .enable_progress(!quiet)
                                      .run();

    if (!quiet) {
        SPEECH_LOG(INFO) << TAG(log_tag) << "\n_download completed: " << final_output_path << std::endl;
    }

    if (!res.ok) {
        // A failure is always logged (with the HTTP status): `quiet` only silences the progress bar and
        // the info lines, and makes the failure a warning -- quiet downloads are the small optional files
        // next to a model (checksum, readme), whose absence the caller may tolerate.
        if (quiet) {
            SPEECH_LOG(WARNING) << TAG(log_tag) << "Failed to download file " << url << ". httpp error: " << res.error
                                << " (status " << res.status << ")" << std::endl;
        } else {
            SPEECH_LOG(ERROR) << TAG(log_tag) << "Error: Failed to download file " << url << ". httpp error: "
                              << res.error << " (status " << res.status << ")" << std::endl;
        }
        return {};
    }

    return final_output_path;
}

// --- SHA-256 ------------------------------------------------------------------------------------
// httpp does the hashing (it needs it to verify downloads); these are the same functions under libspeech's names.

std::string speech::utils::sha256_hex(const void* data, std::size_t size) { return httpp::sha256_hex(data, size); }

std::string speech::utils::sha256_file(const std::filesystem::path& path) { return httpp::sha256_file(path.string()); }

// --- log level ----------------------------------------------------------------------------------
// The state itself lives in libspeech/detail/log.h (inline, shared by every library bundled into
// libspeech.so); these exported functions are the only way code outside the library reaches it.

void speech::utils::set_log_level(log_level level) noexcept {
    speech::detail::level_storage().store(static_cast<int>(level), std::memory_order_relaxed);
}

speech::utils::log_level speech::utils::get_log_level() noexcept {
    return speech::detail::current_level();
}

speech::utils::log_level speech::utils::parse_log_level(const std::string& name) {
    log_level level = log_level::warning;
    if (!speech::detail::try_parse_level(name, level)) {
        throw std::invalid_argument("Unknown log level '" + name +
                                    "' (use trace, debug, info, warning, error or off).");
    }
    return level;
}

const char* speech::utils::log_level_name(log_level level) noexcept {
    return speech::detail::level_name(level);
}

void speech::utils::set_log_callback(log_callback fn) { speech::detail::install_callback(std::move(fn)); }

bool speech::utils::set_log_file(const std::string& path) { return speech::detail::install_file(path); }
