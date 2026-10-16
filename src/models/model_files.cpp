#include "libspeech/models/model_files.h"

#include <cctype>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <vector>

#include "aixlog.hpp"
#include "libspeech/detail/log.h"
#include "libspeech/utils/utils.h"
#include "model_files_detail.h"

namespace fs = std::filesystem;

namespace speech::models {

namespace {

constexpr const char* log_tag = "speech::models::fetch_model";
constexpr std::size_t max_checksum_file_bytes = 1024;

bool starts_with(const std::string& s, const char* prefix) { return s.rfind(prefix, 0) == 0; }

bool is_hex(char c) { return std::isxdigit(static_cast<unsigned char>(c)) != 0; }

// The part of the text that is safe to put into an error message (an HTML page is not).
std::string preview(const std::string& text) {
    std::string out;
    for (char c : text.substr(0, 60)) {
        out += (std::isprint(static_cast<unsigned char>(c)) != 0) ? c : '.';
    }
    return out;
}

std::string read_text(const fs::path& path, std::size_t limit) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Cannot read " + path.string());
    }
    std::string text(limit + 1, '\0');
    in.read(&text[0], static_cast<std::streamsize>(text.size()));
    text.resize(static_cast<std::size_t>(in.gcount()));
    return text;
}

std::string read_whole(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Cannot read " + path.string());
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

void remove_quietly(const fs::path& path) {
    std::error_code ec;
    fs::remove(path, ec);
}

// Deletes the temporary files of a download unless they were installed.
struct part_guard {
    std::vector<fs::path> paths;
    ~part_guard() {
        for (const auto& p : paths) remove_quietly(p);
    }
};

// Moves `from` over `to` (replacing it).
void install(const fs::path& from, const fs::path& to) {
    std::error_code ec;
    fs::rename(from, to, ec);
    if (ec) {  // some platforms refuse to rename over an existing file
        fs::remove(to, ec);
        fs::rename(from, to);  // throws fs::filesystem_error (a std::runtime_error) if it fails again
    }
}

}  // namespace

std::string parse_checksum(const std::string& text, const std::string& file_name) {
    const std::string what = "not a valid checksum file (expected one line: <64 hex digits>[  <file name>]); starts with: '" +
                             preview(text) + "'";
    if (text.size() > max_checksum_file_bytes) {
        throw std::runtime_error("Checksum file is too large to be one line. It is " + what);
    }
    std::string line = text;
    if (!line.empty() && line.back() == '\n') line.pop_back();
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.find('\n') != std::string::npos || line.find('\r') != std::string::npos) {
        throw std::runtime_error("Checksum file has more than one line. It is " + what);
    }
    if (line.size() < 64) {
        throw std::runtime_error("Checksum file is " + what);
    }
    std::string digest;
    for (std::size_t i = 0; i < 64; ++i) {
        if (!is_hex(line[i])) {
            throw std::runtime_error("Checksum file is " + what);
        }
        digest += static_cast<char>(std::tolower(static_cast<unsigned char>(line[i])));
    }
    if (line.size() > 64) {
        std::size_t pos = 64;
        if (line[pos] != ' ' && line[pos] != '\t') {
            throw std::runtime_error("Checksum file is " + what);
        }
        while (pos < line.size() && (line[pos] == ' ' || line[pos] == '\t')) ++pos;
        if (pos < line.size() && line[pos] == '*') ++pos;  // sha256sum's binary-mode marker
        const std::string name = line.substr(pos);
        if (name != file_name) {
            throw std::runtime_error("Checksum file is for '" + name + "', not for '" + file_name + "'");
        }
    }
    return digest;
}

namespace detail {

model_files fetch_model_with(const fetch_fn& fetch, const std::string& model, const fetch_options& options) {
    const bool is_url = starts_with(model, "http://") || starts_with(model, "https://");
    const std::string url = is_url ? model : std::string(release_url) + model;
    const bool required = !is_url;  // checksum + readme are mandatory for the libspeech release

    std::string file_name = url.substr(url.find_last_of('/') + 1);
    file_name = file_name.substr(0, file_name.find_first_of("?#"));
    if (file_name.empty() || file_name == "." || file_name == "..") {
        throw std::invalid_argument("Cannot get a model file name from '" + model + "'");
    }

    const fs::path dir = options.cache_dir.empty() ? utils::get_default_model_cache_dir() : options.cache_dir;
    fs::create_directories(dir);
    model_files out;
    out.model = dir / file_name;
    const fs::path sum_path = dir / (file_name + ".sha256");
    const fs::path readme_path = dir / (file_name + ".readme");
    const std::string sum_url = url + ".sha256";
    const std::string readme_url = url + ".readme";

    part_guard parts;
    const auto part = [&](const fs::path& final_path) {
        fs::path p = final_path;
        p += ".part";
        remove_quietly(p);  // leftover of an interrupted download
        parts.paths.push_back(p);
        return p;
    };
    // Downloads a small sidecar file; false if the server does not have it.
    const auto fetch_sidecar = [&](const std::string& from, const fs::path& to) {
        const bool ok = fetch(from, to);
        if (!ok) remove_quietly(to);
        return ok;
    };

    // ---- 1. the cache -------------------------------------------------------------------------
    if (!options.ignore_cache && fs::exists(out.model)) {
        std::string expected;
        bool have_checksum = false;
        if (fs::exists(sum_path)) {
            try {
                expected = parse_checksum(read_text(sum_path, max_checksum_file_bytes), file_name);
                have_checksum = true;
            } catch (const std::exception& e) {
                SPEECH_LOG(WARNING) << TAG(log_tag) << "Cached " << sum_path << " is unusable (" << e.what()
                                    << "); downloading it again." << std::endl;
                remove_quietly(sum_path);
            }
        }
        if (!have_checksum) {  // a model cached by an older version, or a damaged checksum file
            const fs::path tmp = part(sum_path);
            if (fetch_sidecar(sum_url, tmp)) {
                // A malformed file here is an error, not "unavailable": the server answered with something wrong.
                expected = parse_checksum(read_text(tmp, max_checksum_file_bytes), file_name);
                have_checksum = true;
                install(tmp, sum_path);
            } else {
                SPEECH_LOG(WARNING) << TAG(log_tag) << "Could not get " << sum_url << "; using the cached model "
                                    << out.model << " without verifying it (pass ignore_cache to download it again)."
                                    << std::endl;
            }
        }
        if (!have_checksum) {
            out.checksum = fs::exists(sum_path) ? sum_path : fs::path();
        } else if (utils::sha256_file(out.model) == expected) {
            out.sha256 = expected;
            out.checksum = sum_path;
        } else {
            SPEECH_LOG(WARNING) << TAG(log_tag) << "Cached model " << out.model
                                << " does not match its SHA-256; downloading it again." << std::endl;
            remove_quietly(out.model);
        }
        if (fs::exists(out.model)) {  // cache hit (verified, or unverifiable offline)
            if (!fs::exists(readme_path)) {
                const fs::path tmp = part(readme_path);
                if (fetch_sidecar(readme_url, tmp)) {
                    install(tmp, readme_path);
                } else {
                    SPEECH_LOG(WARNING) << TAG(log_tag) << "Could not get " << readme_url << std::endl;
                }
            }
            out.readme = fs::exists(readme_path) ? readme_path : fs::path();
            return out;
        }
    }

    // ---- 2. download: checksum first, then the model, then the readme ---------------------------
    SPEECH_LOG(DEBUG) << TAG(log_tag) << "Downloading model from: " << url << std::endl;
    std::string expected;
    const fs::path sum_tmp = part(sum_path);
    if (fetch_sidecar(sum_url, sum_tmp)) {
        expected = parse_checksum(read_text(sum_tmp, max_checksum_file_bytes), file_name);  // throws on e.g. HTML
    } else if (required) {
        throw std::runtime_error("Failed to download the checksum file " + sum_url);
    } else {
        SPEECH_LOG(WARNING) << TAG(log_tag) << "No checksum file at " << sum_url << "; the model will not be verified."
                            << std::endl;
    }

    const fs::path model_tmp = part(out.model);
    if (!fetch(url, model_tmp)) {
        throw std::runtime_error("Failed to download the model.");
    }
    if (!expected.empty()) {
        const std::string actual = utils::sha256_file(model_tmp);
        if (actual != expected) {
            throw std::runtime_error("Downloaded model " + url + " does not match its SHA-256 (expected " + expected +
                                     ", got " + actual + "); it was deleted.");
        }
    }

    const fs::path readme_tmp = part(readme_path);
    const bool have_readme = fetch_sidecar(readme_url, readme_tmp);
    if (!have_readme) {
        if (required) throw std::runtime_error("Failed to download the readme " + readme_url);
        SPEECH_LOG(WARNING) << TAG(log_tag) << "No readme at " << readme_url << std::endl;
    }

    // Everything is here and verified: only now touch the cache. The model goes last, so a model file
    // in the cache always has its sidecars next to it.
    if (have_readme) install(readme_tmp, readme_path);
    if (!expected.empty()) install(sum_tmp, sum_path);
    install(model_tmp, out.model);
    out.sha256 = expected;
    out.checksum = expected.empty() ? fs::path() : sum_path;
    out.readme = have_readme ? readme_path : fs::path();
    return out;
}

}  // namespace detail

model_files fetch_model(const std::string& model, const fetch_options& options) {
    const bool quiet = options.quiet;
    const detail::fetch_fn http = [quiet](const std::string& url, const fs::path& dest) {
        // Only the (big) model gets a progress bar; the sidecar files are a few hundred bytes.
        const bool is_sidecar = url.size() > 7 && (url.compare(url.size() - 7, 7, ".readme") == 0 ||
                                                     (url.size() > 7 && url.compare(url.size() - 7, 7, ".sha256") == 0));
        const fs::path got = utils::download_file(url, dest, /*force=*/true, /*quiet=*/quiet || is_sidecar);
        if (got.empty()) {
            remove_quietly(dest);  // never leave a partial / error-page file behind
            return false;
        }
        return true;
    };
    return detail::fetch_model_with(http, model, options);
}

std::string model_readme(const std::string& model, const fetch_options& options) {
    const model_files files = fetch_model(model, options);
    if (files.readme.empty()) {
        throw std::runtime_error("No readme is available for model '" + model + "'.");
    }
    return read_whole(files.readme);
}

}  // namespace speech::models
