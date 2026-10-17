#include "libspeech/models/model_files.h"

#include <atomic>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <vector>

#include <httpp/download.hpp>
#include <httpp/sha256.hpp>

#include "aixlog.hpp"
#include "libspeech/detail/log.h"
#include "libspeech/utils/utils.h"
#include "model_files_detail.h"

namespace fs = std::filesystem;

namespace speech::models {

namespace {

constexpr const char* log_tag = "speech::models::fetch_model";
constexpr std::size_t max_checksum_file_bytes = 1024;  // a checksum file is one short line

bool starts_with(const std::string& s, const char* prefix) { return s.rfind(prefix, 0) == 0; }

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

// Deletes the staged files of a download unless they were installed.
struct staged_files {
    std::vector<fs::path> paths;
    ~staged_files() {
        for (const auto& p : paths) remove_quietly(p);
    }
};

// A name part that no other download of the same model, in this or another process, uses.
std::string unique_tag() {
    static std::atomic<unsigned> counter{0};
    std::random_device rd;
    std::ostringstream tag;
    tag << std::hex << rd() << rd() << '-' << counter.fetch_add(1);
    return tag.str();
}

// Moves `from` over `to` (replacing it).
void install(const fs::path& from, const fs::path& to) {
    std::error_code ec;
    fs::rename(from, to, ec);
    if (!ec) return;
    fs::remove(to, ec);    // some platforms refuse to rename over an existing file
    fs::rename(from, to);  // throws fs::filesystem_error (a std::runtime_error) if it fails again
}

}  // namespace

std::string parse_checksum(const std::string& text, const std::string& file_name) {
    return httpp::parse_sha256_checksum(text, file_name);
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

    // Downloads a small sidecar file (always fresh: a cached one of the same size can still be an old version);
    // false if the server does not have it.
    const auto fetch_sidecar = [&](const std::string& from, const fs::path& to) {
        fetch_request request;
        request.url = from;
        request.dest = to;
        request.force = true;
        const fetch_result result = fetch(request);
        if (!result.ok) remove_quietly(to);
        return result.ok;
    };

    // ---- 1. the cache -------------------------------------------------------------------------
    // Returns true (and fills `out`) if the model is cached and passes its checksum.
    const auto try_cache = [&]() -> bool {
        out.sha256.clear();
        out.checksum.clear();
        out.readme.clear();
        if (!fs::exists(out.model)) return false;
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
            if (fetch_sidecar(sum_url, sum_path)) {
                // A malformed file here is an error, not "unavailable": the server answered with something wrong.
                try {
                    expected = parse_checksum(read_text(sum_path, max_checksum_file_bytes), file_name);
                } catch (const std::exception&) {
                    remove_quietly(sum_path);
                    throw;
                }
                have_checksum = true;
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
            return false;
        }
        // cache hit (verified, or unverifiable offline)
        if (!fs::exists(readme_path)) {
            if (!fetch_sidecar(readme_url, readme_path)) {
                SPEECH_LOG(WARNING) << TAG(log_tag) << "Could not get " << readme_url << std::endl;
            }
        }
        out.readme = fs::exists(readme_path) ? readme_path : fs::path();
        return true;
    };
    if (!options.ignore_cache && try_cache()) return out;

    // ---- 2. download: checksum, readme, then the model ------------------------------------------
    // The checksum and the readme are fetched under temporary, per-download names (`*.<tag>.new`) and installed only after the model
    // arrived and matched: a failure anywhere leaves the cache exactly as it was (a failed refresh keeps the old
    // model together with its old checksum). A missing sidecar fails before the big download starts. Several
    // processes that want the same model download it once: the transport holds a per-file lock, and a process
    // that had to wait finds the finished file.
    SPEECH_LOG(DEBUG) << TAG(log_tag) << "Downloading model from: " << url << std::endl;
    const std::string tag = unique_tag();  // two downloads of the same model must not share staged files
    const fs::path sum_new = dir / (file_name + ".sha256." + tag + ".new");
    const fs::path readme_new = dir / (file_name + ".readme." + tag + ".new");
    staged_files staged;
    staged.paths = {sum_new, readme_new};

    std::string expected;
    const bool have_sum = fetch_sidecar(sum_url, sum_new);
    if (have_sum) {
        expected = parse_checksum(read_text(sum_new, max_checksum_file_bytes), file_name);  // throws on e.g. HTML
    } else if (required) {
        throw std::runtime_error("Failed to download the checksum file " + sum_url);
    } else {
        SPEECH_LOG(WARNING) << TAG(log_tag) << "No checksum file at " << sum_url << "; the model will not be verified."
                            << std::endl;
    }

    const bool have_readme = fetch_sidecar(readme_url, readme_new);
    if (!have_readme) {
        if (required) throw std::runtime_error("Failed to download the readme " + readme_url);
        SPEECH_LOG(WARNING) << TAG(log_tag) << "No readme at " << readme_url << std::endl;
    }

    fetch_request request;
    request.url = url;
    request.dest = out.model;
    request.sha256 = expected;
    request.force = options.ignore_cache;
    request.show_progress = true;
    const fetch_result result = fetch(request);
    if (!result.ok) {
        throw std::runtime_error("Failed to download the model " + url + (result.error.empty() ? "" : ": " + result.error));
    }

    // The model is in place and verified: now the sidecars.
    if (have_sum) install(sum_new, sum_path);
    if (have_readme) install(readme_new, readme_path);
    out.sha256 = expected;
    out.checksum = have_sum ? sum_path : fs::path();
    out.readme = have_readme ? readme_path : fs::path();
    return out;
}

}  // namespace detail

model_files fetch_model(const std::string& model, const fetch_options& options) {
    const bool quiet = options.quiet;
    const detail::fetch_fn http = [quiet](const detail::fetch_request& request) {
        httpp::download download(request.url, request.dest.string());
        download.enable_progress(request.show_progress && !quiet).resume().force(request.force);
        if (!request.sha256.empty()) download.sha256(request.sha256);
        const httpp::download_result res = download.run();

        detail::fetch_result result;
        result.ok = res.ok;
        if (!res.ok) {
            result.error = res.error + " (status " + std::to_string(res.status) + ")";
            // A failure is always logged with the URL; a sidecar that is absent is only a warning (the caller
            // decides whether it is required), the model itself is an error.
            if (request.show_progress) {
                SPEECH_LOG(ERROR) << TAG(log_tag) << "Failed to download " << request.url << ". httpp error: "
                                  << result.error << std::endl;
            } else {
                SPEECH_LOG(WARNING) << TAG(log_tag) << "Failed to download " << request.url << ". httpp error: "
                                    << result.error << std::endl;
            }
        }
        return result;
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
