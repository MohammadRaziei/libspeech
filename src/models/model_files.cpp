#include "libspeech/models/model_files.h"

#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <iterator>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

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
    if (!ec) return;
    // Do not touch `to` if the file to install is gone: whoever deleted it may have installed `to` itself.
    if (!fs::exists(from)) throw std::runtime_error("The downloaded file " + from.string() + " has disappeared");
    fs::remove(to, ec);    // some platforms refuse to rename over an existing file
    fs::rename(from, to);  // throws fs::filesystem_error (a std::runtime_error) if it fails again
}

long current_pid() {
#ifdef _WIN32
    return static_cast<long>(_getpid());
#else
    return static_cast<long>(getpid());
#endif
}

// A name part that no other process or thread uses, so two downloads of the same model never write to the
// same temporary file.
std::string unique_tag() {
    static std::atomic<unsigned long long> counter{0};
    std::random_device rd;
    std::ostringstream ss;
    ss << std::hex << current_pid() << '-' << rd() << '-' << counter++;
    return ss.str();
}

// Removes the temporary files (`<model>[.sha256|.readme][.<tag>].part`) that downloads which were killed
// left behind. Called while holding the download lock.
void remove_orphan_parts(const fs::path& dir, const std::string& file_name) {
    static const char* const suffixes[] = {"", ".sha256", ".readme"};
    std::error_code ec;
    for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
        const std::string name = it->path().filename().string();
        if (name.size() <= 5 || name.compare(name.size() - 5, 5, ".part") != 0) continue;
        const std::string stem = name.substr(0, name.size() - 5);
        for (const char* suffix : suffixes) {
            const std::string base = file_name + suffix;
            if (stem == base || (stem.size() > base.size() + 1 && stem.compare(0, base.size() + 1, base + ".") == 0)) {
                remove_quietly(it->path());
                break;
            }
        }
    }
}

// Keeps several processes (or threads) that want the same model from downloading it at the same time.
// The lock is a directory `<model>.lock`, because creating a directory is atomic everywhere. While the owner
// downloads, a helper thread rewrites `<model>.lock/alive` every `heartbeat`; a waiter that sees a lock whose
// heartbeat is older than `stale_after` knows the owner died and takes the lock over. A waiter that sees a
// live lock waits, and when the lock is gone it looks at the cache again (the owner has put the model there).
//
// The lock only saves work: every download writes to its own temporary files and the result is verified
// before it is installed, so the rare overlap (two waiters taking over the same dead lock at the same
// moment, or a waiter that gave up after `max_wait`) costs a duplicate download, not a damaged cache.
class download_lock {
   public:
    download_lock(fs::path path, const detail::lock_timing& timing, const std::string& what)
        : path_(std::move(path)), timing_(timing) {
        using clock = std::chrono::steady_clock;
        const auto give_up = clock::now() + timing_.max_wait;
        bool announced = false;
        for (;;) {
            std::error_code ec;
            if (fs::create_directory(path_, ec) && !ec) {
                owned_ = true;
                break;
            }
            if (ec && !fs::exists(path_)) break;  // cannot create it (read-only cache?): go on without a lock
            if (is_stale()) {
                SPEECH_LOG(WARNING) << TAG(log_tag) << "The download lock " << path_
                                    << " has no heartbeat; its process is gone. Taking it over." << std::endl;
                fs::remove_all(path_, ec);
                continue;
            }
            if (!announced) {
                SPEECH_LOG(INFO) << TAG(log_tag) << "Another process is downloading " << what << "; waiting for it."
                                 << std::endl;
                announced = true;
            }
            waited_ = true;
            if (clock::now() >= give_up) {
                SPEECH_LOG(WARNING) << TAG(log_tag) << "Still locked by another download after "
                                    << std::chrono::duration_cast<std::chrono::seconds>(timing_.max_wait).count()
                                    << " s; downloading " << what << " anyway." << std::endl;
                break;
            }
            std::this_thread::sleep_for(timing_.poll);
        }
        if (owned_) {
            beat(0);
            heart_ = std::thread([this] { run_heartbeat(); });
        }
    }

    ~download_lock() {
        if (!owned_) return;  // never touch a lock that belongs to someone else
        {
            std::lock_guard<std::mutex> guard(mutex_);
            stop_ = true;
        }
        wake_.notify_all();
        if (heart_.joinable()) heart_.join();
        std::error_code ec;
        fs::remove_all(path_, ec);
    }

    download_lock(const download_lock&) = delete;
    download_lock& operator=(const download_lock&) = delete;

    /** true if the lock was held by someone else at some point, i.e. the cache may have been filled meanwhile. */
    bool waited() const noexcept { return waited_; }

   private:
    bool is_stale() const {
        std::error_code ec;
        auto last = fs::last_write_time(path_ / "alive", ec);
        if (ec) last = fs::last_write_time(path_, ec);  // the owner has not written `alive` yet
        if (ec) return false;                           // gone, or unreadable: look again
        return fs::file_time_type::clock::now() - last > timing_.stale_after;
    }

    void beat(unsigned long long n) const {
        std::ofstream out(path_ / "alive", std::ios::binary | std::ios::trunc);
        out << current_pid() << ' ' << n << '\n';
    }

    void run_heartbeat() {
        std::unique_lock<std::mutex> lk(mutex_);
        unsigned long long n = 0;
        while (!stop_) {
            beat(++n);
            wake_.wait_for(lk, timing_.heartbeat, [this] { return stop_; });
        }
    }

    fs::path path_;
    detail::lock_timing timing_;
    bool owned_ = false;
    bool waited_ = false;
    bool stop_ = false;
    std::mutex mutex_;
    std::condition_variable wake_;
    std::thread heart_;
};

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

model_files fetch_model_with(const fetch_fn& fetch, const std::string& model, const fetch_options& options,
                             const lock_timing& timing) {
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
    const std::string tag = unique_tag();
    const auto part = [&](const fs::path& final_path) {
        fs::path p = final_path;
        p += "." + tag + ".part";  // unique to this call: no other download writes here
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
            return false;
        }
        // cache hit (verified, or unverifiable offline)
        if (!fs::exists(readme_path)) {
            const fs::path tmp = part(readme_path);
            if (fetch_sidecar(readme_url, tmp)) {
                install(tmp, readme_path);
            } else {
                SPEECH_LOG(WARNING) << TAG(log_tag) << "Could not get " << readme_url << std::endl;
            }
        }
        out.readme = fs::exists(readme_path) ? readme_path : fs::path();
        return true;
    };
    if (!options.ignore_cache && try_cache()) return out;

    // Only one process downloads a model at a time; the others wait and then find it in the cache. (With
    // ignore_cache a process that had to wait also takes the fresh download its predecessor made.)
    download_lock lock(dir / (file_name + ".lock"), timing, file_name);
    if (lock.waited() && try_cache()) return out;
    remove_orphan_parts(dir, file_name);  // what killed downloads left behind

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
    try {
        if (have_readme) install(readme_tmp, readme_path);
        if (!expected.empty()) install(sum_tmp, sum_path);
        install(model_tmp, out.model);
    } catch (const std::exception&) {
        // Our temporary files are gone: another process took over our lock (we were suspended for longer than the
        // heartbeat timeout), cleaned them up and downloaded the model itself. If it has finished, use that.
        if (try_cache()) return out;
        throw;
    }
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
