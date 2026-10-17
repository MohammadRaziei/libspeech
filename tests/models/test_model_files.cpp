#include "utest.h"

#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "libspeech/models/model_files.h"
#include "libspeech/utils/utils.h"
#include "model_files_detail.h"

// The download/verify logic runs here against a fake transport (no network): a map url -> body, plus a
// log of every request, so the tests can check what was asked for, in which order, and what was left behind.
// The fake behaves like httpp::download(...).resume(): the file appears under its name only when complete,
// a complete file with the right checksum is not fetched again unless `force`, and a file that fails its
// checksum is deleted. (The real transport, including the cross-process lock, is covered by the Python tests
// in tests/python/test_model_files.py.)

namespace fs = std::filesystem;
namespace sm = speech::models;

namespace {

const std::string model_bytes = "pretend these are the weights of an onnx model";
const std::string readme_text = "# tiny.onnx\nWhat it is, who made it, license.\n";

std::string sum_line(const std::string& bytes, const std::string& name) {
    return speech::utils::sha256_hex(bytes.data(), bytes.size()) + "  " + name + "\n";
}

struct fake_server {
    std::map<std::string, std::string> files;  // url -> body; absent = 404
    std::vector<std::string> requests;
    std::function<void(const std::string&)> on_fetch;  // called for every request, e.g. to be slow
    std::mutex mutex;                                   // the concurrency tests call fetch() from several threads

    // What the transport was asked, per request (parallel to `requests`).
    std::vector<sm::detail::fetch_request> asked;

    sm::detail::fetch_fn fetch() {
        return [this](const sm::detail::fetch_request& request) {
            sm::detail::fetch_result result;
            if (on_fetch) on_fetch(request.url);
            std::string body;
            {
                std::lock_guard<std::mutex> guard(mutex);
                requests.push_back(request.url);
                asked.push_back(request);
                const auto it = files.find(request.url);
                if (it == files.end()) {
                    result.error = "HTTP 404";
                    return result;
                }
                body = it->second;
            }
            const std::string digest = speech::utils::sha256_hex(body.data(), body.size());
            if (!request.sha256.empty() && !request.force && fs::exists(request.dest) &&
                speech::utils::sha256_file(request.dest) == request.sha256) {
                result.ok = true;  // already complete: no download (httpp reports skipped)
                return result;
            }
            if (!request.sha256.empty() && digest != request.sha256) {
                std::error_code ec;
                fs::remove(request.dest, ec);  // httpp never leaves a file that failed its check
                result.error = "SHA-256 mismatch: expected " + request.sha256 + ", got " + digest + "; the file was deleted";
                return result;
            }
            fs::path part = request.dest;
            part += ".part";  // written aside, renamed when complete
            {
                std::ofstream out(part, std::ios::binary);
                out << body;
            }
            fs::rename(part, request.dest);
            result.ok = true;
            return result;
        };
    }

    std::size_t request_count() {
        std::lock_guard<std::mutex> guard(mutex);
        return requests.size();
    }

    // A complete, correct model under `url`.
    void publish(const std::string& url, const std::string& name, const std::string& bytes = model_bytes) {
        files[url] = bytes;
        files[url + ".sha256"] = sum_line(bytes, name);
        files[url + ".readme"] = readme_text;
    }
};

struct temp_dir {
    fs::path path;
    explicit temp_dir(const std::string& tag)
        : path(fs::temp_directory_path() / ("libspeech_mf_" + tag)) {
        fs::remove_all(path);
        fs::create_directories(path);
    }
    ~temp_dir() { std::error_code ec; fs::remove_all(path, ec); }
    std::vector<std::string> listing() const {
        std::vector<std::string> names;
        for (const auto& e : fs::directory_iterator(path)) names.push_back(e.path().filename().string());
        return names;
    }
    bool has(const std::string& name) const { return fs::exists(path / name); }
};

std::string slurp(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

sm::fetch_options opts(const temp_dir& d, bool ignore_cache = false) {
    sm::fetch_options o;
    o.cache_dir = d.path;
    o.ignore_cache = ignore_cache;
    return o;
}

template <typename F>
std::string error_of(F&& f) {
    try {
        f();
    } catch (const std::exception& e) {
        return e.what();
    }
    return "";
}

const std::string release_tiny = std::string(sm::release_url) + "tiny.onnx";

}  // namespace

// ---- parse_checksum ----------------------------------------------------------------------------

UTEST(parse_checksum, accepts_sha256sum_formats) {
    const std::string hex(64, 'a');
    ASSERT_STREQ(sm::parse_checksum(hex + "  m.onnx\n", "m.onnx").c_str(), hex.c_str());
    ASSERT_STREQ(sm::parse_checksum(hex + "  m.onnx", "m.onnx").c_str(), hex.c_str());
    ASSERT_STREQ(sm::parse_checksum(hex + " *m.onnx\r\n", "m.onnx").c_str(), hex.c_str());
    ASSERT_STREQ(sm::parse_checksum(hex + "\n", "m.onnx").c_str(), hex.c_str());
    ASSERT_STREQ(sm::parse_checksum(hex, "m.onnx").c_str(), hex.c_str());
    ASSERT_STREQ(sm::parse_checksum(std::string(64, 'A') + "  m.onnx\n", "m.onnx").c_str(), hex.c_str());  // lowercased
}

UTEST(parse_checksum, rejects_everything_else) {
    const std::string hex(64, 'a');
    const std::vector<std::string> bad = {
        "<!DOCTYPE html>\n<html><body>Not Found</body></html>\n",
        "",
        "\n",
        std::string(63, 'a') + "\n",
        std::string(65, 'a') + "\n",
        hex + "garbage\n",
        hex + "m.onnx\n",                           // the name must be separated from the digest
        hex + "  m.onnx\n" + hex + "  m.onnx\n",  // two lines
        hex + "  other.onnx\n",                      // checksum of a different file
        std::string(63, 'a') + "g  m.onnx\n",        // not hex
        std::string(2000, 'a'),                      // far too long to be a checksum file
    };
    for (const std::string& text : bad) {
        ASSERT_FALSE(error_of([&] { sm::parse_checksum(text, "m.onnx"); }).empty());
    }
}

// ---- fetch_model -------------------------------------------------------------------------------

UTEST(fetch_model, downloads_checksum_then_readme_then_model) {
    temp_dir dir("order");
    fake_server srv;
    srv.publish(release_tiny, "tiny.onnx");
    const sm::model_files f = sm::detail::fetch_model_with(srv.fetch(), "tiny.onnx", opts(dir));

    ASSERT_EQ(srv.requests.size(), 3u);
    ASSERT_STREQ(srv.requests[0].c_str(), (release_tiny + ".sha256").c_str());  // checksum first
    ASSERT_STREQ(srv.requests[1].c_str(), (release_tiny + ".readme").c_str());  // a missing readme fails early
    ASSERT_STREQ(srv.requests[2].c_str(), release_tiny.c_str());                // the big file last

    ASSERT_STREQ(slurp(f.model).c_str(), model_bytes.c_str());
    ASSERT_STREQ(slurp(f.readme).c_str(), readme_text.c_str());
    ASSERT_STREQ(f.sha256.c_str(), speech::utils::sha256_hex(model_bytes.data(), model_bytes.size()).c_str());
    ASSERT_TRUE(f.checksum == dir.path / "tiny.onnx.sha256");
    ASSERT_EQ(dir.listing().size(), 3u);  // the three files, no .new / .part leftovers
}

UTEST(fetch_model, html_instead_of_checksum_is_an_error_and_nothing_is_kept) {
    temp_dir dir("html");
    fake_server srv;
    srv.publish(release_tiny, "tiny.onnx");
    srv.files[release_tiny + ".sha256"] = "<!DOCTYPE html><html><body>Not Found</body></html>";
    const std::string err = error_of([&] { sm::detail::fetch_model_with(srv.fetch(), "tiny.onnx", opts(dir)); });
    ASSERT_FALSE(err.empty());
    ASSERT_TRUE(err.find("checksum") != std::string::npos);
    ASSERT_EQ(srv.requests.size(), 1u);  // the big file was never requested
    ASSERT_EQ(dir.listing().size(), 0u);
}

UTEST(fetch_model, hash_mismatch_deletes_the_download) {
    temp_dir dir("mismatch");
    fake_server srv;
    srv.publish(release_tiny, "tiny.onnx");
    srv.files[release_tiny] = model_bytes + " tampered";
    const std::string err = error_of([&] { sm::detail::fetch_model_with(srv.fetch(), "tiny.onnx", opts(dir)); });
    ASSERT_TRUE(err.find("SHA-256") != std::string::npos);
    ASSERT_EQ(dir.listing().size(), 0u);
}

UTEST(fetch_model, release_model_requires_checksum_and_readme) {
    {
        temp_dir dir("nosum");
        fake_server srv;
        srv.publish(release_tiny, "tiny.onnx");
        srv.files.erase(release_tiny + ".sha256");
        ASSERT_FALSE(error_of([&] { sm::detail::fetch_model_with(srv.fetch(), "tiny.onnx", opts(dir)); }).empty());
        ASSERT_EQ(dir.listing().size(), 0u);
    }
    {
        temp_dir dir("noreadme");
        fake_server srv;
        srv.publish(release_tiny, "tiny.onnx");
        srv.files.erase(release_tiny + ".readme");
        ASSERT_FALSE(error_of([&] { sm::detail::fetch_model_with(srv.fetch(), "tiny.onnx", opts(dir)); }).empty());
        ASSERT_EQ(srv.requests.size(), 2u);   // checksum, readme: the big model was never requested
        ASSERT_EQ(dir.listing().size(), 0u);  // and the checksum that was fetched is not kept without the rest
    }
}

UTEST(fetch_model, model_download_failure_leaves_nothing) {
    temp_dir dir("nomodel");
    fake_server srv;
    srv.publish(release_tiny, "tiny.onnx");
    srv.files.erase(release_tiny);
    const std::string err = error_of([&] { sm::detail::fetch_model_with(srv.fetch(), "tiny.onnx", opts(dir)); });
    ASSERT_TRUE(err.find("Failed to download the model") != std::string::npos);
    ASSERT_TRUE(err.find("HTTP 404") != std::string::npos);  // the transport's reason is passed on
    ASSERT_EQ(dir.listing().size(), 0u);
}

UTEST(fetch_model, explicit_url_without_sidecars_is_used_unverified) {
    temp_dir dir("url");
    fake_server srv;
    const std::string url = "http://example.invalid/m/tiny.onnx";
    srv.files[url] = model_bytes;
    const sm::model_files f = sm::detail::fetch_model_with(srv.fetch(), url, opts(dir));
    ASSERT_TRUE(f.sha256.empty());
    ASSERT_TRUE(f.checksum.empty());
    ASSERT_TRUE(f.readme.empty());
    ASSERT_STREQ(slurp(f.model).c_str(), model_bytes.c_str());
    ASSERT_TRUE(dir.has("tiny.onnx"));
    ASSERT_FALSE(dir.has("tiny.onnx.sha256"));
}

UTEST(fetch_model, explicit_url_with_malformed_checksum_is_still_an_error) {
    temp_dir dir("urlhtml");
    fake_server srv;
    const std::string url = "http://example.invalid/m/tiny.onnx";
    srv.files[url] = model_bytes;
    srv.files[url + ".sha256"] = "<html>soft 404</html>";
    ASSERT_FALSE(error_of([&] { sm::detail::fetch_model_with(srv.fetch(), url, opts(dir)); }).empty());
    ASSERT_EQ(dir.listing().size(), 0u);
}

UTEST(fetch_model, file_name_ignores_query_string) {
    temp_dir dir("query");
    fake_server srv;
    const std::string url = "https://example.invalid/dl/tiny.onnx?token=1";
    srv.files[url] = model_bytes;
    srv.files[url + ".sha256"] = sum_line(model_bytes, "tiny.onnx");
    srv.files[url + ".readme"] = readme_text;
    const sm::model_files f = sm::detail::fetch_model_with(srv.fetch(), url, opts(dir));
    ASSERT_TRUE(f.model == dir.path / "tiny.onnx");
}

// ---- the cache ---------------------------------------------------------------------------------

UTEST(fetch_model, second_call_uses_the_cache) {
    temp_dir dir("cache");
    fake_server srv;
    srv.publish(release_tiny, "tiny.onnx");
    sm::detail::fetch_model_with(srv.fetch(), "tiny.onnx", opts(dir));
    srv.requests.clear();
    const sm::model_files f = sm::detail::fetch_model_with(srv.fetch(), "tiny.onnx", opts(dir));
    ASSERT_EQ(srv.requests.size(), 0u);
    ASSERT_FALSE(f.sha256.empty());
    ASSERT_FALSE(f.readme.empty());
}

UTEST(fetch_model, ignore_cache_downloads_again_and_replaces) {
    temp_dir dir("ignore");
    fake_server srv;
    srv.publish(release_tiny, "tiny.onnx");
    sm::detail::fetch_model_with(srv.fetch(), "tiny.onnx", opts(dir));

    const std::string newer = "a newer version of the weights";
    srv.publish(release_tiny, "tiny.onnx", newer);
    srv.requests.clear();
    const sm::model_files f = sm::detail::fetch_model_with(srv.fetch(), "tiny.onnx", opts(dir, /*ignore_cache=*/true));
    ASSERT_EQ(srv.requests.size(), 3u);
    ASSERT_STREQ(slurp(f.model).c_str(), newer.c_str());
    ASSERT_STREQ(f.sha256.c_str(), speech::utils::sha256_hex(newer.data(), newer.size()).c_str());
    ASSERT_EQ(dir.listing().size(), 3u);
}

UTEST(fetch_model, failed_refresh_keeps_the_old_cache) {
    temp_dir dir("keep");
    fake_server srv;
    srv.publish(release_tiny, "tiny.onnx");
    sm::detail::fetch_model_with(srv.fetch(), "tiny.onnx", opts(dir));

    srv.files[release_tiny + ".sha256"] = "<html>error page</html>";  // the server breaks
    ASSERT_FALSE(error_of([&] { sm::detail::fetch_model_with(srv.fetch(), "tiny.onnx", opts(dir, true)); }).empty());
    ASSERT_STREQ(slurp(dir.path / "tiny.onnx").c_str(), model_bytes.c_str());
    ASSERT_STREQ(slurp(dir.path / "tiny.onnx.readme").c_str(), readme_text.c_str());
    ASSERT_EQ(dir.listing().size(), 3u);
}

UTEST(fetch_model, corrupted_cached_model_is_downloaded_again) {
    temp_dir dir("corrupt");
    fake_server srv;
    srv.publish(release_tiny, "tiny.onnx");
    sm::detail::fetch_model_with(srv.fetch(), "tiny.onnx", opts(dir));
    {
        std::ofstream out(dir.path / "tiny.onnx", std::ios::binary | std::ios::trunc);
        out << "<html>half a download</html>";
    }
    srv.requests.clear();
    const sm::model_files f = sm::detail::fetch_model_with(srv.fetch(), "tiny.onnx", opts(dir));
    ASSERT_STREQ(slurp(f.model).c_str(), model_bytes.c_str());
    ASSERT_TRUE(!srv.requests.empty());
}

UTEST(fetch_model, damaged_cached_checksum_file_is_replaced) {
    temp_dir dir("badsum");
    fake_server srv;
    srv.publish(release_tiny, "tiny.onnx");
    sm::detail::fetch_model_with(srv.fetch(), "tiny.onnx", opts(dir));
    {
        std::ofstream out(dir.path / "tiny.onnx.sha256", std::ios::binary | std::ios::trunc);
        out << "<html>oops</html>";
    }
    const sm::model_files f = sm::detail::fetch_model_with(srv.fetch(), "tiny.onnx", opts(dir));
    ASSERT_STREQ(slurp(dir.path / "tiny.onnx.sha256").c_str(), sum_line(model_bytes, "tiny.onnx").c_str());
    ASSERT_FALSE(f.sha256.empty());
}

UTEST(fetch_model, legacy_cache_gets_its_sidecars_and_keeps_the_model) {
    temp_dir dir("legacy");
    {
        std::ofstream out(dir.path / "tiny.onnx", std::ios::binary);
        out << model_bytes;  // cached by an older version: the model alone
    }
    fake_server srv;
    srv.publish(release_tiny, "tiny.onnx");
    const sm::model_files f = sm::detail::fetch_model_with(srv.fetch(), "tiny.onnx", opts(dir));
    ASSERT_EQ(srv.requests.size(), 2u);  // checksum + readme, not the model
    ASSERT_STREQ(srv.requests[0].c_str(), (release_tiny + ".sha256").c_str());
    ASSERT_STREQ(srv.requests[1].c_str(), (release_tiny + ".readme").c_str());
    ASSERT_FALSE(f.sha256.empty());
    ASSERT_EQ(dir.listing().size(), 3u);
}

UTEST(fetch_model, legacy_cache_that_fails_the_checksum_is_downloaded_again) {
    temp_dir dir("legacybad");
    {
        std::ofstream out(dir.path / "tiny.onnx", std::ios::binary);
        out << "stale or damaged file from an older version";
    }
    fake_server srv;
    srv.publish(release_tiny, "tiny.onnx");
    const sm::model_files f = sm::detail::fetch_model_with(srv.fetch(), "tiny.onnx", opts(dir));
    ASSERT_STREQ(slurp(f.model).c_str(), model_bytes.c_str());
}

UTEST(fetch_model, cached_model_still_works_offline_when_it_has_no_checksum_yet) {
    temp_dir dir("offline");
    {
        std::ofstream out(dir.path / "tiny.onnx", std::ios::binary);
        out << model_bytes;
    }
    fake_server srv;  // nothing is reachable
    const sm::model_files f = sm::detail::fetch_model_with(srv.fetch(), "tiny.onnx", opts(dir));
    ASSERT_STREQ(slurp(f.model).c_str(), model_bytes.c_str());
    ASSERT_TRUE(f.sha256.empty());  // not verified, and it says so
}

// ---- what the transport is asked for -------------------------------------------------------------

static std::size_t count_staged(const fs::path& dir) {
    std::size_t n = 0;
    for (const auto& e : fs::directory_iterator(dir)) {
        if (e.path().extension() == ".new") ++n;
    }
    return n;
}

UTEST(fetch_model, the_transport_gets_the_checksum_the_model_must_match_and_force_only_for_ignore_cache) {
    temp_dir dir("request");
    fake_server srv;
    srv.publish(release_tiny, "tiny.onnx");
    const std::string digest = speech::utils::sha256_hex(model_bytes.data(), model_bytes.size());
    sm::detail::fetch_model_with(srv.fetch(), "tiny.onnx", opts(dir));
    ASSERT_EQ(srv.asked.size(), 3u);
    // the sidecars: always fresh, no checksum of their own, no progress bar, staged under *.new
    ASSERT_TRUE(srv.asked[0].force);
    ASSERT_TRUE(srv.asked[0].sha256.empty());
    ASSERT_FALSE(srv.asked[0].show_progress);
    ASSERT_STREQ(srv.asked[0].dest.extension().string().c_str(), ".new");
    ASSERT_TRUE(srv.asked[0].dest.filename().string().rfind("tiny.onnx.sha256.", 0) == 0);
    ASSERT_STREQ(srv.asked[1].dest.extension().string().c_str(), ".new");
    ASSERT_TRUE(srv.asked[1].dest.filename().string().rfind("tiny.onnx.readme.", 0) == 0);
    // the model: verified against the checksum file, straight to its final path, with a progress bar
    ASSERT_STREQ(srv.asked[2].sha256.c_str(), digest.c_str());
    ASSERT_TRUE(srv.asked[2].dest == dir.path / "tiny.onnx");
    ASSERT_TRUE(srv.asked[2].show_progress);
    ASSERT_FALSE(srv.asked[2].force);  // a waiter must be able to take the finished file of another process

    sm::detail::fetch_model_with(srv.fetch(), "tiny.onnx", opts(dir, /*ignore_cache=*/true));
    ASSERT_EQ(srv.asked.size(), 6u);
    ASSERT_TRUE(srv.asked[5].force);
}

UTEST(fetch_model, an_explicit_url_without_checksum_asks_for_no_verification) {
    temp_dir dir("urlrequest");
    fake_server srv;
    const std::string url = "http://example.invalid/m/tiny.onnx";
    srv.files[url] = model_bytes;
    sm::detail::fetch_model_with(srv.fetch(), url, opts(dir));
    ASSERT_STREQ(srv.requests.back().c_str(), url.c_str());
    ASSERT_TRUE(srv.asked.back().sha256.empty());
}

UTEST(fetch_model, the_sidecars_are_installed_only_after_the_model_has_arrived) {
    temp_dir dir("staged");
    fake_server srv;
    srv.publish(release_tiny, "tiny.onnx");
    bool checked = false;
    srv.on_fetch = [&](const std::string& url) {
        if (url != release_tiny) return;  // the moment the model is requested
        checked = true;
        ASSERT_EQ(count_staged(dir.path), 2u);
        ASSERT_FALSE(dir.has("tiny.onnx.sha256"));
        ASSERT_FALSE(dir.has("tiny.onnx.readme"));
    };
    sm::detail::fetch_model_with(srv.fetch(), "tiny.onnx", opts(dir));
    ASSERT_TRUE(checked);
    ASSERT_TRUE(dir.has("tiny.onnx.sha256"));
    ASSERT_EQ(count_staged(dir.path), 0u);
}

UTEST(fetch_model, a_failed_refresh_never_replaces_the_old_checksum_of_the_old_model) {
    temp_dir dir("keepsum");
    fake_server srv;
    srv.publish(release_tiny, "tiny.onnx");
    sm::detail::fetch_model_with(srv.fetch(), "tiny.onnx", opts(dir));

    const std::string newer = "a newer version of the weights";
    srv.publish(release_tiny, "tiny.onnx", newer);
    srv.files.erase(release_tiny);  // checksum and readme are new, but the model cannot be fetched
    ASSERT_FALSE(error_of([&] { sm::detail::fetch_model_with(srv.fetch(), "tiny.onnx", opts(dir, true)); }).empty());
    ASSERT_STREQ(slurp(dir.path / "tiny.onnx").c_str(), model_bytes.c_str());
    ASSERT_STREQ(slurp(dir.path / "tiny.onnx.sha256").c_str(), sum_line(model_bytes, "tiny.onnx").c_str());
    ASSERT_EQ(dir.listing().size(), 3u);  // and no *.new left behind
}

UTEST(fetch_model, two_downloads_of_the_same_model_never_share_a_staged_file) {
    // They would install (move away) each other's file: with a shared name one could find its checksum gone.
    temp_dir dir("unique");
    fake_server srv;
    srv.publish(release_tiny, "tiny.onnx");
    sm::detail::fetch_model_with(srv.fetch(), "tiny.onnx", opts(dir));
    sm::detail::fetch_model_with(srv.fetch(), "tiny.onnx", opts(dir, /*ignore_cache=*/true));
    ASSERT_EQ(srv.asked.size(), 6u);
    ASSERT_TRUE(srv.asked[0].dest != srv.asked[3].dest);  // checksum, first vs second download
    ASSERT_TRUE(srv.asked[1].dest != srv.asked[4].dest);  // readme
}
