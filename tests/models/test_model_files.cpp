#include "utest.h"

#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "libspeech/models/model_files.h"
#include "libspeech/utils/utils.h"
#include "model_files_detail.h"

// The download/verify logic runs here against a fake transport (no network): a map url -> body, plus a
// log of every request, so the tests can check what was asked for, in which order, and what was left behind.

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

    sm::detail::fetch_fn fetch() {
        return [this](const std::string& url, const fs::path& dest) {
            requests.push_back(url);
            const auto it = files.find(url);
            if (it == files.end()) return false;
            std::ofstream out(dest, std::ios::binary);
            out << it->second;
            return true;
        };
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

UTEST(fetch_model, downloads_checksum_then_model_then_readme) {
    temp_dir dir("order");
    fake_server srv;
    srv.publish(release_tiny, "tiny.onnx");
    const sm::model_files f = sm::detail::fetch_model_with(srv.fetch(), "tiny.onnx", opts(dir));

    ASSERT_EQ(srv.requests.size(), 3u);
    ASSERT_STREQ(srv.requests[0].c_str(), (release_tiny + ".sha256").c_str());  // checksum first
    ASSERT_STREQ(srv.requests[1].c_str(), release_tiny.c_str());
    ASSERT_STREQ(srv.requests[2].c_str(), (release_tiny + ".readme").c_str());

    ASSERT_STREQ(slurp(f.model).c_str(), model_bytes.c_str());
    ASSERT_STREQ(slurp(f.readme).c_str(), readme_text.c_str());
    ASSERT_STREQ(f.sha256.c_str(), speech::utils::sha256_hex(model_bytes.data(), model_bytes.size()).c_str());
    ASSERT_TRUE(f.checksum == dir.path / "tiny.onnx.sha256");
    ASSERT_EQ(dir.listing().size(), 3u);  // the three files, no .part leftovers
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
        ASSERT_EQ(dir.listing().size(), 0u);  // the model was good, but nothing is installed without all three
    }
}

UTEST(fetch_model, model_download_failure_leaves_nothing) {
    temp_dir dir("nomodel");
    fake_server srv;
    srv.publish(release_tiny, "tiny.onnx");
    srv.files.erase(release_tiny);
    ASSERT_STREQ(error_of([&] { sm::detail::fetch_model_with(srv.fetch(), "tiny.onnx", opts(dir)); }).c_str(),
                 "Failed to download the model.");
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
