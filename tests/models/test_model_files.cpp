#include "utest.h"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
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
    std::function<void(const std::string&)> on_fetch;  // called for every request, e.g. to be slow
    std::mutex mutex;                                   // the concurrency tests call fetch() from several threads

    sm::detail::fetch_fn fetch() {
        return [this](const std::string& url, const fs::path& dest) {
            if (on_fetch) on_fetch(url);
            std::string body;
            {
                std::lock_guard<std::mutex> guard(mutex);
                requests.push_back(url);
                const auto it = files.find(url);
                if (it == files.end()) return false;
                body = it->second;
            }
            std::ofstream out(dest, std::ios::binary);
            out << body;
            return true;
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

// ---- several callers at once (the download lock) -------------------------------------------------

namespace {

using std::chrono::milliseconds;

// Short timings, so that the tests are quick; max_wait is small so that a bug shows up as a failure, not a hang.
sm::detail::lock_timing fast_locks() {
    sm::detail::lock_timing t;
    t.heartbeat = milliseconds(50);
    t.stale_after = milliseconds(2000);
    t.poll = milliseconds(20);
    t.max_wait = milliseconds(6000);
    return t;
}

// Pretends to be another process that holds the lock: rewrites <lock>/alive until stopped.
struct lock_owner {
    fs::path lock;
    std::atomic<bool> beating{true};
    std::thread thread;
    explicit lock_owner(const fs::path& lock_dir) : lock(lock_dir) {
        fs::create_directories(lock);
        std::ofstream(lock / "alive") << "1 0\n";
        thread = std::thread([this] {
            while (beating) {
                std::ofstream(lock / "alive", std::ios::trunc) << "1 1\n";
                std::this_thread::sleep_for(milliseconds(20));
            }
        });
    }
    void stop() {
        beating = false;
        if (thread.joinable()) thread.join();
    }
    ~lock_owner() { stop(); }
};

}  // namespace

UTEST(download_lock, concurrent_callers_download_the_model_once) {
    temp_dir dir("concurrent");
    fake_server srv;
    srv.publish(release_tiny, "tiny.onnx");
    srv.on_fetch = [](const std::string& url) {
        if (url == release_tiny) std::this_thread::sleep_for(milliseconds(300));  // long enough to overlap
    };
    constexpr int callers = 4;
    std::vector<fs::path> results(callers);
    std::vector<std::string> errors(callers);
    std::vector<std::thread> threads;
    for (int i = 0; i < callers; ++i) {
        threads.emplace_back([&, i] {
            try {
                results[i] = sm::detail::fetch_model_with(srv.fetch(), "tiny.onnx", opts(dir), fast_locks()).model;
            } catch (const std::exception& e) {
                errors[i] = e.what();
            }
        });
    }
    for (auto& t : threads) t.join();
    for (int i = 0; i < callers; ++i) {
        ASSERT_TRUE(errors[i].empty());
        ASSERT_STREQ(slurp(results[i]).c_str(), model_bytes.c_str());
    }
    ASSERT_EQ(srv.requests.size(), 3u);  // checksum, model, readme: once, not once per caller
    ASSERT_EQ(dir.listing().size(), 3u);  // no lock directory, no .part files
}

UTEST(download_lock, a_lock_without_heartbeat_belongs_to_a_dead_process_and_is_taken_over) {
    temp_dir dir("stale");
    fake_server srv;
    srv.publish(release_tiny, "tiny.onnx");
    const fs::path lock = dir.path / "tiny.onnx.lock";
    fs::create_directories(lock);
    std::ofstream(lock / "alive") << "999999 7\n";
    const auto long_ago = fs::file_time_type::clock::now() - std::chrono::hours(1);
    fs::last_write_time(lock / "alive", long_ago);
    fs::last_write_time(lock, long_ago);

    const auto start = std::chrono::steady_clock::now();
    const sm::model_files f = sm::detail::fetch_model_with(srv.fetch(), "tiny.onnx", opts(dir), fast_locks());
    const auto waited = std::chrono::duration_cast<milliseconds>(std::chrono::steady_clock::now() - start);
    ASSERT_STREQ(slurp(f.model).c_str(), model_bytes.c_str());
    ASSERT_TRUE(waited < milliseconds(1500));  // it did not wait for the dead owner
    ASSERT_FALSE(dir.has("tiny.onnx.lock"));
    ASSERT_EQ(dir.listing().size(), 3u);
}

UTEST(download_lock, a_live_lock_makes_the_caller_wait_until_it_is_released) {
    temp_dir dir("wait");
    fake_server srv;
    srv.publish(release_tiny, "tiny.onnx");
    lock_owner owner(dir.path / "tiny.onnx.lock");

    std::atomic<bool> done{false};
    fs::path result;
    std::thread caller([&] {
        result = sm::detail::fetch_model_with(srv.fetch(), "tiny.onnx", opts(dir), fast_locks()).model;
        done = true;
    });
    std::this_thread::sleep_for(milliseconds(400));  // many polls, and inside stale_after
    const bool still_waiting = !done.load();
    const std::size_t requests_while_waiting = srv.request_count();
    owner.stop();
    fs::remove_all(dir.path / "tiny.onnx.lock");  // the owner is finished
    caller.join();

    ASSERT_TRUE(still_waiting);
    ASSERT_EQ(requests_while_waiting, 0u);
    ASSERT_STREQ(slurp(result).c_str(), model_bytes.c_str());
}

UTEST(download_lock, a_waiter_takes_the_model_the_lock_owner_put_in_the_cache) {
    temp_dir dir("handover");
    fake_server srv;
    srv.publish(release_tiny, "tiny.onnx");
    lock_owner owner(dir.path / "tiny.onnx.lock");

    std::atomic<bool> done{false};
    std::thread caller([&] {
        sm::detail::fetch_model_with(srv.fetch(), "tiny.onnx", opts(dir), fast_locks());
        done = true;
    });
    std::this_thread::sleep_for(milliseconds(200));
    // The owner finishes: it installs the three files, then releases the lock.
    {
        std::ofstream(dir.path / "tiny.onnx", std::ios::binary) << model_bytes;
        std::ofstream(dir.path / "tiny.onnx.sha256", std::ios::binary) << sum_line(model_bytes, "tiny.onnx");
        std::ofstream(dir.path / "tiny.onnx.readme", std::ios::binary) << readme_text;
    }
    owner.stop();
    fs::remove_all(dir.path / "tiny.onnx.lock");
    caller.join();
    ASSERT_TRUE(done.load());
    ASSERT_EQ(srv.requests.size(), 0u);  // nothing was downloaded: the waiter used what the owner left
}

UTEST(download_lock, a_waiter_gives_up_on_a_live_lock_after_max_wait_and_leaves_that_lock_alone) {
    temp_dir dir("giveup");
    fake_server srv;
    srv.publish(release_tiny, "tiny.onnx");
    lock_owner owner(dir.path / "tiny.onnx.lock");
    sm::detail::lock_timing t = fast_locks();
    t.max_wait = milliseconds(300);
    const sm::model_files f = sm::detail::fetch_model_with(srv.fetch(), "tiny.onnx", opts(dir), t);
    ASSERT_STREQ(slurp(f.model).c_str(), model_bytes.c_str());
    ASSERT_TRUE(dir.has("tiny.onnx.lock"));  // it is not ours: we must not delete it
    owner.stop();
}

UTEST(download_lock, leftovers_of_killed_downloads_are_removed) {
    temp_dir dir("orphans");
    for (const char* name : {"tiny.onnx.dead-1.part", "tiny.onnx.sha256.dead-2.part", "tiny.onnx.readme.part"}) {
        std::ofstream(dir.path / name) << "half a download";
    }
    std::ofstream(dir.path / "other.onnx.dead-3.part") << "belongs to another model";
    fake_server srv;
    srv.publish(release_tiny, "tiny.onnx");
    sm::detail::fetch_model_with(srv.fetch(), "tiny.onnx", opts(dir), fast_locks());
    ASSERT_FALSE(dir.has("tiny.onnx.dead-1.part"));
    ASSERT_FALSE(dir.has("tiny.onnx.sha256.dead-2.part"));
    ASSERT_FALSE(dir.has("tiny.onnx.readme.part"));
    ASSERT_TRUE(dir.has("other.onnx.dead-3.part"));
    ASSERT_EQ(dir.listing().size(), 4u);  // the model's three files + the unrelated leftover
}

UTEST(download_lock, a_cache_hit_does_not_touch_the_lock) {
    temp_dir dir("nolock");
    fake_server srv;
    srv.publish(release_tiny, "tiny.onnx");
    sm::detail::fetch_model_with(srv.fetch(), "tiny.onnx", opts(dir), fast_locks());
    // Someone else holds the lock (e.g. a refresh with ignore_cache): reading the cache must not wait for it.
    lock_owner owner(dir.path / "tiny.onnx.lock");
    const auto start = std::chrono::steady_clock::now();
    sm::detail::fetch_model_with(srv.fetch(), "tiny.onnx", opts(dir), fast_locks());
    const auto took = std::chrono::duration_cast<milliseconds>(std::chrono::steady_clock::now() - start);
    owner.stop();
    ASSERT_TRUE(took < milliseconds(1000));
}

UTEST(download_lock, the_owner_keeps_its_lock_alive_for_as_long_as_it_downloads) {
    temp_dir dir("keepalive");
    fake_server srv;
    srv.publish(release_tiny, "tiny.onnx");
    srv.on_fetch = [](const std::string& url) {
        if (url == release_tiny) std::this_thread::sleep_for(milliseconds(1200));  // 3x stale_after
    };
    sm::detail::lock_timing t = fast_locks();
    t.stale_after = milliseconds(400);
    std::vector<std::thread> threads;
    for (int i = 0; i < 2; ++i) {
        threads.emplace_back([&] { sm::detail::fetch_model_with(srv.fetch(), "tiny.onnx", opts(dir), t); });
    }
    for (auto& th : threads) th.join();
    // Without the heartbeat the second caller would think the first one died and download again.
    ASSERT_EQ(srv.requests.size(), 3u);
}

UTEST(download_lock, if_our_temporary_files_vanish_the_finished_download_of_the_other_process_is_used) {
    temp_dir dir("vanish");
    fake_server srv;
    srv.publish(release_tiny, "tiny.onnx");
    srv.on_fetch = [&](const std::string& url) {
        if (url != release_tiny + ".readme") return;
        // Meanwhile "another process" took over our lock (it thought we had died), deleted our temporary files
        // and finished the download itself.
        std::vector<fs::path> parts;
        for (const auto& e : fs::directory_iterator(dir.path)) {
            if (e.path().extension() == ".part") parts.push_back(e.path());
        }
        for (const auto& p : parts) fs::remove(p);
        std::ofstream(dir.path / "tiny.onnx", std::ios::binary) << model_bytes;
        std::ofstream(dir.path / "tiny.onnx.sha256", std::ios::binary) << sum_line(model_bytes, "tiny.onnx");
        std::ofstream(dir.path / "tiny.onnx.readme", std::ios::binary) << readme_text;
    };
    const sm::model_files f = sm::detail::fetch_model_with(srv.fetch(), "tiny.onnx", opts(dir), fast_locks());
    ASSERT_STREQ(slurp(f.model).c_str(), model_bytes.c_str());
    ASSERT_FALSE(f.sha256.empty());
    ASSERT_EQ(srv.requests.size(), 3u);  // the other process's checksum file was kept, not deleted and fetched again
    ASSERT_STREQ(slurp(dir.path / "tiny.onnx.sha256").c_str(), sum_line(model_bytes, "tiny.onnx").c_str());
    ASSERT_EQ(dir.listing().size(), 3u);
}
