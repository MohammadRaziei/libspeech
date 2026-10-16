#include "utest.h"

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "libspeech/utils/utils.h"

// Reference digests come from Python's hashlib (an implementation independent of the vendored PicoSHA2).

namespace {

std::string pattern(std::size_t n, int mul, int add) {
    std::string s(n, '\0');
    for (std::size_t i = 0; i < n; ++i) s[i] = static_cast<char>((i * mul + add) & 255);
    return s;
}

// Runs `check` once with the hardware implementation (if this CPU has one) and once with the portable one.
template <typename F>
void on_every_backend(F&& check) {
    namespace u = speech::utils;
    for (const bool hardware : {true, false}) {
        u::set_sha256_hardware_enabled(hardware);
        check();
    }
    u::set_sha256_hardware_enabled(true);
}

}  // namespace

UTEST(sha256, backend_switch) {
    namespace u = speech::utils;
    u::set_sha256_hardware_enabled(false);
    ASSERT_STREQ(u::sha256_backend(), "portable");
    u::set_sha256_hardware_enabled(true);
    const std::string backend = u::sha256_backend();
    ASSERT_TRUE(backend == "sha-ni" || backend == "portable");
    printf("    sha256 backend on this machine: %s\n", backend.c_str());
}

UTEST(sha256, backends_agree_for_every_length_up_to_300) {
    namespace u = speech::utils;
    for (std::size_t n = 0; n <= 300; ++n) {
        const std::string data = pattern(n, 13, 5);
        u::set_sha256_hardware_enabled(true);
        const std::string fast = u::sha256_hex(data.data(), data.size());
        u::set_sha256_hardware_enabled(false);
        const std::string portable = u::sha256_hex(data.data(), data.size());
        ASSERT_STREQ(fast.c_str(), portable.c_str());
    }
    u::set_sha256_hardware_enabled(true);
}

UTEST(sha256, nist_vectors) {
    namespace u = speech::utils;
    on_every_backend([&] {
        ASSERT_STREQ(u::sha256_hex("", 0).c_str(), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        ASSERT_STREQ(u::sha256_hex("abc", 3).c_str(), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        const std::string million(1000000, 'a');
        ASSERT_STREQ(u::sha256_hex(million.data(), million.size()).c_str(),
                     "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
    });
}

UTEST(sha256, padding_boundaries) {
    // Lengths around the 55/56 and 64-byte block edges, where padding bugs live.
    struct row { std::size_t n; const char* digest; };
    const row rows[] = {
        {55, "e7313d333c272e639f790978283f9eb392e843d0f29b7016828bb1daa4aac70b"},
        {56, "4324d65f3c103567f5589c710bc08f8523f929a9272e3af36fc968e52abc6c27"},
        {63, "81c80242132f230c3bd41b3e63bbcff16107339549214a99614ff26664625055"},
        {64, "39e3d7b6b5d075d37d053ad89b24b41bef4f3c29760c84447cab3f3be1882241"},
        {65, "aacca6ff74fdbb296d165a45cecfa04e5127bc008770fbbdd48006f2d2fae95e"},
        {119, "9ce7368e4daf32341631b492e80359dc9f594b48453cd0dd5bf0b19279cc177e"},
        {120, "7836b787757e95e58b3ca5aec90b1b004e8deba1e50e9675af9cabf1a13a04b5"},
    };
    on_every_backend([&] {
        for (const row& r : rows) {
            const std::string data = pattern(r.n, 7, 3);
            ASSERT_STREQ(speech::utils::sha256_hex(data.data(), data.size()).c_str(), r.digest);
        }
    });
}

UTEST(sha256, file_matches_reference_across_chunk_boundaries) {
    // 3 MiB + 12345 bytes: not a multiple of the 1 MiB read chunk or of the 64-byte block.
    const std::filesystem::path path = std::filesystem::temp_directory_path() / "libspeech_sha256_big.bin";
    const std::string data = pattern(3 * 1024 * 1024 + 12345, 31, 7);
    {
        std::ofstream out(path, std::ios::binary);
        out.write(data.data(), static_cast<std::streamsize>(data.size()));
    }
    on_every_backend([&] {
        ASSERT_STREQ(speech::utils::sha256_file(path).c_str(),
                     "5484bbdbfa09aaf96421d09d2c57be9d4fdbffe2ef1d33964a754591a0ac11ba");
    });
    std::filesystem::remove(path);
}

UTEST(sha256, file_errors) {
    bool threw = false;
    try {
        speech::utils::sha256_file(std::filesystem::temp_directory_path() / "libspeech_no_such_file.bin");
    } catch (const std::runtime_error&) {
        threw = true;
    }
    ASSERT_TRUE(threw);
}
