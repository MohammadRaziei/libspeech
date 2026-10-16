// SHA-256 helpers.
//
// Two implementations of the same hash, picked at run time:
//  * x86 CPUs with the SHA extensions (GCC/Clang only): the SHA-NI instructions, ~4x faster;
//  * everything else (other CPUs, ARM, MSVC): PicoSHA2 (MIT, vendored in src/third_party/picosha2).
// The model files libspeech checks are up to ~130 MB and are hashed every time a cached model is loaded,
// so speed matters here. Both paths are tested against the same reference digests.
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>

#include "libspeech/utils/utils.h"
#include "picosha2/picosha2.h"

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#define SPEECH_SHA256_SHANI 1
#include <cpuid.h>
#include <immintrin.h>
#endif

namespace speech::utils {

namespace {

constexpr std::size_t block_size = 64;

#ifdef SPEECH_SHA256_SHANI

alignas(16) const std::uint32_t k256[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

const std::uint32_t initial_state[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};

bool cpu_has_sha_extensions() {
    unsigned a = 0, b = 0, c = 0, d = 0;
    if (!__get_cpuid(1, &a, &b, &c, &d)) return false;
    const bool ssse3 = (c & (1u << 9)) != 0;
    const bool sse41 = (c & (1u << 19)) != 0;
    if (!ssse3 || !sse41) return false;
    if (__get_cpuid_max(0, nullptr) < 7) return false;
    __cpuid_count(7, 0, a, b, c, d);
    return (b & (1u << 29)) != 0;  // CPUID.(EAX=7,ECX=0):EBX.SHA
}

// Processes `blocks` consecutive 64-byte blocks. state = a..h as in FIPS 180-4.
__attribute__((target("sha,sse4.1,ssse3"))) void compress_shani(std::uint32_t state[8], const std::uint8_t* data,
                                                                  std::size_t blocks) {
    const __m128i byte_swap = _mm_set_epi64x(0x0c0d0e0f08090a0bULL, 0x0405060700010203ULL);

    // The SHA instructions want the state as (A,B,E,F) and (C,D,G,H).
    __m128i tmp = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&state[0]));
    __m128i state1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&state[4]));
    tmp = _mm_shuffle_epi32(tmp, 0xB1);
    state1 = _mm_shuffle_epi32(state1, 0x1B);
    __m128i state0 = _mm_alignr_epi8(tmp, state1, 8);
    state1 = _mm_blend_epi16(state1, tmp, 0xF0);

    while (blocks-- > 0) {
        const __m128i abef_save = state0;
        const __m128i cdgh_save = state1;
        __m128i w[4];  // the last four groups of four message words, as a ring
        for (int j = 0; j < 16; ++j) {
            if (j < 4) {
                w[j] = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(data + 16 * j)), byte_swap);
            } else {
                // w[j] from w[j-4], w[j-3], w[j-2], w[j-1] (ring slots j&3, (j+1)&3, (j+2)&3, (j+3)&3)
                __m128i next = _mm_sha256msg1_epu32(w[j & 3], w[(j + 1) & 3]);
                next = _mm_add_epi32(next, _mm_alignr_epi8(w[(j + 3) & 3], w[(j + 2) & 3], 4));
                w[j & 3] = _mm_sha256msg2_epu32(next, w[(j + 3) & 3]);
            }
            __m128i msg = _mm_add_epi32(w[j & 3], _mm_load_si128(reinterpret_cast<const __m128i*>(&k256[4 * j])));
            state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
            msg = _mm_shuffle_epi32(msg, 0x0E);
            state0 = _mm_sha256rnds2_epu32(state0, state1, msg);
        }
        state0 = _mm_add_epi32(state0, abef_save);
        state1 = _mm_add_epi32(state1, cdgh_save);
        data += block_size;
    }

    tmp = _mm_shuffle_epi32(state0, 0x1B);
    state1 = _mm_shuffle_epi32(state1, 0xB1);
    state0 = _mm_blend_epi16(tmp, state1, 0xF0);
    state1 = _mm_alignr_epi8(state1, tmp, 8);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(&state[0]), state0);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(&state[4]), state1);
}

class shani_hasher {
   public:
    shani_hasher() { std::memcpy(h_, initial_state, sizeof h_); }

    void update(const std::uint8_t* p, std::size_t n) {
        total_ += n;
        if (buffered_ > 0) {
            const std::size_t take = std::min(n, block_size - buffered_);
            std::memcpy(buf_ + buffered_, p, take);
            buffered_ += take;
            p += take;
            n -= take;
            if (buffered_ < block_size) return;
            compress_shani(h_, buf_, 1);
            buffered_ = 0;
        }
        if (n >= block_size) {
            compress_shani(h_, p, n / block_size);
            p += n - n % block_size;
            n %= block_size;
        }
        if (n > 0) {
            std::memcpy(buf_, p, n);
            buffered_ = n;
        }
    }

    std::string finish() {
        const std::uint64_t bits = total_ * 8;
        std::uint8_t pad[2 * block_size] = {0x80};
        const std::size_t pad_len = (buffered_ < 56 ? 56 : 120) - buffered_;
        for (int i = 0; i < 8; ++i) pad[pad_len + i] = static_cast<std::uint8_t>(bits >> (56 - 8 * i));
        const std::uint64_t saved_total = total_;
        update(pad, pad_len + 8);
        total_ = saved_total;
        static const char* digits = "0123456789abcdef";
        std::string hex;
        for (std::uint32_t word : h_) {
            for (int shift = 28; shift >= 0; shift -= 4) hex += digits[(word >> shift) & 0xF];
        }
        return hex;
    }

   private:
    std::uint32_t h_[8];
    std::uint8_t buf_[block_size];
    std::size_t buffered_ = 0;
    std::uint64_t total_ = 0;
};

#endif  // SPEECH_SHA256_SHANI

// One streaming hash, whichever implementation is in use.
class hasher {
   public:
    hasher() {
#ifdef SPEECH_SHA256_SHANI
        if (hardware_enabled().load(std::memory_order_relaxed) && hardware_available()) {
            fast_ = std::make_unique<shani_hasher>();
            return;
        }
#endif
    }

    void update(const void* data, std::size_t n) {
        const auto* p = static_cast<const std::uint8_t*>(data);
        if (n == 0) return;
#ifdef SPEECH_SHA256_SHANI
        if (fast_) {
            fast_->update(p, n);
            return;
        }
#endif
        portable_.process(p, p + n);
    }

    std::string finish() {
#ifdef SPEECH_SHA256_SHANI
        if (fast_) return fast_->finish();
#endif
        portable_.finish();
        return picosha2::get_hash_hex_string(portable_);
    }

    static std::atomic<bool>& hardware_enabled() {
        static std::atomic<bool> enabled{true};
        return enabled;
    }

    static bool hardware_available() {
#ifdef SPEECH_SHA256_SHANI
        static const bool available = cpu_has_sha_extensions();
        return available;
#else
        return false;
#endif
    }

   private:
    picosha2::hash256_one_by_one portable_;
#ifdef SPEECH_SHA256_SHANI
    std::unique_ptr<shani_hasher> fast_;
#endif
};

}  // namespace

const char* sha256_backend() noexcept {
    return hasher::hardware_available() && hasher::hardware_enabled().load() ? "sha-ni" : "portable";
}

void set_sha256_hardware_enabled(bool enabled) noexcept { hasher::hardware_enabled().store(enabled); }

std::string sha256_hex(const void* data, std::size_t size) {
    hasher h;
    h.update(data, size);
    return h.finish();
}

std::string sha256_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Cannot open file for hashing: " + path.string());
    }
    hasher h;
    std::array<char, 1 << 20> chunk;  // 1 MiB
    while (in) {
        in.read(chunk.data(), static_cast<std::streamsize>(chunk.size()));
        const std::streamsize got = in.gcount();
        if (got > 0) h.update(chunk.data(), static_cast<std::size_t>(got));
    }
    if (in.bad()) {
        throw std::runtime_error("Error while reading file for hashing: " + path.string());
    }
    return h.finish();
}

}  // namespace speech::utils
