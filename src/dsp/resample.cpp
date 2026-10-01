#include "libspeech/dsp/resample.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

#include "aixlog.hpp"
#include "dsp/resample_algorithm.h"  // Vendored AudioFlux C header (src/third_party/audioflux)

namespace speech::dsp {

namespace {
constexpr const char* kTag = "speech::dsp::Resample";

// Polyphase kernels. Output i uses weight row (i*down) mod up and the input
// window starting at n_i = floor(i*down/up): exact integer arithmetic, no float
// time accumulation. Rows are kLanes-padded with zeros.
constexpr int kLanes = 8;

// Portable kernel: 8 independent accumulators, vectorizes without -ffast-math
// because the lanes never mix until the final reduction.
void runPolyphaseGeneric(const float* weights, int wlen, int up, int down, const float* xpad,
                         float* out, std::size_t outLen) {
    const int stepN = down / up;
    const int stepP = down % up;
    int n = 0;
    int ph = 0;
    for (std::size_t i = 0; i < outLen; ++i) {
        const float* w = weights + static_cast<std::size_t>(ph) * wlen;
        const float* x = xpad + n;
        float acc[kLanes] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int k = 0; k < wlen; k += kLanes) {
            for (int l = 0; l < kLanes; ++l) acc[l] += w[k + l] * x[k + l];
        }
        out[i] = ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]));
        n += stepN;
        ph += stepP;
        if (ph >= up) {
            ph -= up;
            ++n;
        }
    }
}

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define LIBSPEECH_HAVE_AVX2_KERNEL 1
#include <immintrin.h>

__attribute__((target("avx2,fma"))) inline float hsum256(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}

// Phase-major: all outputs with the same (i mod up) share one weight row, and their
// input windows are exactly `down` samples apart, so each 8-float weight chunk
// is loaded once and FMA'd against 8 outputs (9 loads per 8 FMAs instead of 16).
__attribute__((target("avx2,fma"))) void runPolyphaseAvx2(const float* weights, int wlen, int up,
                                                          int down, const float* xpad, float* out,
                                                          std::size_t outLen) {
    const std::size_t d = static_cast<std::size_t>(down);
    for (int ph = 0; ph < up; ++ph) {
        if (static_cast<std::size_t>(ph) >= outLen) break;
        // Outputs i = ph + up*m all have phase (i*down) mod up = (ph*down) mod up, hence one
        // weight row -- a permutation of ph, NOT ph itself -- and n_i = n0 + m*down.
        const long long phDown = static_cast<long long>(ph) * down;
        const float* w = weights + static_cast<std::size_t>(phDown % up) * wlen;
        const std::size_t n0 = static_cast<std::size_t>(phDown / up);
        const std::size_t count = (outLen - static_cast<std::size_t>(ph) + up - 1) / up;
        std::size_t m = 0;
        for (; m + 8 <= count; m += 8) {
            const float* x = xpad + n0 + m * d;
            __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0,
                   a7 = a0;
            for (int k = 0; k < wlen; k += 8) {
                const __m256 wv = _mm256_load_ps(w + k);
                a0 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x + k), a0);
                a1 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x + d + k), a1);
                a2 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x + 2 * d + k), a2);
                a3 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x + 3 * d + k), a3);
                a4 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x + 4 * d + k), a4);
                a5 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x + 5 * d + k), a5);
                a6 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x + 6 * d + k), a6);
                a7 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x + 7 * d + k), a7);
            }
            float* o = out + ph + static_cast<std::size_t>(up) * m;
            o[0] = hsum256(a0);
            o[static_cast<std::size_t>(up)] = hsum256(a1);
            o[2 * static_cast<std::size_t>(up)] = hsum256(a2);
            o[3 * static_cast<std::size_t>(up)] = hsum256(a3);
            o[4 * static_cast<std::size_t>(up)] = hsum256(a4);
            o[5 * static_cast<std::size_t>(up)] = hsum256(a5);
            o[6 * static_cast<std::size_t>(up)] = hsum256(a6);
            o[7 * static_cast<std::size_t>(up)] = hsum256(a7);
        }
        for (; m < count; ++m) {  // tail: fewer than 8 outputs left for this phase
            const float* x = xpad + n0 + m * d;
            __m256 a = _mm256_setzero_ps();
            for (int k = 0; k < wlen; k += 8) {
                a = _mm256_fmadd_ps(_mm256_load_ps(w + k), _mm256_loadu_ps(x + k), a);
            }
            out[ph + static_cast<std::size_t>(up) * m] = hsum256(a);
        }
    }
}
#endif

bool cpuHasAvx2Fma() {
#ifdef LIBSPEECH_HAVE_AVX2_KERNEL
    // LIBSPEECH_DISABLE_AVX2=1 forces the portable kernel (testing / debugging).
    static const bool ok = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") &&
                           std::getenv("LIBSPEECH_DISABLE_AVX2") == nullptr;
    return ok;
#else
    return false;
#endif
}

void runPolyphase(const float* weights, int wlen, int up, int down, const float* xpad,
                  float* out, std::size_t outLen) {
#ifdef LIBSPEECH_HAVE_AVX2_KERNEL
    if (cpuHasAvx2Fma()) {
        runPolyphaseAvx2(weights, wlen, up, down, xpad, out, outLen);
        return;
    }
#endif
    runPolyphaseGeneric(weights, wlen, up, down, xpad, out, outLen);
}
}  // namespace

struct Resample::Polyphase {
    int up = 0;     // target / gcd  (number of output phases)
    int down = 0;   // source / gcd
    int lmax = 0;   // taps reaching back from the centre sample
    int rmax = 0;   // taps reaching forward
    int wlen = 0;   // lmax + rmax, rounded up to a multiple of kLanes
    std::vector<float> weightStore;  // over-allocated so rows can start 64-byte aligned
    const float* weights = nullptr;  // [up][wlen], 64-byte aligned, points into weightStore
};

void Resample::buildPolyphase() {
    poly_.reset();
    if (!resampleObj) return;
    const float* interp = nullptr;
    const float* delta = nullptr;
    int interpLength = 0, bitLength = 0, p = 0, q = 0, isScale = 0, isContinue = 0;
    float ratio = 0.0f;
    if (resampleObj_getTables(resampleObj, &interp, &delta, &interpLength, &bitLength, &ratio, &p,
                              &q, &isScale, &isContinue) != 0) {
        return;
    }
    if (isScale || isContinue || p <= 0 || q <= 0) return;

    const float scale = std::min(1.0f, ratio);
    const int step = static_cast<int>(std::floor(scale * bitLength));
    if (step <= 0) return;
    // Bound the filter bank (p phases x ~2*interpLength/step taps): rates with a
    // huge reduced ratio (e.g. 44101 -> 16000) fall back to the direct kernel.
    const int approxTaps = 2 * (interpLength / step) + 2;
    if (static_cast<long long>(p) * approxTaps > (1 << 22)) return;

    struct Side { int offset; float delta; int len; };
    auto side = [&](float factor) {
        const float fv = factor * bitLength;
        const int off = static_cast<int>(std::floor(fv));
        return Side{off, fv - static_cast<float>(off), (interpLength - off) / step};
    };

    std::vector<Side> left(p), right(p);
    int lmax = 0, rmax = 0;
    for (int ph = 0; ph < p; ++ph) {
        const float factor = scale * static_cast<float>(static_cast<double>(ph) / p);
        left[ph] = side(factor);
        right[ph] = side(scale - factor);
        lmax = std::max(lmax, left[ph].len);
        rmax = std::max(rmax, right[ph].len);
    }
    if (lmax <= 0 || rmax <= 0) return;

    auto pl = std::make_unique<Polyphase>();
    pl->up = p;
    pl->down = q;
    pl->lmax = lmax;
    pl->rmax = rmax;
    pl->wlen = (lmax + rmax + kLanes - 1) / kLanes * kLanes;
    pl->weightStore.assign(static_cast<std::size_t>(p) * pl->wlen + 16, 0.0f);
    {
        auto addr = reinterpret_cast<std::uintptr_t>(pl->weightStore.data());
        const std::size_t pad = ((64 - (addr % 64)) % 64) / sizeof(float);
        pl->weights = pl->weightStore.data() + pad;
    }
    for (int ph = 0; ph < p; ++ph) {
        float* row = const_cast<float*>(pl->weights) + static_cast<std::size_t>(ph) * pl->wlen;
        for (int j = 0; j < left[ph].len; ++j) {
            const int idx = left[ph].offset + j * step;
            row[lmax - 1 - j] = interp[idx] + left[ph].delta * delta[idx];
        }
        for (int j = 0; j < right[ph].len; ++j) {
            const int idx = right[ph].offset + j * step;
            row[lmax + j] = interp[idx] + right[ph].delta * delta[idx];
        }
    }
    poly_ = std::move(pl);
}

detail::UninitVector<float> Resample::resamplePolyphase(const float* data, std::size_t n) {
    const Polyphase& pl = *poly_;
    const int outLen = resampleObj_calDataLength(resampleObj, static_cast<int>(n));
    if (outLen <= 0) {
        throw std::runtime_error("Resampling failed or produced no output.");
    }
    // Zero-padded copy of the input so every output is one uniform dot product:
    // input index k lives at xpad[k + lmax - 1]; samples outside [0, n) are zero,
    // exactly what the direct kernel's boundary clamping amounts to.
    const std::size_t lead = static_cast<std::size_t>(pl.lmax - 1);
    detail::UninitVector<float> xpad(lead + n + static_cast<std::size_t>(pl.wlen) + 8);
    std::fill(xpad.begin(), xpad.begin() + lead, 0.0f);
    std::copy(data, data + n, xpad.begin() + lead);
    std::fill(xpad.begin() + lead + n, xpad.end(), 0.0f);

    detail::UninitVector<float> out(static_cast<std::size_t>(outLen));
    runPolyphase(pl.weights, pl.wlen, pl.up, pl.down, xpad.data(), out.data(),
                 static_cast<std::size_t>(outLen));
    return out;
}

Resample::Resample(int sourceRate, int targetRate)
    : resampleObj(nullptr), isIdentity(sourceRate == targetRate) {
    if (isIdentity) {
        LOG(DEBUG) << TAG(kTag) << "Source/target rate both " << sourceRate
                   << "Hz; using identity passthrough (skipping AudioFlux)." << std::endl;
        return;
    }

    ResampleQualityType qualType = ResampleQuality_Best;
    // isScale=0: AudioFlux's isScale=1 divides the output by sqrt(ratio), an
    // energy-domain normalization (see resampleObj_resample()). For a
    // downsample from 44100->16000 that means dividing by ~0.60, i.e.
    // amplifying the waveform by ~1.66x -- which can push an already
    // near-full-scale signal well past +/-1.0. We want amplitude-preserving
    // resampling by default for a general audio pipeline (feeding VAD,
    // denoisers, playback, etc.); advanced users who want the energy-scaled
    // behavior can use the extended constructor with isScale=true.
    // Full write-up: /audioflux_issues.md (Issue 2).
    int isScale = 0;
    int isContinue = 0;   // Disable continuous mode by default

    if (resampleObj_new(&resampleObj, &qualType, &isScale, &isContinue) != 0) {
        LOG(ERROR) << TAG(kTag) << "Failed to create resample object (" << sourceRate
                   << "Hz -> " << targetRate << "Hz)." << std::endl;
        throw std::runtime_error("Failed to create resample object.");
    }

    resampleObj_setSamplate(resampleObj, sourceRate, targetRate);
    buildPolyphase();
    LOG(DEBUG) << TAG(kTag) << "Created resampler: " << sourceRate << "Hz -> " << targetRate
               << "Hz." << std::endl;
}

Resample::Resample(int sourceRate, int targetRate, int zeroNum, int nbit, WindowType winType,
                     float value, float rollOff, bool isScale, bool isContinue)
    : resampleObj(nullptr), isIdentity(sourceRate == targetRate) {
    if (isIdentity) {
        LOG(DEBUG) << TAG(kTag) << "Source/target rate both " << sourceRate
                   << "Hz; using identity passthrough (skipping AudioFlux)." << std::endl;
        return;
    }

    int isScaleInt = isScale ? 1 : 0;
    int isContinueInt = isContinue ? 1 : 0;

    if (resampleObj_newWithWindow(&resampleObj, &zeroNum, &nbit, &winType, &value, &rollOff,
                                   &isScaleInt, &isContinueInt) != 0) {
        LOG(ERROR) << TAG(kTag) << "Failed to create resample object with custom window settings."
                   << std::endl;
        throw std::runtime_error("Failed to create resample object with custom settings.");
    }

    resampleObj_setSamplate(resampleObj, sourceRate, targetRate);
    buildPolyphase();
    LOG(DEBUG) << TAG(kTag) << "Created resampler (custom window): " << sourceRate << "Hz -> "
               << targetRate << "Hz." << std::endl;
}

Resample::~Resample() {
    reset();
}

void Resample::setSampleRateRatio(float ratio) {
    if (isIdentity) {
        LOG(DEBUG) << TAG(kTag)
                   << "setSampleRateRatio() ignored: resampler is in identity passthrough mode."
                   << std::endl;
        return;
    }
    poly_.reset();  // arbitrary float ratio: no rational phase structure
    resampleObj_setSamplateRatio(resampleObj, ratio);
    LOG(DEBUG) << TAG(kTag) << "Sample rate ratio set to " << ratio << std::endl;
}

void Resample::enableContinuous(bool flag) {
    if (isIdentity) {
        return;
    }
    poly_.reset();  // continuous mode carries tail state through the direct kernel
    resampleObj_enableContinue(resampleObj, flag ? 1 : 0);
}

detail::UninitVector<float> Resample::resampleFlat(const float* data, std::size_t n) {
    if (isIdentity) {
        return detail::UninitVector<float>(data, data + n);
    }
    if (n == 0) {
        LOG(DEBUG) << TAG(kTag) << "resample() called with empty input; returning empty output."
                   << std::endl;
        return {};
    }

    if (poly_) {
        auto out = resamplePolyphase(data, n);
        LOG(DEBUG) << TAG(kTag) << "Resampled " << n << " -> " << out.size()
                   << " samples (polyphase)." << std::endl;
        return out;
    }

    const int inputLength = static_cast<int>(n);
    const int outputLength = resampleObj_calDataLength(resampleObj, inputLength);
    LOG(TRACE) << TAG(kTag) << "calDataLength(" << inputLength << ") -> expected output length "
               << outputLength << std::endl;

    // The direct AudioFlux kernel ACCUMULATES into its output (dataArr2[i] += ...),
    // so this buffer must start zeroed -- unlike the polyphase path above, which
    // overwrites every sample.
    detail::UninitVector<float> out(static_cast<std::size_t>(outputLength));
    std::fill(out.begin(), out.end(), 0.0f);
    const int actual = resampleObj_resample(resampleObj, const_cast<float*>(data), inputLength,
                                            out.data());
    if (actual <= 0) {
        LOG(ERROR) << TAG(kTag) << "Resampling produced no output (input length=" << inputLength
                   << ")." << std::endl;
        throw std::runtime_error("Resampling failed or produced no output.");
    }
    out.resize(static_cast<std::size_t>(actual));
    LOG(DEBUG) << TAG(kTag) << "Resampled " << inputLength << " -> " << actual << " samples."
               << std::endl;
    return out;
}

std::vector<float> Resample::resample(const std::vector<float>& inputData) {
    auto flat = resampleFlat(inputData.data(), inputData.size());
    return std::vector<float>(flat.begin(), flat.end());
}

void Resample::reset() {
    if (resampleObj) {
        resampleObj_free(resampleObj);
        resampleObj = nullptr;
    }
}

}  // namespace speech::dsp
