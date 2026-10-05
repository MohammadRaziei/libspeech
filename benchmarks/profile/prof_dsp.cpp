// Native C++ split of the untouched DSP pieces: where does the time go once Python lists are out of the picture?
// (Used only to attribute Python-API time to "conversion" vs "compute"; never compared against Python libraries.)
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <vector>
#include "libspeech/dsp/dct.h"
#include "libspeech/dsp/fft.h"
#include "libspeech/dsp/stft.h"
#include "stft_algorithm.h"
static std::atomic<unsigned long> g_allocs{0}, g_bytes{0};
void* operator new(std::size_t n) { g_allocs++; g_bytes += n; if (void* p = std::malloc(n)) return p; throw std::bad_alloc(); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
using namespace speech::dsp;
using clk = std::chrono::high_resolution_clock;
template <class F> double best_ms(F f, int reps = 9) { f(); double b = 1e18; for (int i = 0; i < reps; ++i) { auto a = clk::now(); f(); auto c = clk::now(); b = std::fmin(b, std::chrono::duration<double, std::milli>(c - a).count()); } return b; }
template <class F> void report(const char* name, F f, int reps = 9) {
    double ms = best_ms(f, reps); g_allocs = 0; g_bytes = 0; f();
    std::printf("%-46s %9.3f ms   %4lu allocs  %9.1f KiB allocated per call\n", name, ms, g_allocs.load(), g_bytes.load() / 1024.0);
}
int main() {
    for (int expo : {9, 12, 16}) {
        const int N = 1 << expo; std::vector<float> x(N); for (int i = 0; i < N; ++i) x[i] = std::sin(0.01 * i) + 0.3f * std::sin(0.37 * i);
        FFT f(expo); auto [re, im] = f.forward(x); char buf[96];
        std::snprintf(buf, sizeof buf, "FFT::forward N=%d", N); report(buf, [&] { auto r = f.forward(x); });
        std::snprintf(buf, sizeof buf, "FFT::inverse N=%d", N); report(buf, [&] { auto r = f.inverse(re, im); });
        std::snprintf(buf, sizeof buf, "FFT::dct     N=%d", N); report(buf, [&] { auto r = f.dct(x); });
        std::snprintf(buf, sizeof buf, "FFT::idct    N=%d (copies its input)", N); report(buf, [&] { auto r = f.idct(x); });
    }
    for (int n : {512, 4096}) { std::vector<float> x(n); for (int i = 0; i < n; ++i) x[i] = std::sin(0.01 * i); char buf[96];
        std::snprintf(buf, sizeof buf, "dctII n=%d (all outputs)", n); report(buf, [&] { auto r = dctII(x, -1, true); }, 3); }
    { std::vector<float> x(26); for (int i = 0; i < 26; ++i) x[i] = std::sin(0.1 * i); report("dctII 26 -> 13 (the old per-MFCC-frame call)", [&] { auto r = dctII(x, 13, true); }, 200); }
    // istft: split wrapper vs the AudioFlux kernel
    const int n = 160000; std::vector<float> sig(n); for (int i = 0; i < n; ++i) sig[i] = std::sin(0.05 * i);
    STFT st(9, Window_Hann, 128); auto sp = st.spectrogram(sig.data(), sig.size(), false);
    std::vector<std::vector<float>> re(sp.numFrames), im(sp.numFrames);
    for (int i = 0; i < sp.numFrames; ++i) { re[i].assign(sp.real.begin() + (size_t)i * 512, sp.real.begin() + (size_t)(i + 1) * 512); im[i].assign(sp.imag.begin() + (size_t)i * 512, sp.imag.begin() + (size_t)(i + 1) * 512); }
    report("STFT::istft 10 s (vector<vector> API)", [&] { auto r = st.istft(re, im, 0); }, 5);
    ::OpaqueSTFT* raw = nullptr; int slide = 128, cont = 0; WindowType wt = Window_Hann; stftObj_new(&raw, 9, &wt, &slide, &cont);
    const int dl = stftObj_calDataLength(raw, sp.numFrames); std::vector<float> out(dl);
    report("  AudioFlux stftObj_istft kernel alone", [&] { std::fill(out.begin(), out.end(), 0.f); stftObj_istft(raw, sp.real.data(), sp.imag.data(), sp.numFrames, 0, out.data()); }, 5);
    report("  flatten vector<vector> -> flat (2 planes)", [&] { std::vector<float> a((size_t)sp.numFrames * 512), b(a.size()); for (int i = 0; i < sp.numFrames; ++i) { std::copy(re[i].begin(), re[i].end(), a.begin() + (size_t)i * 512); std::copy(im[i].begin(), im[i].end(), b.begin() + (size_t)i * 512); } }, 5);
    // the forward kernels for context: is AudioFlux's own complex FFT competitive?
    std::vector<float> x4(4096); for (int i = 0; i < 4096; ++i) x4[i] = std::sin(0.01 * i);
}
