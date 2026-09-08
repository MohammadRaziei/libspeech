// Benchmark for speech::dsp::STFT -- specifically to measure whether
// AudioFlux's OpenMP-parallel-frame path (see cmake/AudioFlux.cmake's
// LIBSPEECH_ENABLE_OPENMP option in the main repo) actually helps on real
// multi-core hardware. STFT frames are independent, so this should scale
// with core count -- but couldn't be verified in the sandbox this was
// developed in (single CPU core available there; see checklist.md).
//
// This benchmark fetches libspeech from GitHub via FetchContent (see
// CMakeLists.txt) -- it is not built as part of the main project. Usage:
//
//   cd benchmarks
//   cmake -S . -B build-bench -DCMAKE_BUILD_TYPE=Release
//   cmake --build build-bench --target bench_stft
//   ./build-bench/cpp/bench_stft
//
// To A/B test OpenMP, pass -DLIBSPEECH_ENABLE_OPENMP=ON or =OFF at the
// `cmake -S . -B build-bench` step (forwarded into the fetched libspeech
// subproject) and rebuild.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "libspeech/dsp/stft.h"

int main() {
    const int sampleRate = 16000;
    const int durationSeconds = 60;
    const int n = sampleRate * durationSeconds;

    std::vector<float> signal(n);
    for (int i = 0; i < n; ++i) {
        signal[i] = std::sin(2.0 * M_PI * 440.0 * i / sampleRate);
    }

    speech::dsp::STFT stft(10);  // fftLength=1024, slideLength=256 (default)
    const int numFrames = stft.calTimeLength(n);
    printf("signal: %d samples (%ds @ %dHz), numFrames=%d, fftLength=%d\n", n, durationSeconds,
           sampleRate, numFrames, stft.fftLength());

    // Warm-up runs (not timed) -- excludes cold-cache/CPU-frequency-ramp-up
    // noise, which otherwise dominates the first couple of timed runs and
    // makes single-run comparisons unreliable.
    for (int i = 0; i < 3; ++i) {
        auto result = stft.stft(signal);
    }

    const int repeats = 20;
    double totalMs = 0.0, minMs = 1e9, maxMs = 0.0;
    for (int r = 0; r < repeats; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto result = stft.stft(signal);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        totalMs += ms;
        minMs = std::min(minMs, ms);
        maxMs = std::max(maxMs, ms);
    }

    printf("avg=%.2fms min=%.2fms max=%.2fms over %d runs (after warm-up)\n", totalMs / repeats,
           minMs, maxMs, repeats);
    return 0;
}
