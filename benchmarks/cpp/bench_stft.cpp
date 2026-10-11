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
    const int sample_rate = 16000;
    const int duration_seconds = 60;
    const int n = sample_rate * duration_seconds;

    std::vector<float> signal(n);
    for (int i = 0; i < n; ++i) {
        signal[i] = std::sin(2.0 * M_PI * 440.0 * i / sample_rate);
    }

    speech::dsp::stft stft(10);  // fft_length=1024, slide_length=256 (default)
    const int num_frames = stft.cal_time_length(n);
    printf("signal: %d samples (%ds @ %d_hz), num_frames=%d, fft_length=%d\n", n, duration_seconds,
           sample_rate, num_frames, stft.fft_length());

    // Warm-up runs (not timed) -- excludes cold-cache/CPU-frequency-ramp-up
    // noise, which otherwise dominates the first couple of timed runs and
    // makes single-run comparisons unreliable.
    for (int i = 0; i < 3; ++i) {
        auto result = stft.compute(signal);
    }

    const int repeats = 20;
    double total_ms = 0.0, min_ms = 1e9, max_ms = 0.0;
    for (int r = 0; r < repeats; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto result = stft.compute(signal);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_ms += ms;
        min_ms = std::min(min_ms, ms);
        max_ms = std::max(max_ms, ms);
    }

    printf("avg=%.2fms min=%.2fms max=%.2fms over %d runs (after warm-up)\n", total_ms / repeats,
           min_ms, max_ms, repeats);
    return 0;
}
