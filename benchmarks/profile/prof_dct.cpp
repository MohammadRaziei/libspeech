// Native timing of speech::dsp::dctII (the function behind libspeech.dct) for several sizes / output counts.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>
#include "libspeech/dsp/dct.h"
using clk = std::chrono::high_resolution_clock;
template <class F> double best_ms(F f, int reps) { f(); double b = 1e18; for (int i = 0; i < reps; ++i) { auto a = clk::now(); f(); auto c = clk::now(); b = std::fmin(b, std::chrono::duration<double, std::milli>(c - a).count()); } return b; }
int main() {
    struct C { int n, k; };
    for (C c : {C{26, 13}, C{512, 13}, C{512, 512}, C{4096, 4096}, C{8192, 8192}, C{1000, 1000}}) {
        std::vector<float> x(c.n); for (int i = 0; i < c.n; ++i) x[i] = std::sin(0.01 * i);
        std::printf("dctII n=%-5d -> %-5d outputs: %9.4f ms\n", c.n, c.k, best_ms([&] { auto r = speech::dsp::dctII(x, c.k, true); }, c.n >= 4096 && c.k > 24 ? 20 : 200));
    }
}
