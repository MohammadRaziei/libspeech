// Profile of the model pipelines (SileroVadModel, FacebookDenoiser, SpeechBrainDenoiser): the libspeech
// wrapper against a bare ONNX Runtime loop that makes the identical inferences, to show how much of the
// time (and how many heap allocations) is the wrapper and how much is ONNX Runtime itself.
// Needs the three model files in ~/.libspeech (see README.md in this directory).
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <random>
#include <vector>
#include "libspeech/models/facebook_denoiser.h"
#include "libspeech/models/silero_vad.h"
#include "libspeech/models/speechbrain_denoiser.h"
#include "onnxruntime_cxx_api.h"
static std::atomic<unsigned long> g_allocs{0};
void* operator new(std::size_t n) { g_allocs++; if (void* p = std::malloc(n)) return p; throw std::bad_alloc(); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
using clk = std::chrono::high_resolution_clock;
static double ms_since(clk::time_point a) { return std::chrono::duration<double, std::milli>(clk::now() - a).count(); }
static std::vector<float> speechLike(int n, int sr) {
    std::mt19937 g(1); std::normal_distribution<float> nd(0.f, 0.01f); std::vector<float> s(n);
    for (int i = 0; i < n; ++i) { double t = (double)i / sr; double env = 0.5 * (1 + std::sin(2 * M_PI * 1.5 * t)); s[i] = (float)(env * (0.3 * std::sin(2 * M_PI * 180 * t) + 0.15 * std::sin(2 * M_PI * 1500 * t))) + nd(g); }
    return s;
}
int main() {
    const int sr = 16000;
    // ---------------- Silero VAD: wrapper vs a bare ONNX Runtime loop doing the identical inferences
    const int secs = 60; auto wav = speechLike(secs * sr, sr);
    auto t0 = clk::now();
    speech::models::SileroVadModel vad("silero-vad.onnx", sr);
    std::printf("SileroVad: construct (session create)            %8.1f ms\n", ms_since(t0));
    vad.processOnVector(wav);   // warm-up
    double best = 1e18; unsigned long allocs = 0;
    for (int r = 0; r < 5; ++r) { g_allocs = 0; auto a = clk::now(); vad.processOnVector(wav); double m = ms_since(a); if (m < best) { best = m; allocs = g_allocs; } }
    const int chunks = (secs * sr) / 512;
    std::printf("SileroVad: processOnVector %d s (%d chunks)       %8.1f ms  = %.1f us/chunk, %.2f allocs/chunk, speech segments: %zu\n", secs, chunks, best, best * 1000 / chunks, (double)allocs / chunks, vad.get_speech_timestamps().size());

    Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "p"); Ort::SessionOptions so; so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    std::string path = std::string(std::getenv("HOME")) + "/.libspeech/silero-vad.onnx";
    Ort::Session sess(env, path.c_str(), so); Ort::AllocatorWithDefaultOptions al;
    std::vector<std::string> in_names, out_names; std::vector<const char*> inn, outn;
    for (size_t i = 0; i < sess.GetInputCount(); ++i) in_names.push_back(sess.GetInputNameAllocated(i, al).get());
    for (size_t i = 0; i < sess.GetOutputCount(); ++i) out_names.push_back(sess.GetOutputNameAllocated(i, al).get());
    for (auto& s : in_names) inn.push_back(s.c_str()); for (auto& s : out_names) outn.push_back(s.c_str());
    auto mi = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPU);
    std::vector<float> inbuf(576, 0.f), state(2 * 128, 0.f); std::vector<int64_t> srv{sr};
    int64_t d_in[2] = {1, 576}, d_st[3] = {2, 1, 128}, d_sr[1] = {1};
    auto raw = [&] {
        for (int j = 0; j + 512 <= secs * sr; j += 512) {
            std::memcpy(inbuf.data() + 64, wav.data() + j, 512 * sizeof(float));   // context left at zeros: same shapes/work
            Ort::Value ins[3] = {Ort::Value::CreateTensor<float>(mi, inbuf.data(), inbuf.size(), d_in, 2), Ort::Value::CreateTensor<float>(mi, state.data(), state.size(), d_st, 3), Ort::Value::CreateTensor<int64_t>(mi, srv.data(), 1, d_sr, 1)};
            auto out = sess.Run(Ort::RunOptions{nullptr}, inn.data(), ins, 3, outn.data(), outn.size());
            std::memcpy(state.data(), out[1].GetTensorMutableData<float>(), state.size() * sizeof(float));
        }
    };
    raw(); double rb = 1e18; unsigned long ra = 0; for (int r = 0; r < 5; ++r) { g_allocs = 0; auto a = clk::now(); raw(); double m = ms_since(a); if (m < rb) { rb = m; ra = g_allocs; } }
    std::printf("bare ONNX Runtime loop, same inferences             %8.1f ms  = %.1f us/chunk, %.2f allocs/chunk\n", rb, rb * 1000 / chunks, (double)ra / chunks);
    std::printf("  => libspeech wrapper overhead on top of ONNX Runtime: %.1f ms (%.1f%%)\n\n", best - rb, 100.0 * (best - rb) / best);

    // ---------------- Denoisers
    const int dsecs = 3; auto noisy = speechLike(dsecs * sr, sr);
    auto bench_den = [&](const char* label, auto& den, const char* file) {
        den.process(noisy); double b = 1e18; for (int r = 0; r < 3; ++r) { auto a = clk::now(); auto o = den.process(noisy); b = std::min(b, ms_since(a)); (void)o; }
        Ort::SessionOptions so1; so1.SetIntraOpNumThreads(1); so1.SetInterOpNumThreads(1); so1.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        std::string p2 = std::string(std::getenv("HOME")) + "/.libspeech/" + file; Ort::Session s2(env, p2.c_str(), so1);
        std::string iname = s2.GetInputNameAllocated(0, al).get(), oname = s2.GetOutputNameAllocated(0, al).get(); const char* in1[] = {iname.c_str()}; const char* on1[] = {oname.c_str()};
        auto ti = s2.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape(); std::vector<int64_t> shape = ti.size() == 3 ? std::vector<int64_t>{1, 1, (int64_t)noisy.size()} : std::vector<int64_t>{1, (int64_t)noisy.size()};
        auto run = [&] { auto t = Ort::Value::CreateTensor<float>(mi, noisy.data(), noisy.size(), shape.data(), shape.size()); auto o = s2.Run(Ort::RunOptions{nullptr}, in1, &t, 1, on1, 1); (void)o; };
        run(); double rbb = 1e18; for (int r = 0; r < 3; ++r) { auto a = clk::now(); run(); rbb = std::min(rbb, ms_since(a)); }
        std::printf("%-22s process(%d s): %9.1f ms (real-time factor %.2fx) | bare ORT, same threads: %9.1f ms | wrapper overhead %.2f ms\n", label, dsecs, b, b / (dsecs * 1000.0), rbb, b - rbb);
    };
    { auto t = clk::now(); speech::models::FacebookDenoiser fd("facebook-denoiser-dns64.onnx", sr); std::printf("Facebook denoiser: construct %.0f ms\n", ms_since(t)); bench_den("Facebook DNS64", fd, "facebook-denoiser-dns64.onnx"); }
    { auto t = clk::now(); speech::models::SpeechBrainDenoiser sd("speechbrain-sepformer-wham16k-enhancement.onnx", sr); std::printf("SpeechBrain denoiser: construct %.0f ms\n", ms_since(t)); bench_den("SpeechBrain SepFormer", sd, "speechbrain-sepformer-wham16k-enhancement.onnx"); }
}
