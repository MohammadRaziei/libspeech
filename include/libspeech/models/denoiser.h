//
// Common interface for all denoiser backends (Facebook, SpeechBrain, ...).
//

#ifndef LIBSPEECH_DENOISER_H
#define LIBSPEECH_DENOISER_H

#include <cstddef>
#include <memory>
#include <shared_mutex>
#include <string>
#include <vector>

#include "libspeech/export.h"

namespace speech::models {

/**
 * Denoiser: shared interface every denoising backend implements.
 * Lets callers swap backends without caring which one is loaded.
 *
 * Design notes:
 *  - Backends derive from Denoiser only (single inheritance) and keep their ONNX Runtime state
 *    behind a private implementation pointer, so a `Denoiser*` always points at the start of the
 *    object. (Language bindings and `delete` through the base rely on that.)
 *  - process()/close() are non-virtual (NVI): argument checks, the "already closed" check and
 *    thread-safety live here once; backends only implement processImpl()/closeImpl().
 *  - Thread-safety: process() may be called concurrently from several threads. close() waits for
 *    running process() calls to finish, then releases the model; it is safe to call repeatedly.
 */
class SPEECH_API Denoiser {
   public:
    virtual ~Denoiser();

    Denoiser(const Denoiser&) = delete;
    Denoiser& operator=(const Denoiser&) = delete;

    /**
     * @param input_audio Mono audio samples normalized to [-1, 1].
     * @return Denoised audio, same length as input.
     * @throws std::invalid_argument if the input is empty.
     * @throws std::runtime_error if the denoiser was closed.
     */
    std::vector<float> process(const std::vector<float>& input_audio);

    /** Same as above for a raw buffer (no copy of the input). */
    std::vector<float> process(const float* data, std::size_t size);

    /**
     * Releases the model and the memory it holds right away instead of at destruction.
     * Idempotent. Afterwards process() throws std::runtime_error.
     */
    void close() noexcept;

    /** True once close() has been called. */
    bool closed() const noexcept;

    /**
     * Factory: picks a backend by name ("facebook" | "speechbrain").
     * @param num_threads ONNX Runtime intra-op threads: 1 (default) = single-threaded,
     *                    0 = one per hardware thread. Must not be negative.
     * @throws std::invalid_argument for an unknown backend or a negative num_threads.
     */
    static std::unique_ptr<Denoiser> Create(const std::string& backend,
                                             const std::string& url,
                                             int sample_rate = 16000,
                                             int num_threads = 1);

    /** Sample rate (Hz) the model expects its input at. */
    const int sample_rate;

   protected:
    explicit Denoiser(int sample_rate);

    /** Backend hook: runs the model. Input is already validated; the denoiser is open. */
    virtual std::vector<float> processImpl(const float* data, std::size_t size) = 0;

    /** Backend hook: frees the model. Called at most once. */
    virtual void closeImpl() noexcept = 0;

   private:
    mutable std::shared_mutex mutex_;
    bool closed_ = false;
};

}  // namespace speech::models

#endif  // LIBSPEECH_DENOISER_H
