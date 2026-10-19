//
// Streaming interface of the denoisers that can run on live audio.
//

#ifndef LIBSPEECH_DENOISER_STREAM_H
#define LIBSPEECH_DENOISER_STREAM_H

#include <cstddef>
#include <vector>

#include "libspeech/export.h"

namespace speech::models {

/**
 * One denoising session over a continuous stream of audio (a microphone, a socket, ...), created with
 * denoiser::create_stream(). Feed audio in chunks of any size; enhanced samples come back as soon as they are
 * final, delayed by at most latency_samples() (the model needs a little look-ahead).
 *
 *     auto stream = den->create_stream();
 *     for each chunk:  auto cleaned = stream->feed(chunk.data(), chunk.size());   // may be shorter than chunk
 *     auto tail = stream->finish();                                               // the delayed samples
 *
 * Over the whole session the output has exactly as many samples as the input, and it is the same audio that
 * denoiser::process() returns for the concatenated input, however the input was split into chunks.
 *
 * One stream = one conversation: it carries the model's recurrent state, so use a separate stream for each
 * unrelated signal. Streams of one denoiser are independent of each other and may run on different threads;
 * a single stream must not be used from several threads at once. A stream keeps the model it was created from
 * alive: closing the denoiser does not stop running streams (the model's memory is released when the last
 * stream is destroyed).
 */
class SPEECH_API denoiser_stream {
   public:
    virtual ~denoiser_stream();

    denoiser_stream(const denoiser_stream&) = delete;
    denoiser_stream& operator=(const denoiser_stream&) = delete;

    /**
     * @param data Mono samples normalized to [-1, 1] at the denoiser's sample rate; may be empty.
     * @return The samples that became final with this chunk (possibly none, e.g. while the first frame fills).
     * @throws std::logic_error after finish().
     * @throws std::invalid_argument if data is null while size > 0.
     */
    std::vector<float> feed(const float* data, std::size_t size);
    std::vector<float> feed(const std::vector<float>& chunk) { return feed(chunk.data(), chunk.size()); }

    /** Ends the stream and returns the remaining samples, so total output == total input. Idempotent. */
    std::vector<float> finish();

    /** Samples fed so far / samples returned so far. */
    std::size_t samples_in() const noexcept { return in_; }
    std::size_t samples_out() const noexcept { return out_; }

    /** Never more than this many samples are fed but not yet returned (e.g. 511 = 32 ms at 16 kHz for GTCRN). */
    std::size_t latency_samples() const noexcept { return latency_; }

    bool finished() const noexcept { return finished_; }

   protected:
    explicit denoiser_stream(std::size_t latency_samples) : latency_(latency_samples) {}

    /** Backend hooks: append the finalized samples to `out`. */
    virtual void feed_impl(const float* data, std::size_t size, std::vector<float>& out) = 0;
    virtual void finish_impl(std::vector<float>& out) = 0;

   private:
    std::size_t latency_;
    std::size_t in_ = 0;
    std::size_t out_ = 0;
    bool finished_ = false;
};

}  // namespace speech::models

#endif  // LIBSPEECH_DENOISER_STREAM_H
