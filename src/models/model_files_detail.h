// Internal: the download-and-verify logic with the transport injected, so it can be tested without a
// network. Not installed.
#ifndef LIBSPEECH_MODEL_FILES_DETAIL_H
#define LIBSPEECH_MODEL_FILES_DETAIL_H

#include <chrono>
#include <filesystem>
#include <functional>
#include <string>

#include "libspeech/models/model_files.h"

namespace speech::models::detail {

/** Downloads `url` to the file `dest`; returns false on failure. May leave a partial file behind. */
using fetch_fn = std::function<bool(const std::string& url, const std::filesystem::path& dest)>;

/** Timing of the cross-process download lock (see download_lock in model_files.cpp). Tests shrink it. */
struct lock_timing {
    std::chrono::milliseconds heartbeat{1000};     ///< the process that downloads rewrites <lock>/alive this often
    std::chrono::milliseconds stale_after{15000};  ///< a lock whose heartbeat is older than this belongs to a dead process
    std::chrono::milliseconds poll{200};           ///< how often a waiting process looks at the lock
    std::chrono::milliseconds max_wait{3600000};   ///< a waiter gives up on a live lock after this long and downloads anyway
};

model_files fetch_model_with(const fetch_fn& fetch, const std::string& model, const fetch_options& options,
                             const lock_timing& timing = {});

}  // namespace speech::models::detail

#endif  // LIBSPEECH_MODEL_FILES_DETAIL_H
