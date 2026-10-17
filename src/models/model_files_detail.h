// Internal: the download-and-verify logic with the transport injected, so it can be tested without a
// network. Not installed.
#ifndef LIBSPEECH_MODEL_FILES_DETAIL_H
#define LIBSPEECH_MODEL_FILES_DETAIL_H

#include <filesystem>
#include <functional>
#include <string>

#include "libspeech/models/model_files.h"

namespace speech::models::detail {

/** What the transport is asked to fetch. */
struct fetch_request {
    std::string url;
    std::filesystem::path dest;  ///< the final path: the transport writes it atomically (temporary file + rename)
    std::string sha256;          ///< expected digest of the file; empty = not verified. A mismatch deletes `dest`.
    bool force = false;          ///< download again even if `dest` already holds the complete file
    bool show_progress = false;
};

struct fetch_result {
    bool ok = false;
    std::string error;  ///< why it failed (HTTP status, checksum mismatch, ...); empty if ok
};

/**
 * Downloads one file. The real transport is httpp::download(...).resume(): only one process at a time
 * downloads a given `dest` (the others wait and then find it complete), the file appears under its name only
 * when it is complete, and a file that fails its checksum is deleted.
 */
using fetch_fn = std::function<fetch_result(const fetch_request&)>;

model_files fetch_model_with(const fetch_fn& fetch, const std::string& model, const fetch_options& options);

}  // namespace speech::models::detail

#endif  // LIBSPEECH_MODEL_FILES_DETAIL_H
