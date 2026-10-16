// Internal: the download-and-verify logic with the transport injected, so it can be tested without a
// network. Not installed.
#ifndef LIBSPEECH_MODEL_FILES_DETAIL_H
#define LIBSPEECH_MODEL_FILES_DETAIL_H

#include <filesystem>
#include <functional>
#include <string>

#include "libspeech/models/model_files.h"

namespace speech::models::detail {

/** Downloads `url` to the file `dest`; returns false on failure. May leave a partial file behind. */
using fetch_fn = std::function<bool(const std::string& url, const std::filesystem::path& dest)>;

model_files fetch_model_with(const fetch_fn& fetch, const std::string& model, const fetch_options& options);

}  // namespace speech::models::detail

#endif  // LIBSPEECH_MODEL_FILES_DETAIL_H
