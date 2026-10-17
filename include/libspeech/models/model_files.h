//
// Fetching and verifying model files: <model>.onnx, <model>.onnx.sha256 and <model>.onnx.readme.
//

#ifndef LIBSPEECH_MODEL_FILES_H
#define LIBSPEECH_MODEL_FILES_H

#include <filesystem>
#include <string>

#include "libspeech/export.h"

namespace speech::models {

/** Where bare model names are downloaded from (the `models` release of the libspeech repository). */
inline constexpr const char* release_url = "https://github.com/MohammadRaziei/libspeech/releases/download/models/";

/** The three files that make up a model in the cache, plus its verified checksum. */
struct model_files {
    std::filesystem::path model;     ///< the .onnx file
    std::filesystem::path checksum;  ///< <model>.sha256 (empty if the server had none, see fetch_model)
    std::filesystem::path readme;    ///< <model>.readme (empty if the server had none, see fetch_model)
    std::string sha256;              ///< the model's SHA-256 (64 lowercase hex chars); empty if it was not verified
};

struct fetch_options {
    /** Cache directory; empty means speech::utils::get_default_model_cache_dir() (~/.libspeech). */
    std::filesystem::path cache_dir;
    /** true: download everything again and replace the cached files (a failed refresh leaves the cache untouched). */
    bool ignore_cache = false;
    /** true: no progress bar for the model download. */
    bool quiet = false;
};

/**
 * Makes sure a model is in the cache, verified, and returns where its files are.
 *
 * `model` is a bare file name ("gtcrn_simple.onnx"), looked up in the `models` release, or a full
 * http(s) URL. Each model comes with `<url>.sha256` and `<url>.readme`; all three are downloaded.
 * The order is: the checksum file first (it must be exactly one line, `<64 hex digits>` optionally
 * followed by the file name, as `sha256sum` writes it; anything else, e.g. an HTML error page, is
 * rejected), then the model, which is hashed and compared, then the readme. Nothing is installed in the
 * cache until all of it has been downloaded and the hash matched: a file that fails the check is deleted.
 *
 * A cached model is hashed on every call and re-downloaded if it does not match its checksum file.
 * A cached model without a checksum file (cached by an older version) gets one downloaded; if that is
 * not possible (e.g. offline) the cached model is used and a warning is logged.
 *
 * For a bare name the checksum and the readme are required. For an explicit URL they are optional: if the
 * server does not have them the model is used unverified and a warning is logged (a checksum file that
 * exists but is malformed is still an error).
 *
 * Several processes or threads may call this for the same model at once: one downloads (it holds the
 * directory `<model>.onnx.lock` and refreshes `<model>.onnx.lock/alive` every second), the others wait and
 * then use what it put in the cache. A lock whose `alive` file has not been touched for 15 s is taken over.
 *
 * @throws std::runtime_error on download failure, malformed checksum file or hash mismatch.
 */
SPEECH_API model_files fetch_model(const std::string& model, const fetch_options& options = {});

/** The readme text of a model (fetched like fetch_model; for a model from the release it describes
 *  what the model is, who made it, its license and its provenance). */
SPEECH_API std::string model_readme(const std::string& model, const fetch_options& options = {});

/**
 * Parses the text of a checksum file: one line, `<64 hex digits>` optionally followed by whitespace,
 * an optional `*` and the file name (which must equal `file_name`). Returns the digest in lowercase.
 * @throws std::runtime_error if it is anything else.
 */
SPEECH_API std::string parse_checksum(const std::string& text, const std::string& file_name);

}  // namespace speech::models

#endif  // LIBSPEECH_MODEL_FILES_H
