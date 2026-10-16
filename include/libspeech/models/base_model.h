//
// Created by mohammad on 3/16/25.
//

#ifndef LIBSPEECH_BASE_MODEL_H
#define LIBSPEECH_BASE_MODEL_H


#include <string>
#include <filesystem>
#include <iostream>
#include "libspeech/utils/utils.h"
#include "libspeech/export.h"
#include "libspeech/models/model_files.h"


namespace speech::models {

class SPEECH_API base_model {
   protected:
    std::string url;
    std::filesystem::path model_path;
    model_files files_;

   public:
    /**
     * Makes sure the model is in `base_dir` (downloading `<url>`, `<url>.sha256` and `<url>.readme` and
     * verifying the hash, see fetch_model()). `url` is a bare file name (looked up in libspeech's `models`
     * release) or a full http(s) URL.
     * @param ignore_cache download everything again even if it is cached.
     */
    base_model(const std::string& url, const std::filesystem::path& base_dir = speech::utils::get_default_model_cache_dir(),
               bool ignore_cache = false);
    virtual ~base_model() = default;

    /** Where the verified model file is. */
    const std::filesystem::path& model_file() const noexcept { return model_path; }
    /** The model's SHA-256 (64 lowercase hex chars); empty if it was not verified (see fetch_model). */
    const std::string& sha256() const noexcept { return files_.sha256; }
    /** The model's readme (what it is, who made it, license, provenance). */
    std::string readme() const;
    /** All files of the model (model, checksum, readme). */
    const model_files& files() const noexcept { return files_; }

   protected:
    void download_model(bool ignore_cache);
};

}  // namespace speech::models

#endif  // LIBSPEECH_BASE_MODEL_H
