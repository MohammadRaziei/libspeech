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


namespace speech::models {

class SPEECH_API base_model {
   protected:
    std::string url;
    std::filesystem::path model_path;

   public:
    base_model(const std::string& url, const std::filesystem::path& base_dir = speech::utils::get_default_model_cache_dir());
    virtual ~base_model() = default;

   protected:
    void download_model();
};

}  // namespace speech::models

#endif  // LIBSPEECH_BASE_MODEL_H
