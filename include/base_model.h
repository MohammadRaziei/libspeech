//
// Created by mohammad on 3/16/25.
//

#ifndef LIBSPEECH_BASE_MODEL_H
#define LIBSPEECH_BASE_MODEL_H


#include <string>
#include <filesystem>
#include <iostream>

namespace speech {
namespace utils {
std::filesystem::path downloadFile(const std::string& url, const std::filesystem::path& destination, bool overwrite, bool verbose);
}
}

class BaseModel {
   protected:
    std::string url;
    std::filesystem::path model_path;

   public:
    BaseModel(const std::string& url, const std::filesystem::path& base_dir = std::filesystem::path(getenv("HOME")) / ".libspeech");
    virtual ~BaseModel() = default;

   protected:
    void download_model();
};

#endif  // LIBSPEECH_BASE_MODEL_H
