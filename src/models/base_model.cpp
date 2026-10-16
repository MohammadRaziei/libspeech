#include "libspeech/models/base_model.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include "libspeech/utils/utils.h"
#include "aixlog.hpp"
#include "libspeech/detail/log.h"

namespace speech::models {

base_model::base_model(const std::string& url, const std::filesystem::path& base_dir, bool ignore_cache)
    : url(url), model_path(base_dir) {
    // A bare file name means libspeech's `models` release; fetch_model() resolves it. Keep the full URL
    // here for the log messages and subclasses.
    if (url.find("http://") != 0 && url.find("https://") != 0) {
        this->url = release_url + url;
    }
    download_model(ignore_cache);
}

void base_model::download_model(bool ignore_cache) {
    try {
        SPEECH_LOG(DEBUG) << TAG("speech::models::BaseModel") << "Fetching model: " << url << std::endl;

        fetch_options options;
        options.cache_dir = model_path;  // at this point model_path is still the base directory
        options.ignore_cache = ignore_cache;
        files_ = fetch_model(url, options);

        model_path = files_.model;
        SPEECH_LOG(INFO) << TAG("speech::models::BaseModel") << "Model saved to: " << model_path << std::endl;
    } catch (const std::exception& e) {
        SPEECH_LOG(ERROR) << TAG("speech::models::BaseModel") << "Error downloading model: " << e.what() << std::endl;
        throw;
    }
}

std::string base_model::readme() const {
    if (files_.readme.empty()) {
        throw std::runtime_error("No readme is available for model " + model_path.string());
    }
    std::ifstream in(files_.readme, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Cannot read " + files_.readme.string());
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

}  // namespace speech::models
