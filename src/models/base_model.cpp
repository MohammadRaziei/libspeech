#include "models/base_model.h"
#include <stdexcept>
#include "utils/utils.h"
#include "aixlog.hpp"

BaseModel::BaseModel(const std::string& url, const std::filesystem::path& base_dir)
    : url(url), model_path(base_dir) {
    // Ensure the base directory exists
    if (!std::filesystem::exists(base_dir)) {
        std::filesystem::create_directories(base_dir);
    }

    // Check if the URL is a valid internet address or a local file name
    if (url.find("http://") != 0 && url.find("https://") != 0) {
        // Convert to default URL
        this->url = "https://github.com/MohammadRaziei/libspeech/releases/download/models/" + url;
    }

    // Download the model
    download_model();
}

void BaseModel::download_model() {
    try {
        LOG(DEBUG) << TAG("speech::models::BaseModel") << "Downloading model from: " << url << std::endl;

        // Download directly into the target directory
        std::filesystem::path fileName = speech::utils::downloadFile(url, model_path, false, false);

        if (fileName.empty()) {
            LOG(DEBUG) << TAG("speech::models::BaseModel") << "Failed to download the model." << std::endl;
            throw std::runtime_error("Failed to download the model.");
        }

        // Set the model path
        model_path = model_path / fileName.filename();
        LOG(INFO) << TAG("speech::models::BaseModel") << "Model saved to: " << model_path << std::endl;
    } catch (const std::exception& e) {
        LOG(ERROR) << TAG("speech::models::BaseModel") << "Error downloading model: " << e.what() << std::endl;
        throw;
    }
}