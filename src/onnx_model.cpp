//
// Created by mohammad on 3/16/25.
//

#include "onnx_model.h"

#include <iostream>

ONNXModel::ONNXModel(const std::string& url, const std::filesystem::path& base_dir)
    : BaseModel(url, base_dir) {
    load_model();
}

void ONNXModel::load_model() {
    try {
        std::cout << "Loading ONNX model from: " << model_path << std::endl;

        // Initialize ONNX Runtime environment
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ONNXModel");

        // Configure session options
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // Load the model
        session = std::make_shared<Ort::Session>(env, model_path.string().c_str(), session_options);
        std::cout << "ONNX model loaded successfully." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error loading ONNX model: " << e.what() << std::endl;
        throw;
    }
}

