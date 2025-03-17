//
// Created by mohammad on 3/16/25.
//

#include "onnx_model.h"
#include <stdexcept>

/**
 * Constructor for ONNXModel.
 * @param url The URL of the ONNX model file to download.
 * @param base_dir The base directory where the model will be stored.
 */

ONNXModel::ONNXModel(const std::string& url, const std::filesystem::path& base_dir)
    : BaseModel(url, base_dir) {}

/**
 * Initializes the ONNX Runtime session with the downloaded model.
 */
void ONNXModel::init_onnx_model() {
    if (!std::filesystem::exists(model_path)) {
        throw std::runtime_error("Model file does not exist: " + model_path.string());
    }
    session = std::make_shared<Ort::Session>(env, model_path.string().c_str(), session_options);
}

/**
 * Configures threading options for the ONNX Runtime session.
 * @param inter_threads Number of inter-op threads.
 * @param intra_threads Number of intra-op threads.
 */
void ONNXModel::init_engine_threads(int inter_threads, int intra_threads) {
    session_options.SetIntraOpNumThreads(intra_threads);
    session_options.SetInterOpNumThreads(inter_threads);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
}