//
// Created by mohammad on 3/16/25.
//

#include "models/onnx_model.h"
#include <stdexcept>

/**
 * Constructor for ONNXModel.
 * @param url The URL of the ONNX model file to download.
 * @param base_dir The base directory where the model will be stored.
 */

ONNXModel::ONNXModel(const std::string& url, const int sample_rate, const std::filesystem::path& base_dir)
    : BaseModel(url, base_dir), sample_rate(sample_rate),
      memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPU)) {}

/**
 * Initializes the ONNX Runtime session with the downloaded model.
 */
//void ONNXModel::init_onnx_model() {
//    if (!std::filesystem::exists(model_path)) {
//        throw std::runtime_error("Model file does not exist: " + model_path.string());
//    }
//    session = std::make_shared<Ort::Session>(env, model_path.string().c_str(), session_options);
//}

void ONNXModel::init_onnx_model() {
    if (!std::filesystem::exists(model_path)) {
        throw std::runtime_error("Model file does not exist: " + model_path.string());
    }
    session = std::make_shared<Ort::Session>(env, model_path.string().c_str(), session_options);

    // Validate input/output names.
    const Ort::AllocatedStringPtr input_names = session->GetInputNameAllocated(0, allocator);
    const Ort::AllocatedStringPtr output_names = session->GetOutputNameAllocated(0, allocator);
    input_name = input_names.get();
    output_name = output_names.get();
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

ONNXModel::~ONNXModel() {

}
