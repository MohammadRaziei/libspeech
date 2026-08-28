//
// Created by mohammad on 3/16/25.
//

#ifndef LIBSPEECH_ONNX_MODEL_H
#define LIBSPEECH_ONNX_MODEL_H

#include <memory>
#include "base_model.h"
#include "onnxruntime_cxx_api.h"

/**
 * ONNXModel serves as a base class for models that use ONNX Runtime.
 * It provides functionality for initializing and managing ONNX sessions.
 */
class ONNXModel : public BaseModel {
   protected:
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::shared_ptr<Ort::Session> session;
    Ort::MemoryInfo memory_info;
    Ort::AllocatorWithDefaultOptions allocator;

    std::string input_name, output_name;

   public:
    /**
     * Constructor for ONNXModel.
     * @param url The URL of the ONNX model file to download.
     * @param base_dir The base directory where the model will be stored. Defaults to ~/.libspeech.
     */
    ONNXModel(const std::string& url, const int sample_rate, const std::filesystem::path& base_dir = std::filesystem::path(getenv("HOME")) / ".libspeech");

    /**
     * Destructor for ONNXModel.
     */
    virtual ~ONNXModel();

    /**
     * Initializes the ONNX Runtime session with the downloaded model.
     */
    virtual void init_onnx_model();

    /**
     * Configures threading options for the ONNX Runtime session.
     * @param inter_threads Number of inter-op threads.
     * @param intra_threads Number of intra-op threads.
     */
    void init_engine_threads(int inter_threads, int intra_threads);

public:
    const int sample_rate;
};

#endif  // LIBSPEECH_ONNX_MODEL_H
