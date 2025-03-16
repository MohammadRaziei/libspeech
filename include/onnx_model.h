//
// Created by mohammad on 3/16/25.
//

#ifndef LIBSPEECH_ONNX_MODEL_H
#define LIBSPEECH_ONNX_MODEL_H

#include <memory>
#include "onnxruntime_cxx_api.h"

#include "base_model.h"

class ONNXModel : public BaseModel {
   protected:
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::shared_ptr<Ort::Session> session;

   public:
    ONNXModel(const std::string& url, const std::filesystem::path& base_dir = std::filesystem::path(getenv("HOME")) / ".libspeech");
    virtual ~ONNXModel() = default;

   protected:
    void load_model();
};

#endif  // LIBSPEECH_ONNX_MODEL_H
