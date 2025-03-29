//
// Created by mohammad on 3/19/25.
//

#include "models/facebook_denoiser.h"


/**
 * Constructor for FacebookDenoiser.
 * @param url The URL of the ONNX model file to download.
 * @param base_dir The base directory where the model will be stored.
 */
FacebookDenoiser::FacebookDenoiser(const std::string& url, const int sample_rate)
    : ONNXModel(url, sample_rate){
    // Initialize threading settings (optional).
    init_engine_threads(1, 1);

    // Load and initialize the ONNX model.
    init_onnx_model();
}

/**
 * Processes an input audio tensor using the denoiser model.
 * @param input_audio A vector of floats representing the input audio (normalized between -1 and 1).
 * @param sample_rate The sample rate of the input audio (e.g., 16000 Hz).
 * @return A vector of floats representing the denoised audio.
 */
std::vector<float> FacebookDenoiser::process(const std::vector<float>& input_audio) {
    if (input_audio.empty()) {
        throw std::invalid_argument("Input audio data is empty.");
    }

    // Define the number of channels (e.g., 1 for mono audio).
    const int64_t batch_size = 1;  // Batch size
    const int64_t channels = 1;    // Number of channels (mono audio)
    const int64_t length = static_cast<int64_t>(input_audio.size());  // Length of the audio

    // Prepare input tensor shape as [batch_size, channels, length].
    std::vector<int64_t> input_shape = {batch_size, channels, length};

    // Create the input tensor with the new shape.
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, (float*) input_audio.data(), input_audio.size(), input_shape.data(), input_shape.size());

    // Prepare input and output names as C-style strings.
    const char* input_names[] = {input_name.c_str()};  // Convert std::string to const char*
    const char* output_names[] = {output_name.c_str()};

    // Run inference.
    auto output_tensors = session->Run(
        Ort::RunOptions{nullptr},
        input_names,                      // Array of input names
        &input_tensor,                    // Array of input tensors
        1,                                // Number of inputs
        output_names,                     // Array of output names
        1                                 // Number of outputs
    );

    // Extract the output tensor.
    const float* output_data = output_tensors[0].GetTensorMutableData<float>();
    size_t output_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
    // Convert the output to a vector.
    std::vector<float> denoised_audio(output_data, output_data + output_size);
    return denoised_audio;
}


FacebookDenoiser::~FacebookDenoiser() {

}
