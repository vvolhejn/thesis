//
// Created by Vaclav Volhejn on 14.06.2022.
//

#include <openvino/openvino.hpp>

#include "openvino.h"

OpenVINO::OpenVINO(std::string modelPath) {
    ov::Core core;
    compiledModel = core.compile_model(
        modelPath, "CPU",
        ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)
    );
    inferRequest = compiledModel.create_infer_request();

    auto input_port = compiledModel.input();
    totalInputSize = 1;
    for (auto axis: input_port.get_shape()) {
        totalInputSize *= axis;
    }
}

float OpenVINO::getLatency() {
    auto input_port = compiledModel.input();

    // Create tensor from external memory
    auto memory_ptr = new float[totalInputSize];

    ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), memory_ptr);
    // Set input tensor for model with one input
    inferRequest.set_input_tensor(input_tensor);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();


    inferRequest.infer();

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    float latency_ms =
        std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() /
        1000.0;

//    get output tensor by tensor name
//    auto output = inferRequest.get_tensor("tensor_name");
//    const float *output_buffer = output.data<const float>();
    return latency_ms;
}