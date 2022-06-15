//
// Created by Vaclav Volhejn on 14.06.2022.
//

#ifndef INFERENCE_RUNTIMES_ONNXRUNTIME_H
#define INFERENCE_RUNTIMES_ONNXRUNTIME_H

#include <onnxruntime_cxx_api.h>

#include "runtime.h"

class ONNXRuntime : public Runtime {
public:
    ONNXRuntime(std::string modelPath, int32_t nThreads = 1);

    float getLatency() override;

private:
    Ort::Session session;
    std::vector<const char *> inputNames;
    std::vector<const char *> outputNames;
    std::vector<Ort::Value> inputTensors;
    std::vector<Ort::Value> outputTensors;
};

#endif //INFERENCE_RUNTIMES_ONNXRUNTIME_H
