//
// Created by Vaclav Volhejn on 14.06.2022.
//

#ifndef INFERENCE_RUNTIMES_OPENVINO_H
#define INFERENCE_RUNTIMES_OPENVINO_H

#include <openvino/openvino.hpp>

#include "runtime.h"

class OpenVINO : public Runtime {
public:
    OpenVINO(std::string modelPath);

    float getLatency() override;
private:
    ov::CompiledModel compiledModel;
    ov::InferRequest inferRequest;
    int32_t totalInputSize;
};

#endif //INFERENCE_RUNTIMES_OPENVINO_H
