//
// Created by Vaclav Volhejn on 14.06.2022.
//

#include <cassert>
#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <cxxopts.hpp>

#include "onnxruntime.h"

template<typename T>
T vectorProduct(const std::vector<T> &v) {
    return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

/**
 * @brief Operator overloading for printing vectors
 * @tparam T
 * @param os
 * @param v
 * @return std::ostream&
 */
template<typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &v) {
    os << "[";
    for (int i = 0; i < v.size(); ++i) {
        os << v[i];
        if (i != v.size() - 1) {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

/**
 * @brief Print ONNX tensor data type
 * https://github.com/microsoft/onnxruntime/blob/rel-1.6.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L93
 * @param os
 * @param type
 * @return std::ostream&
 */
std::ostream &operator<<(std::ostream &os, const ONNXTensorElementDataType &type) {
    switch (type) {
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
            os << "undefined";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            os << "float";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            os << "uint8_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            os << "int8_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
            os << "uint16_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            os << "int16_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            os << "int32_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            os << "int64_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
            os << "std::string";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            os << "bool";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            os << "float16";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            os << "double";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
            os << "uint32_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
            os << "uint64_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
            os << "float real + float imaginary";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
            os << "double real + float imaginary";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
            os << "bfloat16";
            break;
        default:
            break;
    }

    return os;
}

ONNXRuntime::ONNXRuntime(std::string modelPath, int32_t nThreads) : session(nullptr) {
    bool verbose = false;

    std::string instance_name = "Inference";

    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, instance_name.c_str());
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(nThreads);

    // Sets graph optimization level
    // Available levels are
    // ORT_DISABLE_ALL -> To disable all optimizations
    // ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node
    // removals) ORT_ENABLE_EXTENDED -> To enable extended optimizations
    // (Includes level 1 + more complex optimizations like node fusions)
    // ORT_ENABLE_ALL -> To Enable All possible optimizations
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    session = Ort::Session(env, modelPath.c_str(), sessionOptions);

    Ort::AllocatorWithDefaultOptions allocator;

    size_t numInputNodes = session.GetInputCount();
    size_t numOutputNodes = session.GetOutputCount();

    const char *inputName = session.GetInputName(0, allocator);

    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();

    std::vector<int64_t> inputDims = inputTensorInfo.GetShape();

    const char *outputName = session.GetOutputName(0, allocator);

    Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();

    std::vector<int64_t> outputDims = outputTensorInfo.GetShape();

    if (verbose) {
        std::cout << "Number of Input Nodes: " << numInputNodes << std::endl;
        std::cout << "Number of Output Nodes: " << numOutputNodes << std::endl;
        std::cout << "Input Name: " << inputName << std::endl;
        std::cout << "Input Type: " << inputType << std::endl;
        std::cout << "Input Dimensions: " << inputDims << std::endl;
        std::cout << "Output Name: " << outputName << std::endl;
        std::cout << "Output Type: " << outputType << std::endl;
        std::cout << "Output Dimensions: " << outputDims << std::endl;
    }

    size_t inputTensorSize = vectorProduct(inputDims);
    std::vector<float> inputTensorValues(inputTensorSize, 0.5);

    size_t outputTensorSize = vectorProduct(outputDims);
    assert(("Output tensor size should equal to the label set size.",
        labels.size() == outputTensorSize));
    std::vector<float> outputTensorValues(outputTensorSize);

    inputNames.push_back(inputName);
    outputNames.push_back(outputName);

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator,
                                                            OrtMemType::OrtMemTypeDefault);
    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, inputTensorValues.data(), inputTensorSize, inputDims.data(),
        inputDims.size()));
    outputTensors.push_back(
        Ort::Value::CreateTensor<float>(memoryInfo, outputTensorValues.data(),
                                        outputTensorSize, outputDims.data(),
                                        outputDims.size()));

    session.Run(Ort::RunOptions{nullptr}, inputNames.data(), inputTensors.data(), 1,
                outputNames.data(), outputTensors.data(), 1);

}

float ONNXRuntime::getLatency() {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    session.Run(Ort::RunOptions{nullptr}, inputNames.data(), inputTensors.data(), 1,
                outputNames.data(), outputTensors.data(), 1);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    float latency_ms =
        std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() /
        1000.0;
    return latency_ms;
}