// Based on https://github.com/leimao/ONNX-Runtime-Inference
// https://github.com/microsoft/onnxruntime/blob/v1.8.2/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/CXX_Api_Sample.cpp
// https://github.com/microsoft/onnxruntime/blob/v1.8.2/include/onnxruntime/core/session/onnxruntime_cxx_api.h

//#include <opencv2/dnn/dnn.hpp>
//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/imgproc.hpp>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <cxxopts.hpp>
//#include <nlohmann/json.hpp>
#include <json.hpp>

#include "onnxruntime.h"
#include "openvino.h"

using json = nlohmann::json;

std::unique_ptr<Runtime> nameToRuntime(const std::string &name, const std::string &modelPath) {
    Runtime *res = nullptr;

    if (name == "onnxruntime") {
        res = new ONNXRuntime(modelPath, 2);
    } else if (name == "openvino") {
        res = new OpenVINO(modelPath);
    } else {
        throw std::runtime_error("Unknown runtime name " + name);
    }

    return std::unique_ptr<Runtime>(res);
}

int main(int argc, char *argv[]) {
    cxxopts::Options options("thesis", "DL inference runtimes benchmark");

    options.add_options()
//        ("d,debug", "Enable debugging") // a bool parameter
        ("p,path", "Path to model", cxxopts::value<std::string>())
        ("i,iterations", "Number of iterations to run for",
         cxxopts::value<int32_t>()->default_value("25"))
        ("r,runtime", "Name of the runtime to use",
         cxxopts::value<std::string>()->default_value("onnxruntime"))
        ("v,verbose", "Verbose output", cxxopts::value<bool>()->default_value("false"))
        ("h,help", "Print usage");

    options.parse_positional("path");

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    if (!result.count("path")) {
        std::cerr << "Missing model path argument" << std::endl;
        exit(1);
    }

    bool verbose = result["verbose"].as<bool>();
    std::string modelPath = result["path"].as<std::string>();
    int nIterations = result["iterations"].as<int32_t>();
    std::string runtimeName = result["runtime"].as<std::string>();

//    std::unique_ptr<Runtime> rt{new ONNXRuntime(modelPath, 2)};
    std::unique_ptr<Runtime> rt = nameToRuntime(runtimeName, modelPath);

    // run once to skip first iteration
    rt->getLatency();

    std::vector<float> latencies;
    for (int i = 0; i < nIterations; i++) {
        latencies.push_back(rt->getLatency());
    }

    float mean = 0;
    for (float x: latencies) {
        mean += x;
    }
    mean /= nIterations;

    json j = {
        {"mean_latency_ms", mean},
        {"n_iterations",    nIterations},
        {"runtime",         runtimeName},
        {"model_path",      modelPath},
//        {"nothing", nullptr},
//        {"answer",  {
//                        {"everything", 42}
//                    }},
//        {"list",    {   1, 0, 2}},
//        {"object",  {
//                        {"currency",   "USD"},
//                           {"value", 42.99}
//                    }}
    };

    std::cout << j.dump(4) << std::endl;
}
