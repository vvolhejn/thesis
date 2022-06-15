# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.13)

# add_executable(model-explorer model-explorer.cpp)
# target_link_libraries(model-explorer PRIVATE onnxruntime)

# add_executable(batch-model-explorer batch-model-explorer.cpp)
# target_link_libraries(batch-model-explorer PRIVATE onnxruntime)

add_executable(inference inference.cpp onnxruntime.cpp onnxruntime.h openvino.cpp openvino.h runtime.cpp runtime.h)
target_include_directories(inference PRIVATE ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
# target_link_libraries(inference PRIVATE ${ONNX_RUNTIME_LIB} ${OpenCV_LIBRARIES})
target_link_libraries(inference PRIVATE onnxruntime ${OpenCV_LIBRARIES})
target_link_libraries(inference PRIVATE openvino::runtime)

target_include_directories(inference PUBLIC cxxopts)
target_link_libraries(inference PUBLIC cxxopts)

#target_include_directories(inference PRIVATE nlohmann_json)
#target_link_libraries(inference PRIVATE nlohmann_json)