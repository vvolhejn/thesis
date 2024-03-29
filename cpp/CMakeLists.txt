# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.13)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE RelWithDebInfo)

# find_package(OpenCV REQUIRED)
# find_path(ONNX_RUNTIME_SESSION_INCLUDE_DIRS onnxruntime_cxx_api.h HINTS /usr/local/include/onnxruntime/core/session/)

# find_library(ONNX_RUNTIME_LIB onnxruntime HINTS /usr/local/lib)
add_subdirectory(extern/cxxopts)

include_directories("extern/nlohmann")

#set(JSON_BuildTests OFF CACHE INTERNAL "")
#add_subdirectory(extern/json)

# Project
project(inference_runtimes C CXX)
if (WIN32)
 string(APPEND CMAKE_CXX_FLAGS " /W4")
else()
 string(APPEND CMAKE_CXX_FLAGS " -O3 -Wall -Wextra")
 string(APPEND CMAKE_C_FLAGS " -O3 -Wall -Wextra")
endif()

#onnxruntime providers
option(onnxruntime_USE_CUDA "Build with CUDA support" OFF)
option(onnxruntime_USE_OPENVINO "Build with OpenVINO support" OFF)
option(onnxruntime_USE_NNAPI_BUILTIN "Build with builtin NNAPI lib for Android NNAPI support" OFF)
option(onnxruntime_USE_DNNL "Build with DNNL support" OFF)
option(onnxruntime_USE_NUPHAR "Build with Nuphar" OFF)
option(onnxruntime_USE_TENSORRT "Build with TensorRT support" OFF)
option(LIBPNG_ROOTDIR "libpng root dir")

option(ONNXRUNTIME_ROOTDIR "onnxruntime root dir")

if(NOT ONNXRUNTIME_ROOTDIR)
  if(WIN32)
    set(ONNXRUNTIME_ROOTDIR "C:/Program Files (x86)/onnxruntime")
  else()
    include_directories("/usr/local/include/onnxruntime")
  endif()
endif()

#TODO: we should only need one of them.
include_directories("${ONNXRUNTIME_ROOTDIR}/include" "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session")
link_directories("${ONNXRUNTIME_ROOTDIR}/lib")

#if JPEG lib is available, we'll use it for image decoding, otherwise we'll use WIC
find_package(JPEG)
if(LIBPNG_ROOTDIR)
  set(PNG_FOUND true)
  if(WIN32)
    set(PNG_LIBRARIES debug libpng16_d optimized libpng16)
  else()
    set(PNG_LIBRARIES png16)
  endif()
  set(PNG_INCLUDE_DIRS "${LIBPNG_ROOTDIR}/include")
  set(PNG_LIBDIR "${LIBPNG_ROOTDIR}/lib")
else()
  find_package(PNG)
endif()

if(onnxruntime_USE_CUDA)
  add_definitions(-DUSE_CUDA)
endif()
if(onnxruntime_USE_OPENVINO)
  add_definitions(-DUSE_OPENVINO)
endif()
if(onnxruntime_USE_NNAPI_BUILTIN)
  add_definitions(-DUSE_NNAPI)
endif()
if(onnxruntime_USE_DNNL)
  add_definitions(-DUSE_DNNL)
endif()
if(onnxruntime_USE_NUPHAR)
  add_definitions(-DUSE_NUPHAR)
endif()
if(onnxruntime_USE_TENSORRT)
  add_definitions(-DUSE_TENSORRT)
endif()
if(onnxruntime_USE_DML)
  message("Enabling DML")
  add_definitions(-DUSE_DML)
endif()

# OpenVINO

find_package(OpenVINO REQUIRED)
#
#option(OPENVINO_ROOTDIR "openvino root dir")
#
#if(NOT OPENVINO_ROOTDIR)
##  set(OPENVINO_ROOTDIR "/opt/intel/openvino_2022")
##  include_directories("/opt/intel/openvino_2022")
##  include_directories("/opt/intel/openvino_2022.1.0.643")
#endif()
#
##TODO: we should only need one of them.
#include_directories("${OPENVINO_ROOTDIR}/runtime/include")
#link_directories("${OPENVINO_ROOTDIR}/runtime/lib/intel64")

# some examples require a Windows build environment
# if(WIN32)
#   add_subdirectory(imagenet)
#   add_subdirectory(MNIST)
# endif()
# add_subdirectory(squeezenet)
# if(PNG_FOUND)
#   add_subdirectory(fns_candy_style_transfer)
# endif()
#missing experimental_onnxruntime_cxx_api.h
add_subdirectory(src)


