# import os
# import subprocess
#
# import numpy as np
# import onnx
# import tensorflow as tf
# import onnxruntime as ort
# import torch
# import tqdm
# from codetiming import Timer
# import matplotlib.pyplot as plt
# import pandas as pd
# import tvm.relay as relay
# import tvm
# from tvm.contrib import graph_executor
# import deepsparse
#
# from thesis import util

from .runtime import Runtime, NeedsPyTorchModel
from .tensorflow import TensorFlow
from .pytorch import PyTorch
from .onnxruntime import ONNXRuntime, ONNXRuntimeFromPyTorch
from .tvm import TVM
from .openvino import OpenVINO
from .tflite import TFLite
from .deepsparse import DeepSparse


def get_runtimes(good_only=True, is_conv=True):
    unsigned_weights = is_conv

    runtimes = [
        (False, TensorFlow()),
        (True, DeepSparse(quantization_mode="off")),
        (True, DeepSparse(quantization_mode="static")),
        (True, DeepSparse(quantization_mode="off", sparsity=0.9)),
        (True, DeepSparse(quantization_mode="static", sparsity=0.9)),
        (True, TVM(quantization_mode="off")),
        # Running TVM with dynamic quantization on dense models, we get an error:
        # tvm.error.OpNotImplemented: The following operators are not supported for frontend ONNX: MatMulInteger
        (is_conv, TVM(quantization_mode="dynamic", unsigned_weights=unsigned_weights)),
        (True, TVM(quantization_mode="static_qdq")),
        # TVM with quantization (both static and dynamic) apparently can't use pruning
        # (True, TVM(quantization_mode="off", sparsity=0.9)),
        (True, PyTorch(quantization_mode="off", use_torchscript=True)),
        (True, PyTorch(quantization_mode="dynamic", use_torchscript=True)),
        (True, PyTorch(quantization_mode="static", use_torchscript=True)),
        (True, TFLite(quantization_mode="off")),
        (not is_conv, TFLite(quantization_mode="dynamic")),  # bad for CNN
        (not is_conv, TFLite(quantization_mode="static")),  # bad for CNN
        # TFLite sparsity doesn't lead to speedups
        # (True, TFLite(quantization_mode="off", sparsity=0.9)),
        # (not is_conv, TFLite(sparsity=0.9, quantization_mode="dynamic")),  # bad for CNN
        # (not is_conv, TFLite(sparsity=0.9, quantization_mode="static")),  # bad for CNN
        (True, ONNXRuntime(quantization_mode="off")),
        # Using unsigned weights for dynamic quantization on CNNs doesn't work:
        # [ONNXRuntimeError] : 9 : NOT_IMPLEMENTED : Could not find an implementation for ConvInteger(10) node with name '...'
        # https://github.com/microsoft/onnxruntime/issues/6430
        # In general, static is preferred for CNNs  :
        # https://onnxruntime.ai/docs/performance/quantization.html#method-selection
        (
            True,
            ONNXRuntime(quantization_mode="dynamic", unsigned_weights=unsigned_weights),
        ),
        (False, ONNXRuntime(quantization_mode="static_qoperator")),  # bad
        (True, ONNXRuntime(quantization_mode="static_qdq")),
        (True, OpenVINO(quantization_mode="off")),
        (
            False,
            OpenVINO(quantization_mode="dynamic", unsigned_weights=unsigned_weights),
        ),  # bad
        (True, OpenVINO(quantization_mode="static_qdq")),
    ]

    # runtimes = [
    #     (True, TVM(quantization_mode="off")),
    #     (
    #         True,
    #         TVM(
    #             quantization_mode="off",
    #             tuning_records_path="/home/vaclav/benchmark_data/tuning_test_100_1cores.json",
    #         ),
    #     ),
    # ]

    # runtimes = [
    #     (True, DeepSparse(quantization_mode="off")),
    #     (True, DeepSparse(quantization_mode="static")),
    #     (True, DeepSparse(quantization_mode="off", sparsity=0.9)),
    #     (True, DeepSparse(quantization_mode="static", sparsity=0.9)),
    # ]

    # should be possible but doesn't work: OpenVINO(quantization_mode="static_qoperator")

    if good_only:
        return [rt for good, rt in runtimes if good]
    else:
        return [rt for good, rt in runtimes]
