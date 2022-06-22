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


def get_runtimes(good_only=True, unsigned_weights=False, is_conv=True):
    runtimes = [
        (False, TensorFlow()),
        (True, DeepSparse(quantization_mode="off")),
        (True, DeepSparse(quantization_mode="static")),
        (True, DeepSparse(quantization_mode="off", sparsity=0.9)),
        (True, DeepSparse(quantization_mode="static", sparsity=0.9)),
        (True, TVM(quantization_mode="off")),
        # For dense we get an error:
        # tvm.error.OpNotImplemented: The following operators are not supported for frontend ONNX: MatMulInteger
        (True, TVM(quantization_mode="static_qdq")),
        (is_conv, TVM(quantization_mode="dynamic", unsigned_weights=unsigned_weights)),
        (True, PyTorch(quantization_mode="off", use_torchscript=True)),
        (True, PyTorch(quantization_mode="dynamic", use_torchscript=True)),
        (True, PyTorch(quantization_mode="static", use_torchscript=True)),
        (True, TFLite(quantization_mode="off")),
        (not is_conv, TFLite(quantization_mode="dynamic")),  # bad for CNN
        (not is_conv, TFLite(quantization_mode="static")),  # bad for CNN
        # (True, TFLite(quantization_mode="off", sparsity=0.9)),
        # (not is_conv, TFLite(sparsity=0.9, quantization_mode="dynamic")),  # bad for CNN
        # (not is_conv, TFLite(sparsity=0.9, quantization_mode="static")),  # bad for CNN
        (True, ONNXRuntime(quantization_mode="off")),
        (
            True,
            ONNXRuntime(quantization_mode="dynamic", unsigned_weights=unsigned_weights),
        ),
        (False, ONNXRuntime(quantization_mode="static_qoperator")),  # bad
        (True, ONNXRuntime(quantization_mode="static_qdq")),
        (False, ONNXRuntimeFromPyTorch(quantization_mode="off")),
        (
            False,
            ONNXRuntimeFromPyTorch(
                quantization_mode="dynamic", unsigned_weights=unsigned_weights
            ),
        ),
        (False, ONNXRuntimeFromPyTorch(quantization_mode="static_qoperator")),  # bad
        (False, ONNXRuntimeFromPyTorch(quantization_mode="static_qdq")),
        #
        #
        (True, OpenVINO(quantization_mode="off")),
        (
            False,
            OpenVINO(quantization_mode="dynamic", unsigned_weights=unsigned_weights),
        ),  # bad
        (True, OpenVINO(quantization_mode="static_qdq")),
    ]

    # runtimes = [
    #     (True, DeepSparse(quantization_mode="off")),
    #     (True, DeepSparse(quantization_mode="static")),
    #     (True, DeepSparse(quantization_mode="off", sparsity=0.9)),
    #     (True, DeepSparse(quantization_mode="static", sparsity=0.9)),
    # ]

    # runtimes = [
    #     (True, PyTorch(quantization_mode="off", use_torchscript=True))
    # ]

    # runtimes = [
    #     (True, ONNXRuntime(quantization_mode="off")),
    #     (True, TVM(quantization_mode="off")),
    # ]

    # should be possible but doesn't work: OpenVINO(quantization_mode="static_qoperator")

    if good_only:
        return [rt for good, rt in runtimes if good]
    else:
        return [rt for good, rt in runtimes]
