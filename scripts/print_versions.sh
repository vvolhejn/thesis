#!/bin/bash

which python

python --version

python -m pip freeze | rg '(torch|tensorflow|onnx|tflite|onnxruntime|tvm|openvino|deepsparse|sparseml)='