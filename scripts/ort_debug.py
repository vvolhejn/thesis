import os

import tensorflow as tf
import onnxruntime as ort
import tf2onnx
import numpy as np
import onnx

import thesis.benchmark as qb

dense_keras = tf.keras.Sequential(
    [
        tf.keras.layers.Input((32, 32, 3)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation=tf.nn.relu),
        tf.keras.layers.Dense(1),
    ]
)

input_signature = [
            tf.TensorSpec(
                [1] + dense_keras.input.shape[1:], dtype=np.float32, name="input"
            )
        ]

onnx_model, _ = tf2onnx.convert.from_keras(
    dense_keras, input_signature, opset=13
)

TEMP_DIR = "/tmp"
save_path = os.path.join(TEMP_DIR, "debug.onnx")
onnx.save(onnx_model, save_path)

onnx_model = onnx.load(save_path)
onnx.checker.check_model(onnx_model)

# We need to set intra_op_num_threads:
# see https://github.com/microsoft/onnxruntime/issues/10113
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 32

session = ort.InferenceSession(
        save_path,
        # providers=[
        #     "CPUExecutionProvider"
        # ],
        sess_options=sess_options
    )

# qb.TEMP_DIR = "/cluster/home/vvolhejn/tmp"
# rt = qb.ONNXRuntime("off")
# rt.convert(dense_keras)
