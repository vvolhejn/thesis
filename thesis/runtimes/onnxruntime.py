import os

import torch
import onnxruntime.quantization as ortq
import onnx
import numpy as np
import tensorflow as tf
import tf2onnx
import onnxruntime as ort

import thesis.util
from . import Runtime, NeedsPyTorchModel
from .runtime import TEMP_DIR


class ONNXRuntime(Runtime):
    def __init__(
        self,
        quantization_mode,
        unsigned_activations=False,
        unsigned_weights=False,
    ):
        super().__init__()

        modes = {"off", "dynamic", "static_qoperator", "static_qdq"}
        assert quantization_mode in modes, f"quantization_mode must be one of {modes}"

        self.quantization_mode = quantization_mode
        self.unsigned_activations = unsigned_activations
        self.unsigned_weights = unsigned_weights

    def save_model(self, orig_model, get_batch_fn):
        input_signature = [
            tf.TensorSpec(
                [1] + orig_model.input.shape[1:], dtype=np.float32, name="input"
            )
        ]
        onnx_model, _ = tf2onnx.convert.from_keras(
            orig_model, input_signature, opset=13
        )

        save_path = os.path.join(TEMP_DIR, self.get_id() + ".onnx")
        onnx.save(onnx_model, save_path)
        self.save_path = save_path

    def convert(self, orig_model, get_batch_fn=None):
        super().convert(orig_model)

        self.save_model(orig_model, get_batch_fn)

        def get_type(is_unsigned):
            return ortq.QuantType.QUInt8 if is_unsigned else ortq.QuantType.QInt8

        if self.quantization_mode in {"dynamic", "static_qoperator", "static_qdq"}:
            save_path_2 = os.path.join(TEMP_DIR, self.get_id() + "_2.onnx")

            if self.quantization_mode == "dynamic":
                ortq.quantize_dynamic(
                    self.save_path,
                    save_path_2,
                    # Signed weights don't work for convolutions?
                    # see https://github.com/microsoft/onnxruntime/issues/3130
                    weight_type=get_type(self.unsigned_weights),
                    # Cannot set activation type for dynamic quantization.
                )
            else:

                class DataReader(ortq.CalibrationDataReader):
                    def __init__(self, get_batch_fn):
                        self.i = 0
                        self.get_batch_fn = get_batch_fn

                    def get_next(self):
                        if self.i == 10:
                            return None
                        else:
                            self.i += 1
                            return {"input": self.get_batch_fn()}

                quant_format = (
                    ortq.QuantFormat.QDQ
                    if self.quantization_mode == "static_qdq"
                    else ortq.QuantFormat.QOperator
                )

                save_path_2 = os.path.join(TEMP_DIR, self.get_id() + "_2.onnx")
                ortq.quantize_static(
                    self.save_path,
                    save_path_2,
                    DataReader(lambda: get_batch_fn()),
                    activation_type=get_type(self.unsigned_activations),
                    weight_type=get_type(self.unsigned_weights),
                    per_channel=True,
                    quant_format=quant_format,
                )

            self.save_path = save_path_2

        # We need to set intra_op_num_threads because otherwise we get a crash on Euler:
        # see https://github.com/microsoft/onnxruntime/issues/10113
        session_options = ort.SessionOptions()
        n_cpus_available = thesis.util.get_n_cpus_available()

        session_options.intra_op_num_threads = n_cpus_available

        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        )

        self.optimized_model_path = os.path.join(
            TEMP_DIR,
            f"{self.get_id()}_optimized.onnx",
        )

        session_options.optimized_model_filepath = self.optimized_model_path

        self.session = ort.InferenceSession(
            self.save_path,
            providers=["CPUExecutionProvider"],
            sess_options=session_options,
        )

    def run(self, data):
        super().run(data)

        input_names = [inp.name for inp in self.session.get_inputs()]
        assert len(input_names) == 1, "Expected only one input to ONNX model"
        output_names = [output.name for output in self.session.get_outputs()]

        outputs = self.session.run(output_names, {input_names[0]: data})

        return outputs[0]

    def get_name(self):
        d = {
            "off": "",
            "dynamic": "_quant_dynamic",
            "static_qoperator": "_quant_static_qoperator",
            "static_qdq": "_quant_static_qdq",
        }
        type_name = type(self).__name__

        return f"{type_name}{d[self.quantization_mode]}"


class ONNXRuntimeFromPyTorch(ONNXRuntime, NeedsPyTorchModel):
    def save_model(self, orig_model, get_batch_fn):
        self.save_path = os.path.join(TEMP_DIR, self.get_id() + ".onnx")
        torch.onnx.export(
            orig_model,  # model being run
            torch.from_numpy(get_batch_fn()),
            self.save_path,
            # where to save the model (can be a file or file-like object)
            export_params=True,
            opset_version=13,
            # whether to execute constant folding for optimization
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
        )
