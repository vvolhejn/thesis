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

    def save_model(self, orig_model, get_batch_fn, input_signature=None):
        if input_signature is None:
            input_signature = [
                tf.TensorSpec(
                    [1] + orig_model.input.shape[1:], dtype=np.float32, name="input"
                )
            ]
        onnx_model, _ = tf2onnx.convert.from_keras(
            orig_model, input_signature, opset=13
        )

        # TEMP_DIR = "/cluster/scratch/vvolhejn/tmp"
        save_path = os.path.join(TEMP_DIR, self.get_id() + ".onnx")
        onnx.save(onnx_model, save_path)
        self.save_path = save_path

    def convert(
        self,
        orig_model,
        get_batch_fn=None,
        n_calibration_batches=10,
        calibration_method: ortq.CalibrationMethod = ortq.CalibrationMethod.MinMax,
    ):
        super().convert(orig_model)

        self.save_model(orig_model, get_batch_fn)
        print(f"Saved model to {self.save_path}")

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
                    # Cannot set activation type for dynamic quantization.
                    weight_type=get_type(self.unsigned_weights),
                    per_channel=True,
                    reduce_range=True,
                    extra_options={
                        # The default is WeightSymmetric == not self.unsigned_weights:
                        # https://github.com/microsoft/onnxruntime/blob/f72288b453bedfc5ae2d1eab4725c014862db8d1/onnxruntime/python/tools/quantization/onnx_quantizer.py#L89
                        # But let's make sure it's always True
                        "WeightSymmetric": True,
                        # This is the default, but let's make it explicit
                        "ActivationSymmetric": False,
                    },
                )
            else:

                class DataReader(ortq.CalibrationDataReader):
                    def __init__(self, get_batch_fn):
                        self.i = 0
                        self.get_batch_fn = get_batch_fn

                    def get_next(self):
                        if self.i == n_calibration_batches:
                            return None
                        else:
                            self.i += 1
                            if self.i % 10 == 0:
                                print(f"Calibration: {self.i}/{n_calibration_batches}")
                            return {"input": self.get_batch_fn()}

                quant_format = (
                    ortq.QuantFormat.QDQ
                    if self.quantization_mode == "static_qdq"
                    else ortq.QuantFormat.QOperator
                )

                calib_moving_average = True

                save_path_2 = os.path.join(TEMP_DIR, self.get_id() + "_2.onnx")
                ortq.quantize_static(
                    self.save_path,
                    save_path_2,
                    DataReader(lambda: get_batch_fn()),
                    activation_type=get_type(self.unsigned_activations),
                    weight_type=get_type(self.unsigned_weights),
                    per_channel=True,
                    reduce_range=True,
                    quant_format=quant_format,
                    extra_options={
                        # These are the defaults, but let's make it explicit
                        "WeightSymmetric": True,
                        "ActivationSymmetric": False,
                        "CalibMovingAverage": calib_moving_average,
                    },
                    calibrate_method=calibration_method,
                )
                print(
                    f"After static quantization with {calibration_method}"
                    + (f" with moving average" if calib_moving_average else "")
                    + f" saved to {save_path_2}"
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
