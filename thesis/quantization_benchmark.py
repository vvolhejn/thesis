import os
import subprocess

import numpy as np
import onnx
import tensorflow as tf
import onnxruntime as ort
import torch
import tqdm
from codetiming import Timer
import tf2onnx
import onnxruntime.quantization as ortq
import matplotlib.pyplot as plt
import openvino.runtime
import pandas as pd


class Runtime:
    def __init__(self):
        self.convert_called = False

    def convert(self, orig_model, get_batch_fn=None):
        self.convert_called = True
        self.orig_model = orig_model

    def run(self, data):
        assert self.convert_called, "No model was converted, call convert() first."

    def get_name(self):
        raise NotImplementedError

    def __repr__(self):
        return self.get_name()

    def get_id(self):
        return self.get_name() + "_" + hex(id(self))


class NeedsPyTorchModel:
    """
    Used only to tag runtimes that converts a PyTorch model rather than a Keras one
    """

    pass


class TFLite(Runtime):
    def __init__(self, quantization_mode):
        super().__init__()

        modes = {"off", "dynamic", "static"}
        assert quantization_mode in modes, f"quantization_mode must be one of {modes}"

        self.quantization_mode = quantization_mode

    def convert(self, orig_model, get_batch_fn=None):
        super().convert(orig_model)

        tflite_converter = tf.lite.TFLiteConverter.from_keras_model(self.orig_model)
        tflite_converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # Enable TensorFlow Lite ops.
            tf.lite.OpsSet.SELECT_TF_OPS,  # Enable extended TensorFlow ops.
        ]

        if self.quantization_mode in {"dynamic", "static"}:
            tflite_converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # print(f"Quantization {'on' if quantize else 'off'}.")

        if self.quantization_mode == "static":

            def representative_dataset():
                for i in range(100):
                    yield [get_batch_fn()]

            tflite_converter.representative_dataset = representative_dataset

            # tflite_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            # tflite_converter.inference_input_type = tf.int8  # or tf.uint8
            # tflite_converter.inference_output_type = tf.int8  # or tf.uint8

        tflite_model = tflite_converter.convert()  # Byte string.
        # # Save the model.
        # model_name = "model_quantized.tflite" if quantize else "model_unquantized.tflite"
        save_path = os.path.join("/tmp", self.get_id() + ".tflite")
        with open(save_path, "wb") as f:
            f.write(tflite_model)

        self.save_path = save_path

        interpreter = tf.lite.Interpreter(self.save_path)
        self.signature = interpreter.get_signature_runner()

    def run(self, data):
        input_keys = list(self.signature.get_input_details().keys())
        assert len(input_keys) == 1, "Expected just one input key in the TFLite model."
        input_name = input_keys[0]

        output_dict = self.signature(**{input_name: data})
        output_keys = list(output_dict.keys())
        assert (
            len(output_keys) == 1
        ), "Expected just one output key in the TFLite model."

        return output_dict[output_keys[0]]

    def get_name(self):
        d = {
            "off": "",
            "dynamic": "_quant_dynamic",
            "static": "_quant_static",
        }

        return f"TFLite{d[self.quantization_mode]}"


class TensorFlow(Runtime):
    def run(self, data):
        super().run(data)

        output = self.orig_model(data)
        return output

    def get_name(self):
        return f"TensorFlow"


class PyTorch(Runtime, NeedsPyTorchModel):
    def __init__(self, quantization_mode, use_torchscript=False):
        super().__init__()

        modes = {"off", "dynamic", "static"}
        assert quantization_mode in modes, f"quantization_mode must be one of {modes}"
        self.quantization_mode = quantization_mode
        self.use_torchscript = use_torchscript

    def convert(self, orig_model, get_batch_fn=None):
        super().convert(orig_model)

        if self.quantization_mode == "off":
            self.model = orig_model
        elif self.quantization_mode == "dynamic":
            self.model = torch.quantization.quantize_dynamic(
                orig_model, {torch.nn.Linear}, dtype=torch.qint8
            )
        else:
            assert self.quantization_mode == "static"

            model = PyTorchQuantizationWrapper(orig_model)
            model.eval()
            model.qconfig = torch.quantization.get_default_qconfig("fbgemm")

            # Note: we could also fuse layers using torch.quantization.fuse_modules

            # Prepare for calibration
            model = torch.quantization.prepare(model)

            for i in range(100):
                data = torch.permute(torch.from_numpy(get_batch_fn()), (0, 3, 1, 2))
                model(data)

            self.model = torch.quantization.convert(model)

        if self.use_torchscript:
            data = torch.permute(torch.from_numpy(get_batch_fn()), (0, 3, 1, 2))
            self.model = torch.jit.trace(self.model, data)

    def run(self, data):
        super().run(data)

        data = torch.from_numpy(data)
        # Torch needs NCHW instead of TensorFlow's NHWC
        data = torch.permute(data, (0, 3, 1, 2))

        output = self.model(data)

        return output.detach().numpy()

    def get_name(self):
        d = {
            "off": "",
            "dynamic": "_quant_dynamic",
            "static": "_quant_static",
        }

        prefix = "TorchScript" if self.use_torchscript else "PyTorch"

        return f"{prefix}{d[self.quantization_mode]}"


class PyTorchQuantizationWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

        # QuantStub converts tensors from floating point to quantized
        self.quant = torch.quantization.QuantStub()
        # DeQuantStub converts tensors from quantized to floating point
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x


class ONNXRuntime(Runtime):
    def __init__(
        self,
        quantization_mode,
        unsigned_activations=False,
        unsigned_weights=False,
        # Warning: if the OpenVINO Execution Provider is not available,
        # silently defaults back to CPU
        use_openvino=False,
    ):
        super().__init__()

        modes = {"off", "dynamic", "static_qoperator", "static_qdq"}
        assert quantization_mode in modes, f"quantization_mode must be one of {modes}"

        self.quantization_mode = quantization_mode
        self.unsigned_activations = unsigned_activations
        self.unsigned_weights = unsigned_weights
        self.use_openvino = use_openvino

    def save_model(self, orig_model, get_batch_fn):
        input_signature = [
            tf.TensorSpec(
                [1] + orig_model.input.shape[1:], dtype=np.float32, name="input"
            )
        ]
        onnx_model, _ = tf2onnx.convert.from_keras(
            orig_model, input_signature, opset=13
        )

        save_path = os.path.join("/tmp", self.get_id() + ".onnx")
        onnx.save(onnx_model, save_path)
        self.save_path = save_path

    def convert(self, orig_model, get_batch_fn=None):
        super().convert(orig_model)

        self.save_model(orig_model, get_batch_fn)

        def get_type(is_unsigned):
            return ortq.QuantType.QUInt8 if is_unsigned else ortq.QuantType.QInt8

        if self.quantization_mode in {"dynamic", "static_qoperator", "static_qdq"}:
            save_path_2 = os.path.join("/tmp", self.get_id() + "_2.onnx")

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
                        if self.i == 100:
                            return None
                        else:
                            self.i += 1
                            return {"input": self.get_batch_fn()}

                quant_format = (
                    ortq.QuantFormat.QDQ
                    if self.quantization_mode == "static_qdq"
                    else ortq.QuantFormat.QOperator
                )

                save_path_2 = os.path.join("/tmp", self.get_id() + "_2.onnx")
                ortq.quantize_static(
                    self.save_path,
                    save_path_2,
                    DataReader(lambda: self.preprocess_input(get_batch_fn())),
                    activation_type=get_type(self.unsigned_activations),
                    weight_type=get_type(self.unsigned_weights),
                    per_channel=True,
                    quant_format=quant_format,
                )

            self.save_path = save_path_2

        self.session = ort.InferenceSession(
            self.save_path,
            providers=[
                "OpenVINOExecutionProvider"
                if self.use_openvino
                else "CPUExecutionProvider"
            ],
        )

    def run(self, data):
        super().run(data)
        data = self.preprocess_input(data)

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
        openvino_s = "_openvino" if self.use_openvino else ""
        type_name = type(self).__name__

        return f"{type_name}{d[self.quantization_mode]}{openvino_s}"

    def preprocess_input(self, data):
        """ONNXRuntimeFromPyTorch overrides this to change the order of axes."""
        return data


class ONNXRuntimeFromPyTorch(ONNXRuntime, NeedsPyTorchModel):
    def save_model(self, orig_model, get_batch_fn):
        self.save_path = os.path.join("/tmp", self.get_id() + ".onnx")
        torch.onnx.export(
            orig_model,  # model being run
            torch.permute(torch.from_numpy(get_batch_fn()), (0, 3, 1, 2)),
            self.save_path,
            # where to save the model (can be a file or file-like object)
            export_params=True,
            opset_version=13,
            # whether to execute constant folding for optimization
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
        )

    def preprocess_input(self, data):
        data = np.moveaxis(data, 3, 1)
        return data


class OpenVINO(ONNXRuntime):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def convert(self, orig_model, get_batch_fn=None):
        super().convert(orig_model, get_batch_fn)
        self.save_dir = os.path.join("/tmp", self.get_id())
        os.makedirs(self.save_dir)

        mo_command = [
            "mo",
            "--input_model",
            self.save_path,
            "--output_dir",
            self.save_dir,
        ]
        subprocess.run(mo_command)

        self.ie = openvino.runtime.Core()
        model = self.ie.read_model(
            model=os.path.join(
                self.save_dir,
                os.path.splitext(os.path.basename(self.save_path))[0] + ".xml",
            )
        )
        self.compiled_model = self.ie.compile_model(model=model, device_name="CPU")
        self.input_names = [x.any_name for x in self.compiled_model.inputs]
        self.request = self.compiled_model.create_infer_request()

    def run(self, data):
        # Checks that the model has been converted
        Runtime.run(self, data)

        output = self.request.infer({self.input_names[0]: data})

        output_list = list(output.values())
        assert len(output_list) == 1, "Only one output was expected."
        return output_list[0]


class BenchmarkedRuntime:
    def __init__(self, runtime):
        self.runtime = runtime
        self.timer = Timer(logger=None)
        self.times = []
        self.losses = []

    def run_batch(self, batch, loss_fn, true_output):
        with self.timer:
            output = self.runtime.run(batch)

        assert output is not None, f"No output returned by runtime {self.runtime}"

        self.times.append(self.timer.last)
        if true_output is None:
            self.losses.append(0.0)
        else:
            # print(output)
            self.losses.append(loss_fn(true_output, output).numpy())

        return output


def benchmark(keras_model, torch_model, runtimes, n_iterations=100):
    input_shape = [1] + list(keras_model.input.shape)[1:]

    def make_batch():
        return np.random.randn(*input_shape).astype(np.float32)

    benchmarked_runtimes = []
    for runtime in runtimes:
        print(f"Converting model to runtime {runtime.get_id()}")

        orig_model = (
            torch_model if isinstance(runtime, NeedsPyTorchModel) else keras_model
        )

        runtime.convert(orig_model, get_batch_fn=make_batch)
        benchmarked_runtimes.append(BenchmarkedRuntime(runtime))
    print("Done converting.")

    loss = tf.keras.losses.MeanSquaredError(reduction="sum", name="mean_squared_error")

    batches = [make_batch() for _ in range(n_iterations + 1)]
    true_outputs = {
        "keras": [None for _ in range(n_iterations + 1)],
        "torch": [None for _ in range(n_iterations + 1)],
    }

    for br in tqdm.tqdm(benchmarked_runtimes):
        for i in range(n_iterations + 1):
            batch = batches[i]

            framework = (
                "torch" if isinstance(br.runtime, NeedsPyTorchModel) else "keras"
            )

            output = br.run_batch(batch, loss, true_outputs[framework][i])

            if true_outputs[framework][i] is None:
                # The first runtime's output is considered to be the ground truth
                true_outputs[framework][i] = output

    all_runs = []

    for i, br in enumerate(benchmarked_runtimes):
        # We drop the first round
        assert len(br.losses) > 1
        assert len(br.times) > 1

        for j in range(1, n_iterations + 1):
            all_runs.append(
                {
                    "name": br.runtime.get_name(),
                    "iteration": j,
                    "loss": br.losses[j],
                    "inference_time": br.times[j],
                },
            )

    all_runs = pd.DataFrame(
        columns=["name", "iteration", "loss", "inference_time"], data=all_runs
    )

    return all_runs


def summarize_runs(all_runs_df):
    df = all_runs_df.groupby("name").agg(
        {"loss": ["mean", "std"], "inference_time": ["mean", "std"]}
    )
    # Flatten the hierarchical index
    df.columns = ["_".join(col).strip() for col in df.columns.values]

    return df


def plot_runs(all_runs_df, sort=True):
    import seaborn as sns

    order = None
    if sort:
        order = (
            all_runs_df.groupby("name")
            .agg({"inference_time": "mean"})
            .sort_values("inference_time")
            .index
        )

    ax = sns.barplot(
        data=all_runs_df,
        y="name",
        x="inference_time",
        order=order,
    )

    return ax


def get_runtimes(good_only=True, unsigned_weights=False):
    runtimes = [
        (False, TensorFlow()),
        (True, PyTorch(quantization_mode="off", use_torchscript=True)),
        (True, PyTorch(quantization_mode="dynamic", use_torchscript=True)),
        (True, PyTorch(quantization_mode="static", use_torchscript=True)),
        (True, TFLite(quantization_mode="off")),
        (False, TFLite(quantization_mode="dynamic")),  # bad for CNN
        (False, TFLite(quantization_mode="static")),  # bad for CNN
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
        (True, OpenVINO(quantization_mode="off")),
        (
            False,
            OpenVINO(quantization_mode="dynamic", unsigned_weights=unsigned_weights),
        ),  # bad
        (True, OpenVINO(quantization_mode="static_qdq")),
    ]

    # should be possible but doesn't work: OpenVINO(quantization_mode="static_qoperator")

    if good_only:
        return [rt for good, rt in runtimes if good]
    else:
        return [rt for good, rt in runtimes]
