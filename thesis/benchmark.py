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
import pandas as pd
import tensorflow_model_optimization as tfmot

TEMP_DIR = "/tmp"


class Runtime:
    def __init__(self):
        self.convert_called = False

    def convert(self, orig_model, get_batch_fn=None):
        self.convert_called = True
        self.orig_model = orig_model

    def run(self, data):
        assert self.convert_called, "No model was converted, call convert() first."

    def run_timed(self, data, timer):
        with timer:
            return self.run(data)

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
    def __init__(self, quantization_mode, sparsity=0.0):
        super().__init__()

        modes = {"off", "dynamic", "static"}
        assert quantization_mode in modes, f"quantization_mode must be one of {modes}"

        self.quantization_mode = quantization_mode

        assert 0.0 <= sparsity < 1.0
        self.sparsity = sparsity

    def convert(self, orig_model, get_batch_fn=None):
        super().convert(orig_model)

        if self.sparsity > 0:
            self.prune()

        tflite_converter = tf.lite.TFLiteConverter.from_keras_model(self.orig_model)
        tflite_converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # Enable TensorFlow Lite ops.
            tf.lite.OpsSet.SELECT_TF_OPS,  # Enable extended TensorFlow ops.
        ]

        tflite_converter.optimizations = []

        if self.quantization_mode in {"dynamic", "static"}:
            tflite_converter.optimizations.append(tf.lite.Optimize.DEFAULT)

        if self.sparsity > 0:
            tflite_converter.optimizations.append(
                tf.lite.Optimize.EXPERIMENTAL_SPARSITY
            )

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
        save_path = os.path.join(TEMP_DIR, self.get_id() + ".tflite")
        with open(save_path, "wb") as f:
            f.write(tflite_model)

        self.save_path = save_path

        interpreter = tf.lite.Interpreter(self.save_path)
        self.signature = interpreter.get_signature_runner()

    def prune(self):
        is_conv = False
        for layer in self.orig_model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                print("It's conv!")
                is_conv = True

        model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(
            # We have to clone the model, otherwise the original weights are modified
            tf.keras.models.clone_model(self.orig_model),
            pruning_schedule=tfmot.sparsity.keras.ConstantSparsity(
                self.sparsity, begin_step=0
            ),
            # block_size=(1, 1) if is_conv else (1, 16),
        )
        callbacks = [
            tfmot.sparsity.keras.UpdatePruningStep(),
        ]

        model_for_pruning.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer="adam",
            metrics=["accuracy"],
        )

        n_samples = 100
        input_shape = [n_samples] + list(self.orig_model.input.shape)[1:]
        output_shape = [n_samples] + list(self.orig_model.output.shape)[1:]

        model_for_pruning.fit(
            np.random.randn(*input_shape).astype(np.float32),
            np.random.randn(*output_shape).astype(np.float32),
            callbacks=callbacks,
            epochs=2,
            # larger batch size misbihaves:
            # https://github.com/tensorflow/model-optimization/issues/973
            batch_size=1,
            # validation_split=0.1,
            verbose=0,
        )

        # Sanity check - did we really prune?
        for l in model_for_pruning.layers:
            if isinstance(l.layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)):
                layer_sparsity = (l.layer.weights[0] == 0).numpy().mean()
                assert (
                    layer_sparsity > 0.99 * self.sparsity
                ), f"Layer {l.layer} has sparsity {layer_sparsity}, expected {self.sparsity}"

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
        res = f"TFLite{d[self.quantization_mode]}"

        if self.sparsity > 0:
            res += f"_{self.sparsity:.2f}_sparse"

        return res


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
                orig_model,
                # {torch.nn.Linear},
                dtype=torch.qint8,
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
                data = torch.from_numpy(get_batch_fn())
                model(data)

            self.model = torch.quantization.convert(model)

        if self.use_torchscript:
            data = torch.from_numpy(get_batch_fn())
            self.model = torch.jit.trace(self.model, data)

    def run(self, data):
        super().run(data)

        data = torch.from_numpy(data)

        output = self.model(data)

        return output.detach().numpy()

    def run_timed(self, data, timer):
        """Run in a way that doesn't include the operations around"""

        data = torch.from_numpy(data)
        # Torch needs NCHW instead of TensorFlow's NHWC
        # data = torch.permute(data, (0, 3, 1, 2))

        with timer:
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
                    DataReader(lambda: self.preprocess_input(get_batch_fn())),
                    activation_type=get_type(self.unsigned_activations),
                    weight_type=get_type(self.unsigned_weights),
                    per_channel=True,
                    quant_format=quant_format,
                )

            self.save_path = save_path_2

        # We need to set intra_op_num_threads because otherwise we get a crash on Euler:
        # see https://github.com/microsoft/onnxruntime/issues/10113
        session_options = ort.SessionOptions()
        try:
            n_cpus_available = len(os.sched_getaffinity(0))
        except AttributeError:
            # `os.sched_getaffinity()` is not available everywhere - e.g. not on my Mac.
            # The alternative `os.cpu_count()` counts all CPUs, not just the ones
            # that the process is allowed to use. Good enough.
            n_cpus_available = os.cpu_count()

        session_options.intra_op_num_threads = n_cpus_available

        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        )
        session_options.optimized_model_filepath = os.path.join(
            TEMP_DIR,
            f"{self.get_id()}_optimized.onnx",
        )

        self.session = ort.InferenceSession(
            self.save_path,
            providers=[
                "OpenVINOExecutionProvider"
                if self.use_openvino
                else "CPUExecutionProvider"
            ],
            sess_options=session_options,
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

    def preprocess_input(self, data):
        data = np.moveaxis(data, 3, 1)
        return data


class OpenVINO(ONNXRuntime):
    def __init__(self, *args, **kwargs):
        # import openvino.runtime

        super().__init__(*args, **kwargs)

    def convert(self, orig_model, get_batch_fn=None):
        from openvino.inference_engine import IECore

        super().convert(orig_model, get_batch_fn)
        self.save_dir = os.path.join(TEMP_DIR, self.get_id())
        os.makedirs(self.save_dir)

        mo_command = [
            "mo",
            "--input_model",
            self.save_path,
            "--output_dir",
            self.save_dir,
        ]
        subprocess.run(mo_command)

        # self.ie = openvino.runtime.Core()
        # model = self.ie.read_model(
        #     model=os.path.join(
        #         self.save_dir,
        #         os.path.splitext(os.path.basename(self.save_path))[0] + ".xml",
        #     )
        # )
        # self.compiled_model = self.ie.compile_model(model=model, device_name="CPU")
        # self.input_names = [x.any_name for x in self.compiled_model.inputs]
        # self.request = self.compiled_model.create_infer_request()

        self.ie = IECore()
        model = self.ie.read_network(
            model=os.path.join(
                self.save_dir,
                os.path.splitext(os.path.basename(self.save_path))[0] + ".xml",
            )
        )
        self.compiled_model = self.ie.load_network(network=model, device_name="CPU")
        self.input_names = list(self.compiled_model.input_info)
        # self.request = self.compiled_model.create_infer_request()

    def run(self, data):
        # Checks that the model has been converted
        Runtime.run(self, data)

        output = self.compiled_model.infer({self.input_names[0]: data})
        # output = self.request.infer({self.input_names[0]: data})

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
        output = self.runtime.run_timed(batch, self.timer)

        assert output is not None, f"No output returned by runtime {self.runtime}"

        self.times.append(self.timer.last)
        if true_output is None:
            self.losses.append(0.0)
        else:
            # print(output)
            self.losses.append(loss_fn(true_output, output).numpy())

        return output


def benchmark(keras_model, torch_model, runtimes, n_iterations=100):
    keras_input_shape = [1] + list(keras_model.input.shape)[1:]

    def keras_make_batch():
        return np.random.randn(*keras_input_shape).astype(np.float32)

    if len(keras_input_shape) == 4:
        b, h, w, c = keras_input_shape
        torch_input_shape = [b, c, h, w]
        print(
            f"4D input detected -> reordering torch input shape from "
            f"{keras_input_shape} to {torch_input_shape}"
        )
    else:
        torch_input_shape = keras_input_shape

    def torch_make_batch():
        return np.random.randn(*torch_input_shape).astype(np.float32)

    benchmarked_runtimes = []
    for runtime in runtimes:
        print(f"Converting model to runtime {runtime.get_id()}")

        is_torch_model = isinstance(runtime, NeedsPyTorchModel)
        orig_model = torch_model if is_torch_model else keras_model

        # The Torch model might be None, in that case just skip the PyTorch runtimes
        if orig_model is not None:
            runtime.convert(
                orig_model,
                get_batch_fn=torch_make_batch if is_torch_model else keras_make_batch,
            )
            benchmarked_runtimes.append(BenchmarkedRuntime(runtime))

    print("Done converting.")

    loss = tf.keras.losses.MeanSquaredError(reduction="sum", name="mean_squared_error")

    batches = [keras_make_batch() for _ in range(n_iterations + 1)]
    true_outputs = {
        "keras": [None for _ in range(n_iterations + 1)],
        "torch": [None for _ in range(n_iterations + 1)],
    }

    for br in benchmarked_runtimes:
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


def get_runtimes(good_only=True, unsigned_weights=False, is_conv=True):
    runtimes = [
        (False, TensorFlow()),
        (True, PyTorch(quantization_mode="off", use_torchscript=True)),
        (True, PyTorch(quantization_mode="dynamic", use_torchscript=True)),
        (True, PyTorch(quantization_mode="static", use_torchscript=True)),
        (True, TFLite(quantization_mode="off")),
        (not is_conv, TFLite(quantization_mode="dynamic")),  # bad for CNN
        (not is_conv, TFLite(quantization_mode="static")),  # bad for CNN
        (True, TFLite(quantization_mode="off", sparsity=0.9)),
        (not is_conv, TFLite(sparsity=0.9, quantization_mode="dynamic")),  # bad for CNN
        (not is_conv, TFLite(sparsity=0.9, quantization_mode="static")),  # bad for CNN
        (True, TFLite(quantization_mode="off", sparsity=0.99)),
        (
            not is_conv,
            TFLite(sparsity=0.99, quantization_mode="dynamic"),
        ),  # bad for CNN
        (not is_conv, TFLite(sparsity=0.99, quantization_mode="static")),  # bad for CNN
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

    # should be possible but doesn't work: OpenVINO(quantization_mode="static_qoperator")

    if good_only:
        return [rt for good, rt in runtimes if good]
    else:
        return [rt for good, rt in runtimes]