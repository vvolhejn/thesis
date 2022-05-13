import os

import numpy as np
import onnx
import tensorflow as tf
import onnxruntime as ort
import rich.table
from rich import print as rprint
import tqdm
from codetiming import Timer
import tf2onnx
import onnxruntime.quantization as ortq
import matplotlib.pyplot as plt


class Runtime:
    def __init__(self):
        self.convert_called = False

    def convert(self, keras_model, get_batch_fn=None):
        self.convert_called = True
        self.keras_model = keras_model

    def run(self, data):
        assert self.convert_called, "No model was converted, call convert() first."

    def get_name(self):
        raise NotImplementedError

    def __repr__(self):
        return self.get_name()

    def get_id(self):
        return self.get_name() + "_" + hex(id(self))


class TFLite(Runtime):
    def __init__(self, quantization_mode):
        super().__init__()

        modes = {"off", "dynamic", "static"}
        assert quantization_mode in modes, f"quantization_mode must be one of {modes}"

        self.quantization_mode = quantization_mode

    def convert(self, keras_model, get_batch_fn=None):
        super().convert(keras_model)

        tflite_converter = tf.lite.TFLiteConverter.from_keras_model(self.keras_model)
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
        output = self.keras_model(data)
        return output

    def get_name(self):
        return f"TensorFlow"


class ONNXRuntime(Runtime):
    def __init__(
        self, quantization_mode, unsigned_activations=False, unsigned_weights=False
    ):
        super().__init__()

        modes = {"off", "dynamic", "static_qoperator", "static_qdq"}
        assert quantization_mode in modes, f"quantization_mode must be one of {modes}"

        self.quantization_mode = quantization_mode
        self.unsigned_activations = unsigned_activations
        self.unsigned_weights = unsigned_weights

    def convert(self, keras_model, get_batch_fn=None):
        super().convert(keras_model)
        input_signature = [
            tf.TensorSpec([1] + keras_model.input.shape[1:], dtype=np.float32, name="input")
        ]
        onnx_model, _ = tf2onnx.convert.from_keras(
            keras_model, input_signature, opset=13
        )

        save_path = os.path.join("/tmp", self.get_id() + ".onnx")
        onnx.save(onnx_model, save_path)
        self.save_path = save_path

        def get_type(is_unsigned):
            return ortq.QuantType.QUInt8 if is_unsigned else ortq.QuantType.QInt8

        if self.quantization_mode in {"dynamic", "static_qoperator", "static_qdq"}:
            save_path_2 = os.path.join("/tmp", self.get_id() + "_2.onnx")

            if self.quantization_mode == "dynamic":
                ortq.quantize_dynamic(
                    save_path,
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
                    save_path,
                    save_path_2,
                    DataReader(get_batch_fn),
                    activation_type=get_type(self.unsigned_activations),
                    weight_type=get_type(self.unsigned_weights),
                    per_channel=True,
                    quant_format=quant_format,
                )

            self.save_path = save_path_2

        self.session = ort.InferenceSession(self.save_path)

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

        return f"ONNXRuntime{d[self.quantization_mode]}"


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


def benchmark(keras_model, runtimes, n_iterations=100):
    input_shape = [1] + list(keras_model.input.shape)[1:]

    def make_batch():
        return np.random.randn(*input_shape).astype(np.float32)

    benchmarked_runtimes = []
    for runtime in runtimes:
        print(f"Converting model to runtime {runtime.get_id()}")
        runtime.convert(keras_model, get_batch_fn=make_batch)
        benchmarked_runtimes.append(BenchmarkedRuntime(runtime))
    print("Done converting.")

    loss = tf.keras.losses.MeanSquaredError(reduction="sum", name="mean_squared_error")

    for i in tqdm.trange(n_iterations):
        batch = make_batch()

        true_output = None

        for br in benchmarked_runtimes:
            output = br.run_batch(batch, loss, true_output)

            if true_output is None:
                # The first runtime's output is considered to be the ground truth
                true_output = output

    table = rich.table.Table(title="Summary")
    table.add_column("Name")
    table.add_column("Loss")
    table.add_column("Inference time")
    table.add_column("Relative time")

    mean_times = []
    stds = []

    baseline_time = None
    for i, br in enumerate(benchmarked_runtimes):
        assert len(br.losses) > 0
        assert len(br.times) > 1  # We drop the first round

        #mean_time = np.median(np.array(br.times)[1:])
        mean_time = np.array(br.times)[1:].mean()
        if baseline_time is None:
            baseline_time = mean_time

        mean_times.append(mean_time)
        stds.append(np.array(br.times)[1:].std())

        table.add_row(
            br.runtime.get_name(),
            f"{np.mean(br.losses):.2e}",
            f"{mean_time:.5f}",
            f"{mean_time / baseline_time:.3f}",
        )

    rprint(table)

    print(mean_times[0], stds[0])

    plt.barh([br.runtime.get_name() for br in benchmarked_runtimes], mean_times)

    return benchmarked_runtimes
