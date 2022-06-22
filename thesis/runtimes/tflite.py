import os

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

from .runtime import TEMP_DIR
from . import Runtime


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
            if hasattr(l, "layer") and isinstance(
                l.layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)
            ):
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
