import argparse

import tensorflow as tf
import torch
import numpy as np
import pandas as pd
import ddsp.training

import thesis.quantization_benchmark as qb


class ParseDictParams(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, {})
        for value in values:
            key, value = value.split("=")
            getattr(namespace, self.dest)[key] = value


def dense_models(n_sizes=10, n_layers=1):
    input_shape = (32, 32, 3)

    for i in range(n_sizes):
        hidden_size = 2 ** (i + 1)

        dense_layers_keras = []
        dense_layers_torch = []

        for j in range(n_layers):
            dense_layers_keras.append(
                tf.keras.layers.Dense(hidden_size, activation=tf.nn.relu)
            )

            dense_layers_torch += [
                torch.nn.Linear(
                    in_features=np.product(input_shape) if j == 0 else hidden_size,
                    out_features=hidden_size,
                ),
                torch.nn.ReLU(),
            ]

        dense_keras = tf.keras.Sequential(
            [
                tf.keras.layers.Input(input_shape),
                tf.keras.layers.Flatten(),
                *dense_layers_keras,
                tf.keras.layers.Dense(1),
            ]
        )

        dense_torch = torch.nn.Sequential(
            torch.nn.Flatten(),
            *dense_layers_torch,
            torch.nn.Linear(in_features=hidden_size, out_features=1),
        )

        yield dense_keras, dense_torch


def cnn_models(n_sizes=10, n_layers=1):
    for i in range(n_sizes):
        size = 8 * 2**i
        n_channels = 64

        conv_layers_keras = []
        conv_layers_torch = []

        for _ in range(n_layers):
            conv_layers_keras.append(
                tf.keras.layers.Conv2D(
                    filters=n_channels,
                    kernel_size=3,
                    activation=tf.nn.relu,
                    padding="same",
                )
            )

            conv_layers_torch += [
                torch.nn.Conv2d(
                    in_channels=n_channels,
                    out_channels=n_channels,
                    kernel_size=3,
                    padding=[1, 1],
                ),
                torch.nn.ReLU(),
            ]

        cnn_keras = tf.keras.Sequential(
            [
                tf.keras.layers.Input((size, size, n_channels)),
                *conv_layers_keras,
                tf.keras.layers.MaxPooling2D(size),
            ]
        )

        cnn_torch = torch.nn.Sequential(
            *conv_layers_torch,
            torch.nn.MaxPool2d(size),
        )

        yield cnn_keras, cnn_torch


def dilated_conv_models(n_sizes, n_layers):

    for size in range(n_sizes):
        channels = 64 * 2 ** size
        model = ddsp.training.nn.DilatedConvStack(
            ch=channels,
            layers_per_stack=n_layers,
            stacks=1,
            kernel_size=3,
            dilation=2,
        )

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input((64000, 1, 1)),
                model,
            ]
        )

        yield model, None


def main(out_file, n_sizes, n_layers, kind):
    qb.TEMP_DIR = "/cluster/scratch/vvolhejn/tmp"

    runs_per_size = []

    if kind == "dense":
        models = dense_models(n_sizes=n_sizes, n_layers=n_layers)
    elif kind == "cnn":
        models = cnn_models(n_sizes=n_sizes, n_layers=n_layers)
    elif kind == "dilated_cnn":
        models = dilated_conv_models(n_sizes=n_sizes, n_layers=n_layers)
    else:
        raise ValueError("Unknown kind")

    for i, (keras_model, torch_model) in enumerate(models):
        # print(keras_model.summary())

        is_conv = kind != "dense"
        runtimes = qb.get_runtimes(good_only=True, unsigned_weights=is_conv)

        runs = qb.benchmark(keras_model, torch_model, runtimes, n_iterations=10)
        runs["network_size"] = i
        runs_per_size.append(runs)

        # Export after every run to have partial results
        df = pd.concat(runs_per_size)
        df.to_csv(out_file)

    print("Done, saved to", out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out_file")
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--n-sizes", type=int, default=1)
    parser.add_argument(
        "--kind", choices=["dense", "cnn", "dilated_cnn"], required=True
    )

    args = parser.parse_args()
    main(args.out_file, n_layers=args.n_layers, n_sizes=args.n_sizes, kind=args.kind)
