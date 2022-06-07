import argparse
import os

import cpuinfo
import wandb
import tensorflow as tf
import torch
import numpy as np
import pandas as pd
import ddsp.training
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

import thesis.benchmark as qb
from thesis.prepare_job_util import get_today_string, add_distinguishing_suffix


class ParseDictParams(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, {})
        for value in values:
            key, value = value.split("=")
            getattr(namespace, self.dest)[key] = value


def dense_models(n_sizes=8):
    base_size = 128
    expansion = 4

    for i in range(1, n_sizes + 1):
        hidden_size = base_size * i

        dense_keras = tf.keras.Sequential(
            [
                tf.keras.layers.Input((hidden_size,)),
                tf.keras.layers.Dense(hidden_size * expansion, activation=tf.nn.relu),
                tf.keras.layers.Dense(hidden_size),
            ]
        )

        dense_torch = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=hidden_size, out_features=hidden_size * expansion
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(
                in_features=hidden_size * expansion, out_features=hidden_size
            ),
        )

        yield dense_keras, dense_torch, {"base_size": base_size, "expansion": expansion}


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

        yield cnn_keras, cnn_torch, {"n_channels": n_channels, "n_layers": n_layers}


def dilated_conv_models(n_sizes, n_layers):

    for size in range(n_sizes):
        channels = 64 * 2**size
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

        yield model, None, {"n_channels": channels, "n_layers": n_layers}


def get_keras_model_flops(keras_model, batch_size=1):
    from tensorflow.python.profiler.model_analyzer import profile
    from tensorflow.python.profiler.option_builder import ProfileOptionBuilder

    # model = tf.keras.applications.ResNet50()
    forward_pass = tf.function(
        keras_model.call,
        input_signature=[
            tf.TensorSpec(shape=[batch_size] + list(keras_model.input.shape)[1:])
        ],
    )

    opts = ProfileOptionBuilder().float_operation()
    # Silence stdout output
    opts["output"] = "none"

    graph_info = profile(
        forward_pass.get_concrete_function().graph,
        options=opts,
    )

    return graph_info.total_float_ops


def benchmark_one_runtime(runtime_fn, models, n_iterations):
    runs_per_size = []

    for i, (keras_model, torch_model, metadata) in enumerate(models):
        # We need to recreate the runtime for each model (this is cheap)
        cur_runtime = runtime_fn()

        # The base runtime to compare with, depending on what framework we're using.
        # We need this to get the loss value for sanity checking.
        base_runtime = (
            qb.PyTorch(quantization_mode="off", use_torchscript=True)
            if isinstance(cur_runtime, qb.NeedsPyTorchModel)
            else qb.TensorFlow()
        )

        runtimes = [base_runtime, cur_runtime]

        runs = qb.benchmark(
            keras_model, torch_model, runtimes, n_iterations=n_iterations
        )

        flops = get_keras_model_flops(keras_model)

        # By discarding the first n_iterations rows we get rid of those corresponding
        # to the base runtime
        runs = (
            runs.iloc[n_iterations:]
            .groupby("name")
            .agg(
                {
                    "loss": "mean",
                    "inference_time": ["mean", "std"],
                }
            )
        )
        runs.columns = ["_".join(col).strip() for col in runs.columns.values]

        # Broadcast metadata to all rows
        for k, v in metadata.items():
            runs[k] = v
        runs["model_i"] = i
        runs["flop"] = flops
        runs["flop/s"] = flops / runs["inference_time_mean"]
        # runs["name"] = cur_runtime.get_name()

        runs_per_size.append(runs)

        # log summary to W&B
        for name, row in runs.iterrows():
            row_data = dict(row)
            row_data.update(metadata)
            wandb.log(
                {
                    "name": name,
                    **row_data,
                }
            )

    return pd.concat(runs_per_size).reset_index(drop=True)


def main(save_dir, n_iterations, n_sizes, n_layers, kind):
    qb.TEMP_DIR = "/cluster/scratch/vvolhejn/tmp"

    is_conv = kind != "dense"
    n_runtimes = len(
        qb.get_runtimes(good_only=True, unsigned_weights=is_conv, is_conv=is_conv)
    )

    df = None
    out_file = os.path.join(save_dir, "results.csv")

    results_per_runtime = []

    for runtime_i in range(n_runtimes):
        if kind == "dense":
            assert n_layers == 1, "n_layers is not used for dense models"
            models = dense_models(n_sizes=n_sizes)
        elif kind == "cnn":
            models = cnn_models(n_sizes=n_sizes, n_layers=n_layers)
        elif kind == "dilated_cnn":
            models = dilated_conv_models(n_sizes=n_sizes, n_layers=n_layers)
        else:
            raise ValueError("Unknown kind")

        def runtime_fn():
            return qb.get_runtimes(
                good_only=True, unsigned_weights=is_conv, is_conv=is_conv
            )[runtime_i]

        runtime_name = runtime_fn().get_name()

        wandb.init(
            project="cpu-inference-benchmark",
            entity="neural-audio-synthesis-thesis",
            name=os.path.basename(save_dir) + f"-{runtime_name}",
            group=os.path.basename(save_dir),
            job_type=runtime_name,
            # sync_tensorboard=True,
            config=dict(
                n_iterations=n_iterations,
                n_sizes=n_sizes,
                kind=kind,
                cpu_info=cpuinfo.get_cpu_info(),
                runtime=runtime_name,
            ),
            dir="/cluster/scratch/vvolhejn/wandb",
            reinit=True,
        )

        df = benchmark_one_runtime(runtime_fn, models, n_iterations)
        results_per_runtime.append(df)

        print(df)

        # Export after every run to have partial results
        df = pd.concat(results_per_runtime).reset_index(drop=True)
        os.makedirs(save_dir, exist_ok=True)
        df.to_csv(out_file)

        wandb.finish()
        # if runtime_i == 3:
        #     break

    # assert df is not None, "No models were benchmarked."

    # fig = px.line(df.reset_index(), x="model_i", y="inference_time_mean", color="index")
    # wandb.log({"table": df, "plot": fig})

    # fig, ax = plt.subplots()
    # sns.lineplot(
    #     data=df,
    #     hue="name",
    #     x="model_i",
    #     y="inference_time",
    #     # ci="sd",
    #     ax=ax
    # )
    # # g.set(yscale="log")

    # wandb.log({"table": df, "plot": fig})

    print("Done, saved to", out_file)


def create_job_dir(base_dir, name):
    res = f"{get_today_string()}"
    if name:
        res += f"-{name}"

    res = os.path.join(base_dir, add_distinguishing_suffix(base_dir, res))

    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--n-sizes", type=int, default=1)
    parser.add_argument("--n-iterations", type=int, default=50)
    parser.add_argument(
        "--kind", choices=["dense", "cnn", "dilated_cnn"], required=True
    )

    args = parser.parse_args()

    job_dir = create_job_dir(
        base_dir="/cluster/home/vvolhejn/benchmark_data", name=args.kind
    )

    main(
        job_dir,
        n_layers=args.n_layers,
        n_sizes=args.n_sizes,
        kind=args.kind,
        n_iterations=args.n_iterations,
    )
