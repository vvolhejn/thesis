import argparse
import os
import socket
import psutil

import numpy as np
import tensorflow as tf
from codetiming import Timer
import pandas as pd
import cpuinfo
import wandb
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

import thesis.benchmarking_models
from thesis.prepare_job_util import get_today_string, add_distinguishing_suffix

import thesis.runtimes

if socket.gethostname().startswith("eu-login"):
    # ETH cluster (Euler)
    TEMP_DIR = "/cluster/scratch/vvolhejn/tmp"
    BENCHMARK_DATA_DIR = "/cluster/home/vvolhejn/benchmark_data"
    WANDB_DIR = "/cluster/scratch/vvolhejn/wandb"
elif socket.gethostname() == "n1-west1-1":
    TEMP_DIR = "/tmp"
    BENCHMARK_DATA_DIR = "/home/vaclav/benchmark_data"
    WANDB_DIR = "/home/vaclav/wandb"
else:
    raise RuntimeError(f"Unrecognized host name: {socket.gethostname()}")


class BenchmarkedRuntime:
    def __init__(self, runtime):
        self.runtime = runtime
        self.timer = Timer(logger=None)
        self.results = []

    def run_batch(self, batch, loss_fn, true_output):
        # clear_cache()
        output = self.runtime.run_timed(batch, self.timer)

        assert output is not None, f"No output returned by runtime {self.runtime}"

        if true_output is None:
            cur_loss = 0.0
        else:
            # print(output)
            cur_loss = loss_fn(true_output, output).numpy()

        self.results.append(
            {
                "cpu_percent": psutil.cpu_percent(),
                "loss": cur_loss,
                "inference_time_s": self.timer.last,
            }
        )

        return output


def clear_cache():
    """
    Clear the cache before benchmarking
    see "Achieving accurate and context-sensitive timing for code optimization"
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.113.2520&rep=rep1&type=pdf
    """
    size = int(5e6)
    a = np.random.rand(size)
    assert a.sum() < size


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

        is_torch_model = isinstance(runtime, thesis.runtimes.NeedsPyTorchModel)
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

    keras_batches = [keras_make_batch() for _ in range(n_iterations + 1)]
    torch_batches = [torch_make_batch() for _ in range(n_iterations + 1)]
    true_outputs = {
        "keras": [None for _ in range(n_iterations + 1)],
        "torch": [None for _ in range(n_iterations + 1)],
    }

    for br in benchmarked_runtimes:
        for i in range(n_iterations + 1):
            framework = (
                "torch"
                if isinstance(br.runtime, thesis.runtimes.NeedsPyTorchModel)
                else "keras"
            )

            batch = keras_batches[i] if framework == "keras" else torch_batches[i]

            output = br.run_batch(batch, loss, true_outputs[framework][i])

            if true_outputs[framework][i] is None:
                # The first runtime's output is considered to be the ground truth
                true_outputs[framework][i] = output

    all_runs = []

    for i, br in enumerate(benchmarked_runtimes):
        # We drop the first round
        assert len(br.results) > 1

        for j in range(1, n_iterations + 1):
            all_runs.append(
                {
                    "name": br.runtime.get_name(),
                    "iteration": j,
                    **br.results[j],
                },
            )

    all_runs = pd.DataFrame(
        columns=["name", "iteration", "loss", "inference_time_s", "cpu_percent"],
        data=all_runs,
    )

    return all_runs


def summarize_runs(all_runs_df):
    df = all_runs_df.groupby("name").agg(
        {"loss": ["mean", "std"], "inference_time_s": ["mean", "std"]}
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
            .agg({"inference_time_s": "mean"})
            .sort_values("inference_time_s")
            .index
        )

    ax = sns.barplot(
        data=all_runs_df,
        y="name",
        x="inference_time_s",
        order=order,
    )

    return ax


class ParseDictParams(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, {})
        for value in values:
            key, value = value.split("=")
            getattr(namespace, self.dest)[key] = value


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
    raw_runs_per_size = []

    for i, (keras_model, torch_model, metadata) in enumerate(models):
        # We need to recreate the runtime for each model (this is cheap)
        cur_runtime = runtime_fn()

        # The base runtime to compare with, depending on what framework we're using.
        # We need this to get the loss value for sanity checking.
        base_runtime = (
            thesis.runtimes.PyTorch(quantization_mode="off", use_torchscript=True)
            if isinstance(cur_runtime, thesis.runtimes.NeedsPyTorchModel)
            else thesis.runtimes.TensorFlow()
        )

        runtimes = [base_runtime, cur_runtime]

        flops = get_keras_model_flops(keras_model)

        raw_runs = benchmark(
            keras_model, torch_model, runtimes, n_iterations=n_iterations
        )

        # By discarding the first n_iterations rows we get rid of those corresponding
        # to the base runtime
        raw_runs = raw_runs.iloc[n_iterations:]

        def apply_metadata(df):
            # Broadcast metadata to all rows
            df["model_i"] = i
            df["flop"] = flops
            df["name"] = cur_runtime.get_name()

        runs = raw_runs.groupby("name").agg(
            {
                "loss": "mean",
                "inference_time_s": ["mean", "std"],
                "cpu_percent": "mean",
            }
        )

        assert len(runs) > 0

        runs.columns = ["_".join(col).strip() for col in runs.columns.values]

        apply_metadata(runs)
        apply_metadata(raw_runs)

        for k, v in metadata.items():
            runs[k] = v
        runs["flop/s"] = flops / runs["inference_time_s_mean"]
        # runs["name"] = cur_runtime.get_name()

        runs_per_size.append(runs)
        raw_runs_per_size.append(raw_runs)

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

    return pd.concat(raw_runs_per_size).reset_index(drop=True), pd.concat(
        runs_per_size
    ).reset_index(drop=True)


def main(save_dir, n_iterations, n_sizes, n_layers, kind):
    # thesis.runtimes.TEMP_DIR = "/cluster/scratch/vvolhejn/tmp"

    is_conv = kind != "dense"
    n_runtimes = len(
        thesis.runtimes.get_runtimes(
            good_only=True, unsigned_weights=is_conv, is_conv=is_conv
        )
    )
    out_file = os.path.join(save_dir, "results.csv")

    results_per_runtime = []
    raw_results_per_runtime = []

    for runtime_i in range(n_runtimes):
        models = thesis.benchmarking_models.get_models(kind, n_sizes, n_layers)

        def runtime_fn():
            return thesis.runtimes.get_runtimes(
                good_only=True, unsigned_weights=is_conv, is_conv=is_conv
            )[runtime_i]

        runtime_name = runtime_fn().get_name()

        if (
            isinstance(runtime_fn(), thesis.runtimes.NeedsPyTorchModel)
            # a bit unfortunate - since `models` is a generator, we can't look at the first
            # element without consuming it
            and next(thesis.benchmarking_models.get_models(kind, n_sizes, n_layers))[1]
            is None
        ):
            print(f"Skipping {runtime_name} because no PyTorch model is available")
            continue

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
            dir=WANDB_DIR,
            reinit=True,
        )

        raw_runs, runs = benchmark_one_runtime(runtime_fn, models, n_iterations)

        results_per_runtime.append(runs)
        raw_results_per_runtime.append(raw_runs)

        print(runs)

        # Export after every run to have partial results
        df = pd.concat(raw_results_per_runtime).reset_index(drop=True)
        os.makedirs(save_dir, exist_ok=True)
        df.to_csv(out_file)

        wandb.finish()
        # if runtime_i == 3:
        #     break

    # assert df is not None, "No models were benchmarked."

    # fig = px.line(df.reset_index(), x="model_i", y="inference_time_s_mean", color="index")
    # wandb.log({"table": df, "plot": fig})

    # fig, ax = plt.subplots()
    # sns.lineplot(
    #     data=df,
    #     hue="name",
    #     x="model_i",
    #     y="inference_time_s",
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
        "--kind",
        choices=["dense", "inverted_bottleneck", "cnn", "dilated_cnn"],
        required=True,
    )

    args = parser.parse_args()

    job_dir = create_job_dir(base_dir=BENCHMARK_DATA_DIR, name=args.kind)

    main(
        job_dir,
        n_layers=args.n_layers,
        n_sizes=args.n_sizes,
        kind=args.kind,
        n_iterations=args.n_iterations,
    )
