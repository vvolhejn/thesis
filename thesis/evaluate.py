"""
Based on DDSP's eval_utils, but the requirements are different (we care a lot about
speed) so it made sense to create a separate module.
"""

import cProfile
import pstats
import os
import time

import plotly.express as px
from codetiming import Timer
import wandb
from absl import logging
import ddsp
from ddsp.training import data
import gin
import tensorflow.compat.v2 as tf
import numpy as np


# ---------------------- Evaluation --------------------------------------------
@gin.configurable
def nas_evaluate(
    data_provider,
    model,
    evaluator_classes=None,
    mode="eval",
    save_dir="/tmp/ddsp/training",
    restore_dir="",
    batch_size=1,
    num_batches=10,
    evaluate_and_sample=False,
):
    """Run evaluation.

    Args:
      data_provider: DataProvider instance.
      model: Model instance.
      evaluator_classes: List of BaseEvaluators subclasses (not instances).
      mode: Whether to 'eval' with metrics or create 'sample' s.
      save_dir: Path to directory to save summary events.
      restore_dir: Path to directory with checkpoints, defaults to save_dir.
      batch_size: Size of each eval/sample batch.
      num_batches: How many batches to eval from dataset. -1 denotes all batches.
      evaluate_and_sample: Run both evaluate() and sample() on the batches.

    Returns:
      If the mode is 'eval', then returns a dictionary of Tensors keyed by loss
      type. Otherwise, returns None.
    """

    # Default to restoring from the save directory.
    restore_dir = save_dir if not restore_dir else restore_dir

    # Set up the summary writer and metrics.
    summary_dir = os.path.join(save_dir, "summaries", "eval")
    summary_writer = tf.summary.create_file_writer(summary_dir)

    checkpoint_path = tf.train.latest_checkpoint(restore_dir, latest_filename=None)
    # Get the dataset.
    dataset = data_provider.get_batch(batch_size=batch_size, shuffle=False, repeats=-1)
    # Set number of batches.
    # If num_batches >=1 set it to a huge value to go through the whole dataset
    # (StopIteration will be caught).
    num_batches = num_batches if num_batches >= 1 else int(1e12)

    # Get audio sample rate
    sample_rate = data_provider.sample_rate
    # Get feature frame rate
    frame_rate = data_provider.frame_rate

    latest_losses = None

    # Initialize evaluators.
    evaluators = [
        evaluator_class(sample_rate, frame_rate)
        for evaluator_class in evaluator_classes
    ]

    wandb_run = wandb.init(
        project="neural-audio-synthesis-thesis",
        entity="neural-audio-synthesis-thesis",
        # name=os.path.basename(save_dir),
        sync_tensorboard=True,
        # config=flag_values_dict,
        dir="/cluster/scratch/vvolhejn/wandb",
        tags=["eval"],
    )

    with summary_writer.as_default():
        step = int(checkpoint_path.split("-")[-1])

        # Redefine the dataset iterator each time to make deterministic.
        dataset_iter = iter(dataset)

        # Load model.
        try:
            model.restore(checkpoint_path, verbose=False)
        except FileNotFoundError:
            logging.warning(
                "No existing checkpoint found in %s, skipping " "checkpoint loading.",
                restore_dir,
            )

        with cProfile.Profile() as pr:
            for batch_idx in range(1, num_batches + 1):
                start_time = time.time()
                logging.info("Predicting batch %d of size %d", batch_idx, batch_size)
                try:
                    with Timer("prediction", logger=None):
                        evaluate_or_sample_batch(
                            model,
                            data_provider,
                            dataset_iter,
                            evaluators,
                            mode,
                            evaluate_and_sample,
                            step,
                        )

                except StopIteration:
                    logging.info("End of dataset.")
                    break

            logging.info(
                "Metrics for batch %i with size %i took %.1f seconds",
                batch_idx,
                batch_size,
                time.time() - start_time,
            )

        # stats = pstats.Stats(pr)
        # stats.sort_stats("cumtime")
        # stats.print_stats(0.1)  # Print the top 10%
        # stats.print_stats(r"/cluster/home/vvolhejn/(thesis|ddsp)")
        # stats.dump_stats(os.path.join(save_dir, "profile.prof"))

        logging.info(
            "All %d batches in checkpoint took %.1f seconds",
            num_batches,
            Timer.timers["prediction"],
        )

        logging.info(Timer.timers)
        for timer_name in Timer.timers:
            logging.info(
                f"{timer_name}: {Timer.timers.count(timer_name)}, {Timer.timers.mean(timer_name)}"
            )

        dummy_batch = next(iter(dataset))["audio"]
        batch_sample_length_secs = (
            dummy_batch.shape[0] * dummy_batch.shape[1] / sample_rate
        )

        columns = {k: np.array(v) for k, v in Timer.timers._timings.items()}
        # columns["real_time_factor"] = columns["prediction"] / batch_sample_length_secs

        # Add an ordering
        columns_l = list(columns.items())
        values = np.array([v for k, v in columns_l]).T

        table = wandb.Table(
            data=[[k, v.mean(), v.std()] for k, v in columns_l],
            # data=[[s, s / batch_sample_length_secs] for s in prediction_times],
            columns=["part_name", "mean", "std"],
            # [k for k, v in columns_l]
        )
        fig = plot_times_hierarchy()

        wandb.log(
            {
                "prediction_time_secs": Timer.timers.mean("prediction"),
                "real_time_factor": Timer.timers.mean("prediction")
                / batch_sample_length_secs,
                "prediction_times_histogram": wandb.plot.histogram(
                    table, "prediction", title="Prediction Time (seconds)"
                ),
                "real_time_factors_histogram": wandb.plot.histogram(
                    table, "real_time_factor", title="Real-Time factor"
                ),
                "time_distribution": wandb.plot.bar(
                    table, "part_name", "mean", title="Time distribution"
                ),
                "time_hierarchy": fig,
            }
        )

        if mode == "eval" or evaluate_and_sample:
            for evaluator in evaluators:
                evaluator.flush(step)

        summary_writer.flush()

    return latest_losses


def evaluate_or_sample_batch(
    model, data_provider, dataset_iter, evaluators, mode, evaluate_and_sample, step
):
    batch = next(dataset_iter)

    if isinstance(data_provider, data.SyntheticNotes):
        batch["audio"] = model.generate_synthetic_audio(batch)
        batch["f0_confidence"] = tf.ones_like(batch["f0_hz"])[:, :, 0]
        batch["loudness_db"] = ddsp.spectral_ops.compute_loudness(batch["audio"])

    # TODO: Find a way to add losses with training=False.

    outputs, losses = model(batch, return_losses=True, training=False)

    outputs["audio_gen"] = model.get_audio_from_outputs(outputs)
    for evaluator in evaluators:
        if mode == "eval" or evaluate_and_sample:
            evaluator.evaluate(batch, outputs, losses)
        if mode == "sample" or evaluate_and_sample:
            evaluator.sample(batch, outputs, step)


def plot_times_hierarchy():
    timers = dict(Timer.timers.copy())
    timers["Autoencoder"] = timers["prediction"]
    del timers["prediction"]

    items = list(Timer.timers.items())
    names = [k.split(".")[-1] for k, v in items]
    values = [v for k, v in items]

    def get_parent(name):
        return ".".join(name.split(".")[:-1])

    parents = [get_parent(name) for name in names]
    data = {"name": names, "parent": parents, "value": values}

    fig = px.icicle(
        data,
        names="name",
        parents="parent",
        values="value",
        branchvalues="total",
    )

    return fig
