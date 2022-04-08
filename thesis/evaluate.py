"""
Based on DDSP's eval_utils, but the requirements are different (we care a lot about
speed) so it made sense to create a separate module.
"""

import os
import time
from typing import Dict

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
from thesis.timbre_transfer_util import adjust_batch, load_dataset_statistics
from thesis.util import get_today_string
import thesis.newt

@gin.configurable
def nas_evaluate(
    data_provider,
    model: ddsp.training.models.Autoencoder,
    evaluator_classes=None,
    mode="eval",
    save_dir="/tmp/ddsp.gin/training",
    restore_dir="",
    batch_size=1,
    num_batches=2,
    evaluate_and_sample=False,
    flag_values_dict=None,
    cache_newt_waveshapers=True,
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
    # checkpoint_path = (
    #     "/Users/vaclav/prog/thesis/data/models/0323-halfrave-1/ckpt-100000"
    # )

    assert checkpoint_path is not None, f"No checkpoint found in {restore_dir}"
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
        name=f"{get_today_string()}-eval-{os.path.basename(save_dir)}",
        sync_tensorboard=True,
        config=flag_values_dict,
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

        if cache_newt_waveshapers:
            thesis.newt.cache_waveshapers_if_possible(model)
        else:
            logging.info("Caching of NEWT waveshapers is disabled.")

        for batch_idx in range(1, num_batches + 1):
            logging.info("Predicting batch %d of size %d", batch_idx, batch_size)
            try:
                batch = next(dataset_iter)
                evaluate_or_sample_batch(
                    model,
                    data_provider,
                    batch,
                    evaluators,
                    mode,
                    evaluate_and_sample,
                    step,
                )

            except StopIteration:
                logging.info("End of dataset.")
                break

        logging.info(
            "All %d batches in checkpoint took %.1f seconds",
            num_batches,
            Timer.timers["Autoencoder"],
        )

        log_timing_info(dataset, sample_rate)

        sample_timbre_transfer(model)

        if mode == "eval" or evaluate_and_sample:
            for evaluator in evaluators:
                evaluator.flush(step)

        summary_writer.flush()

    return latest_losses


@gin.configurable
def evaluate_or_sample_batch(
    model,
    data_provider,
    batch,
    evaluators,
    mode,
    evaluate_and_sample,
    step,
):
    if isinstance(data_provider, data.SyntheticNotes):
        batch["audio"] = model.generate_synthetic_audio(batch)
        batch["f0_confidence"] = tf.ones_like(batch["f0_hz"])[:, :, 0]
        batch["loudness_db"] = ddsp.spectral_ops.compute_loudness(batch["audio"])

    # Delete the original keys to be sure we're computing them here. This is so that
    # pitch estimation is included in the timing information.
    # `F0LoudnessPreprocessor.compute_f0` needs to be set to True in Gin for models
    # that use pitch info to work.
    # The configurable `compute_f0` can be used to select the CREPE model size
    # and other parameters.
    batch["f0_hz"] = None
    batch["f0_confidence"] = None

    with Timer("Autoencoder", logger=None):
        outputs, losses = model(batch, return_losses=True, training=False)

    outputs["audio_gen"] = model.get_audio_from_outputs(outputs)
    for evaluator in evaluators:
        if mode == "eval" or evaluate_and_sample:
            evaluator.evaluate(batch, outputs, losses)
        if mode == "sample" or evaluate_and_sample:
            evaluator.sample(batch, outputs, step)


def log_timing_info(dataset, sample_rate):
    dummy_batch = next(iter(dataset))["audio"]
    batch_sample_length_secs = dummy_batch.shape[0] * dummy_batch.shape[1] / sample_rate

    table = wandb.Table(
        data=[
            [k, Timer.timers.mean(k), Timer.timers.stdev(k)]
            for k in Timer.timers._timings.keys()
        ],
        columns=["model_part", "mean", "std"],
    )
    time_hierarchy_plot = plot_time_hierarchy(
        {k: Timer.timers.mean(k) for k in Timer.timers._timings.keys()}
    )

    wandb.log(
        {
            "prediction_time_secs": Timer.timers.mean("Autoencoder"),
            "real_time_factor": Timer.timers.mean("Autoencoder")
            / batch_sample_length_secs,
            "time_distribution": wandb.plot.bar(
                table, "model_part", "mean", title="Time distribution"
            ),
            "time_hierarchy": time_hierarchy_plot,
        }
    )


def plot_time_hierarchy(data: Dict[str, float]):
    items = list(data.items())
    names = [k.split(".")[-1] for k, v in items]
    values = [v for k, v in items]

    def get_parent(name):
        return ".".join(name.split(".")[:-1])

    parents = [get_parent(k) for k, v in items]
    data = {"name": names, "parent": parents, "value": values}

    fig = px.icicle(
        data,
        names="name",
        parents="parent",
        values="value",
        branchvalues="total",
    )

    return fig


def sample_timbre_transfer(model):
    data_provider = ddsp.training.data.TFRecordProvider(
        file_pattern="/cluster/home/vvolhejn/datasets/transfer2/transfer2.tfrecord*"
    )

    dataset = data_provider.get_batch(batch_size=1, shuffle=True, repeats=1)

    dataset_stats = load_dataset_statistics(
        "/cluster/home/vvolhejn/datasets/violin/dataset_statistics_violin.pkl"
    )

    for i, batch in enumerate(dataset):
        if i % 5 != 0:
            continue

        # logging.info(f'before {batch}')
        max_loudness_before = batch["loudness_db"].max()
        batch = adjust_batch(batch, dataset_stats)
        max_loudness_after = batch["loudness_db"].max()

        logging.info(
            f"Loudness adjusted from {max_loudness_before:.2f}"
            f" to {max_loudness_after:.2f}."
        )

        if batch["mask_on"].mean() < 0.5:
            # An mostly silent segment.
            # A completely silent one can be detected by `batch["loudness_db"].max() == -80`
            continue

        outputs, losses = model(batch, return_losses=True, training=False)
        sample_rate = 16000

        audio_both = tf.concat(
            [
                outputs["audio_synth"][:1],
                outputs["audio"][:1],
            ],
            axis=1,
        )

        ddsp.training.summaries.audio_summary(
            audio_both, step=i, sample_rate=sample_rate, name="timbre_transfer"
        )

        logging.info(f"Predicted sample {i+1}")

        # if i == 80:
        #     break
