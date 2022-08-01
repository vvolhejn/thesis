"""
Based on DDSP's eval_utils, but the requirements are different (we care a lot about
speed) so it made sense to create a separate module.
"""

import os
import time
from typing import Dict

import pandas as pd
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
from thesis.timbre_transfer_util import adjust_batch, load_dataset_statistics, get_lufs
from thesis.util import get_today_string
import thesis.newt
import thesis.runtimes


@gin.configurable
def nas_evaluate(
    data_provider,
    model: ddsp.training.models.Autoencoder,
    evaluator_classes=None,
    mode="eval",
    save_dir="/tmp/ddsp.gin/training",
    restore_dir="",
    batch_size=1,
    num_batches=100,
    evaluate_and_sample=False,
    flag_values_dict=None,
    cache_newt_waveshapers=True,
    use_runtime=False,
    quantization=False,
    num_calibration_batches=100,
    calibration_method="minmax",
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
    save_dir = save_dir.rstrip("/")

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

    name_without_date = f"eval-{os.path.basename(save_dir)}"
    if use_runtime:
        name_without_date += "-rt"
        if quantization:
            name_without_date += "q"

    name = f"{get_today_string()}-{name_without_date}"

    wandb.init(
        project="nas-evaluation",
        entity="neural-audio-synthesis-thesis",
        name=name,
        sync_tensorboard=True,
        config={
            **flag_values_dict,
            "num_batches": num_batches,
            "num_calibration_batches": num_calibration_batches,
            "calibration_method": calibration_method,
            "operative_config": ddsp.training.train_util.config_string_to_markdown(
                gin.operative_config_str()
            ),
        },
        dir="/cluster/scratch/vvolhejn/wandb",
        tags=["eval"],
    )

    train_data_provider = data_provider
    data_provider = data_provider.get_evaluation_set()

    # Get the dataset.
    dataset = data_provider.get_batch(batch_size=batch_size, shuffle=False, repeats=1)
    # Set number of batches.
    # If num_batches <1 set it to a huge value to go through the whole dataset
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

        if use_runtime:
            # Calibrate on training data.
            use_runtime_for_decoder(
                model,
                train_data_provider,
                quantization=quantization,
                num_calibration_batches=num_calibration_batches,
            )
        else:
            assert (
                not quantization
            ), "Quantization cannot be applied without --use-runtime"

        # Clear timers from when we might have run the model in initialization
        Timer.timers.clear()

        logging.info(
            f"Predicting {num_batches if num_batches < 1e9 else 'all'} batches."
        )
        for batch_idx in range(1, num_batches + 1):
            # logging.info("Predicting batch %d of size %d", batch_idx, batch_size)
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

        log_timing_info(dataset, sample_rate, name=name_without_date)

        timbre_transfer_data_provider = ddsp.training.data.WandbTFRecordProvider(
            "neural-audio-synthesis-thesis/transfer4:v0"
        )

        sample_timbre_transfer(
            model,
            timbre_transfer_data_provider,
            name="timbre_transfer",
            every_nth=5,
        )

        sample_timbre_transfer(
            model, data_provider, name="audio_both", every_nth=10, adjust=False
        )

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
    # To be set via gin
    recompute_f0=True,
):
    if isinstance(data_provider, data.SyntheticNotes):
        batch["audio"] = model.generate_synthetic_audio(batch)
        batch["f0_confidence"] = tf.ones_like(batch["f0_hz"])[:, :, 0]
        batch["loudness_db"] = ddsp.spectral_ops.compute_loudness(batch["audio"])

    if recompute_f0:
        # Delete the original keys to be sure we're computing them here. This is so that
        # pitch estimation is included in the timing information.
        # `F0LoudnessPreprocessor.compute_f0` needs to be set to True in Gin for models
        # that use pitch info to work.
        # The configurable `compute_f0` can be used to select the CREPE model size
        # and other parameters.
        batch["f0_hz"] = None
        batch["f0_confidence"] = None

    if isinstance(
        model.preprocessor, ddsp.training.preprocessing.F0LoudnessPreprocessor
    ):
        model.preprocessor.compute_f0 = recompute_f0

    with Timer("Autoencoder", logger=None):
        outputs, losses = model(batch, return_losses=True, training=False)

    outputs["audio_gen"] = model.get_audio_from_outputs(outputs)
    for evaluator in evaluators:
        if mode == "eval" or evaluate_and_sample:
            evaluator.evaluate(batch, outputs, losses)
        if mode == "sample" or evaluate_and_sample:
            evaluator.sample(batch, outputs, step)


def log_timing_info(dataset, sample_rate, name):
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
            "decoder_real_time_factor": Timer.timers.mean("Autoencoder.decoder")
            / batch_sample_length_secs,
            "time_distribution": wandb.plot.bar(
                table, "model_part", "mean", title="Time distribution"
            ),
            "time_hierarchy": time_hierarchy_plot,
        }
    )

    df = pd.DataFrame(
        index=pd.RangeIndex(len(Timer.timers._timings["Autoencoder"])),
        columns=Timer.timers._timings.keys(),
    )
    for k, v in Timer.timers._timings.items():
        df.loc[:, k] = v

    path = os.path.join("/tmp", f"{name}.csv")
    df.to_csv(path)
    artifact = wandb.Artifact(name, type="timing")
    artifact.add_file(path)
    wandb.run.log_artifact(artifact)


def plot_time_hierarchy(data: Dict[str, float]):
    items = list(data.items())
    logging.info(items)
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

    # Do not sort by size, instead preserve order in which data is given
    fig.update_traces(sort=False)

    return fig


def sample_timbre_transfer(model, data_provider, name, every_nth=5, adjust=True):
    dataset = data_provider.get_batch(batch_size=1, shuffle=False, repeats=1)

    artifact = wandb.run.use_artifact(
        "neural-audio-synthesis-thesis/violin_dataset_statistics:latest",
        type="dataset",
    )
    artifact_dir = artifact.download()

    dataset_stats = load_dataset_statistics(
        os.path.join(artifact_dir, "dataset_statistics.pkl")
    )

    logging.info(f"Timbre transfer")

    for i, batch in enumerate(dataset):
        if i % every_nth != 0:
            continue

        if adjust:
            batch = adjust_batch(batch, dataset_stats, lufs_normalization=-20)

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
            audio_both, step=i, sample_rate=sample_rate, name=name
        )

        logging.info(f"Predicted sample {i+1}")

        # if i == 80:
        #     break


@gin.configurable
def use_runtime_for_decoder(
    model,
    data_provider,
    quantization=False,
    runtime=None,
    num_calibration_batches=100,
    calibration_method="minmax",
):
    import onnxruntime.quantization as ortq

    calibration_method_d = {
        "minmax": ortq.CalibrationMethod.MinMax,
        "entropy": ortq.CalibrationMethod.Entropy,
        "percentile": ortq.CalibrationMethod.Percentile,
    }
    assert calibration_method in calibration_method_d
    # From string to ORT's enum
    calibration_method = calibration_method_d[calibration_method]

    if runtime is None:
        runtime = thesis.runtimes.ONNXRuntime(
            quantization_mode="static_qdq" if quantization else "off"
        )

    dataset = data_provider.get_batch(
        batch_size=1,
        shuffle=True,
        repeats=1,
    )

    dataset_iter = iter(dataset)

    def get_batch_fn():
        batch = next(dataset_iter)

        # Only run the encoder to save time
        features = model.encode(batch, training=False)

        stacked_features = tf.concat(
            [features["ld_scaled"], features["f0_scaled"]], axis=-1
        )
        return np.array(stacked_features)

    # Take one batch to determine the input shape to the decoder
    batch = get_batch_fn()

    # From a Keras Layer to a Model
    decoder = tf.keras.Sequential(
        [
            # 2 channels because we have f0 and loudness
            tf.keras.layers.Input(batch.shape[1:]),
            model.decoder.dilated_conv_stack,
        ]
    )

    runtime.convert(
        decoder,
        get_batch_fn=get_batch_fn,
        n_calibration_batches=num_calibration_batches,
        calibration_method=calibration_method,
    )
    logging.info(f"Converted decoder and saved to {runtime.save_path}")

    def alternative_decoder(inputs):
        # Inputs is a list typically containing ld_scaled and f0_scaled tensors
        #     stacked_features = tf.concat(inputs, axis=-1)

        output_numpy = runtime.run(np.array(inputs))
        return tf.convert_to_tensor(output_numpy)

    model.decoder.dilated_conv_stack = alternative_decoder
