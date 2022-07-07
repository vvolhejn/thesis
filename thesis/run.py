""" Based on ddsp_run.py. """


import os
import time

from absl import app
from absl import flags
from absl import logging
from ddsp.training import cloud
from ddsp.training import models
from ddsp.training import train_util
from ddsp.training import trainers
import gin
import pkg_resources
import tensorflow as tf

from thesis import evaluate
from thesis.util import get_today_string

gfile = tf.io.gfile
FLAGS = flags.FLAGS

# Program flags.
flags.DEFINE_enum(
    "mode",
    "train",
    ["train", "eval"],
    "Whether to train, evaluate, or sample from the model.",
)
flags.DEFINE_string(
    "save_dir",
    "/tmp/ddsp",
    "Path where checkpoints and summary events will be saved "
    "during training and evaluation.",
)
flags.DEFINE_string(
    "restore_dir",
    "",
    "Path from which checkpoints will be restored before "
    "training. Can be different than the save_dir.",
)
flags.DEFINE_string("tpu", "", "Address of the TPU. No TPU if left blank.")
flags.DEFINE_string(
    "cluster_config",
    "",
    "Worker-specific JSON string for multiworker setup. "
    "For more information see train_util.get_strategy().",
)
flags.DEFINE_boolean(
    "allow_memory_growth",
    False,
    "Whether to grow the GPU memory usage as is needed by the "
    "process. Prevents crashes on GPUs with smaller memory.",
)
flags.DEFINE_boolean(
    "hypertune",
    False,
    "Enable metric reporting for hyperparameter tuning, such "
    "as on Google Cloud AI-Platform.",
)
flags.DEFINE_float(
    "early_stop_loss_value",
    None,
    "Stops training early when the `total_loss` reaches below "
    "this value during training.",
)

# Gin config flags.
flags.DEFINE_multi_string("gin_search_path", [], "Additional gin file search paths.")
flags.DEFINE_multi_string(
    "gin_file",
    [],
    "List of paths to the config files. If file "
    "in gstorage bucket specify whole gstorage path: "
    "gs://bucket-name/dir/in/bucket/file.gin.",
)
flags.DEFINE_multi_string(
    "gin_param", [], "Newline separated list of Gin parameter bindings."
)

# Evaluation/sampling specific flags.
flags.DEFINE_boolean("run_once", False, "Whether evaluation will run once.")
flags.DEFINE_integer(
    "initial_delay_secs", None, "Time to wait before evaluation starts"
)
flags.DEFINE_boolean(
    "use_runtime",
    False,
    "Use a specialized runtime library to run the DilatedConvDecoder when evaluating",
)

# W&B flags
flags.DEFINE_string(
    "wandb_group",
    get_today_string(),
    "An identifier that can be used in W&B to group runs.",
)

# GIN_PATH = pkg_resources.resource_filename(__name__, "gin")
DDSP_GIN_PATH = pkg_resources.resource_filename("ddsp.training", "gin")


def delay_start():
    """Optionally delay the start of the run."""
    delay_time = FLAGS.initial_delay_secs
    if delay_time:
        logging.info("Waiting for %i second(s)", delay_time)
        time.sleep(delay_time)


def parse_gin(restore_dir, require_operative_config=False):
    """Parse gin config from --gin_file, --gin_param, and the model directory."""
    # Enable parsing gin files on Google Cloud.
    gin.config.register_file_reader(tf.io.gfile.GFile, tf.io.gfile.exists)
    # Add user folders to the gin search path.
    for gin_search_path in [DDSP_GIN_PATH] + FLAGS.gin_search_path:
        gin.add_config_file_search_path(gin_search_path)

    # Parse gin configs, later calls override earlier ones.
    with gin.unlock_config():
        # Optimization defaults.
        use_tpu = bool(FLAGS.tpu)
        # This is a DDSP gin
        opt_default = "base.gin" if not use_tpu else "base_tpu.gin"
        gin.parse_config_file(os.path.join("optimization", opt_default))
        # This is our gin
        eval_default = "evaluation.gin"
        gin.parse_config_file(eval_default)

        # Load operative_config if it exists (model has already trained).
        try:
            operative_config = train_util.get_latest_operative_config(restore_dir)
            logging.info("Using operative config: %s", operative_config)
            operative_config = cloud.make_file_paths_local(
                operative_config, DDSP_GIN_PATH
            )
            gin.parse_config_file(operative_config, skip_unknown=True)
        except FileNotFoundError as e:
            if require_operative_config:
                logging.error("Operative config not found in %s", restore_dir)
                raise e
            else:
                logging.info("Operative config not found in %s", restore_dir)

        # User gin config and user hyperparameters from flags.
        gin_file = cloud.make_file_paths_local(FLAGS.gin_file, DDSP_GIN_PATH)
        gin.parse_config_files_and_bindings(
            gin_file, FLAGS.gin_param, skip_unknown=True
        )


def allow_memory_growth():
    """Sets the GPUs to grow the memory usage as is needed by the process."""
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs.
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized.
            print(e)


def main(unused_argv):
    """Parse gin config and run ddsp training, evaluation, or sampling."""
    restore_dir = os.path.expanduser(FLAGS.restore_dir)
    save_dir = os.path.expanduser(FLAGS.save_dir)
    # If no separate restore directory is given, use the save directory.
    restore_dir = save_dir if not restore_dir else restore_dir
    logging.info("Restore Dir: %s", restore_dir)
    logging.info("Save Dir: %s", save_dir)

    gfile.makedirs(restore_dir)  # Only makes dirs if they don't exist.
    parse_gin(restore_dir, require_operative_config=(FLAGS.mode == "eval"))
    logging.info("Operative Gin Config:\n%s", gin.config.config_str())

    if FLAGS.allow_memory_growth:
        allow_memory_growth()

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("WARNING: no GPUs detected by TensorFlow!")
    else:
        print(f"Found GPU(s): {gpus}")

    # Training.
    if FLAGS.mode == "train":
        strategy = train_util.get_strategy(
            tpu=FLAGS.tpu, cluster_config=FLAGS.cluster_config
        )
        with strategy.scope():
            model = models.get_model()
            trainer = trainers.get_trainer_class()(model, strategy)

        train_util.train(
            data_provider=gin.REQUIRED,
            trainer=trainer,
            save_dir=save_dir,
            restore_dir=restore_dir,
            early_stop_loss_value=FLAGS.early_stop_loss_value,
            report_loss_to_hypertune=FLAGS.hypertune,
            flag_values_dict=FLAGS.flag_values_dict(),
        )

    # Evaluation.
    elif FLAGS.mode == "eval":
        model = models.get_model()
        evaluate.nas_evaluate(
            data_provider=gin.REQUIRED,
            model=model,
            mode=FLAGS.mode,
            save_dir=save_dir,
            restore_dir=restore_dir,
            flag_values_dict=FLAGS.flag_values_dict(),
            use_runtime=FLAGS.use_runtime,
        )


def console_entry_point():
    """From pip installed script."""
    app.run(main)


if __name__ == "__main__":
    console_entry_point()
